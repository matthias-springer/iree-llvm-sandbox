from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import multiprocessing as mp
import os
import signal
import sys
import threading
import typing as tp

from ..core.harness import *
from ..core.nevergrad_tuner_utils import NGSchedulerInterface
from ..core.problem_definition import ProblemDefinition
from ..core.utils import compute_quantiles


class IPCState:
  """State shared across parent and children processes in a MP run.
  
  All contained information must "pickle" (i.e. serialize across processes).
  """

  def __init__(self,
               success: bool,
               throughputs: tp.Sequence[float],
               problem=None):
    self.success = success
    self.throughputs = throughputs
    self.problem = problem


class NGEvaluation:
  """Handle to a nevergrad evaluation."""

  def __init__(self, proposal, problem_instance):
    self.proposal = proposal
    self.problem_instance = problem_instance


class NGMPEvaluation(NGEvaluation):
  """Handle to a multiprocess nevergrad evaluation."""

  def __init__(self, proposal, problem_instance, process, ipc_dict, time_left):
    super().__init__(proposal, problem_instance)
    self.process = process
    self.ipc_dict = ipc_dict
    self.time_left = time_left
    self.joined_with_root = False

  def ipc_state(self):
    return self.ipc_dict['result'] if 'result' in self.ipc_dict else None


def compile_and_run_checked_mp(problem: ProblemInstance, \
                               scheduler: NGSchedulerInterface,
                               proposal,
                               n_iters: int):
  """Entry point to compile and run while catching and reporting exceptions.

  This can run in interruptible multiprocess mode.
  ipc_dict must be provided, and it is used to return information across the
  root / children process boundary:
    - 'throughputs': the measured throughputs.
    - 'success': the return status.
  """
  try:

    # Construct the schedule and save the module in case we need to replay later.
    def schedule_and_save(module):
      scheduler.schedule(module, proposal)
      # TODO: save and report on error.

    f = io.StringIO()
    with redirect_stdout(f):
      problem.compile_with_schedule_builder(
          entry_point_name=scheduler.entry_point_name,
          fun_to_benchmark_name=scheduler.fun_to_benchmark_name,
          compile_time_problem_sizes_dict= \
            scheduler.build_compile_time_problem_sizes(),
          schedule_builder=schedule_and_save)

      throughputs = problem.run(
          n_iters=n_iters,
          entry_point_name=scheduler.entry_point_name,
          runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict)

    # TODO: redirect to a file if we want this information.
    f.flush()

    return proposal, throughputs
  except Exception as e:
    import traceback
    traceback.print_exc()
    # TODO: save to replay errors.
    print(e)
    return proposal, None


def cpu_count():
  return len(os.sched_getaffinity(0))

def ask_and_fork_process(mp_manager: mp.Manager, \
                         problem_definition: ProblemDefinition,
                         problem_types: tp.Sequence[np.dtype],
                         ng_mp_evaluations: tp.Sequence[NGMPEvaluation],
                         evaluation_slot_idx: int,
                         scheduler: NGSchedulerInterface,
                         parsed_args):
  """Ask for the next proposal and fork its evaluation in a new process"""

  proposal = scheduler.optimizer.ask()

  # Create problem instance, which holds the compiled module and the
  # ExecutionEngine.
  problem_instance = ProblemInstance(problem_definition, problem_types)

  # The optimizer may chose to ignore our constraints.
  # Override this, do not evaluate and just give it max cost.
  if not scheduler.validate_proposal(proposal):
    ng_mp_evaluations[evaluation_slot_idx] = NGMPEvaluation(
        proposal=proposal,
        problem_instances=None,
        process=None,
        ipc_state=None,
        time_left=parsed_args.timeout_per_compilation)
    return

  # Start process that compiles and runs.
  ipc_dict = mp_manager.dict()
  p = mp.Process(target=compile_and_run_checked_mp,
                 args=[
                     problem_instance, scheduler, proposal, parsed_args.n_iters,
                     ipc_dict
                 ])
  p.start()
  # Best effort pin process in a round-robin fashion.
  # This is noisy so suppress its name.
  f = io.StringIO()
  with redirect_stdout(f):
    os.system(
        f'taskset -p -c {evaluation_slot_idx % cpu_count()} {p.pid} > /dev/null 2>&1'
    )
  # Append NGMPEvaluation. After synchronization results will be available
  # in ipc_dict['result'].
  ng_mp_evaluations[evaluation_slot_idx] = NGMPEvaluation(
      proposal=proposal,
      problem_instance=problem_instance,
      process=p,
      ipc_dict=ipc_dict,
      time_left=parsed_args.timeout_per_compilation)


def tell_joined_process(ng_mp_evaluations: tp.Sequence[NGMPEvaluation], \
                        evaluation_slot_idx: int,
                        scheduler: NGSchedulerInterface,
                        # TODO: extract info from final recommendation instead
                        # of an auxiliary `throughputs` list.
                        throughputs: tp.Sequence[float],
                        parsed_args):
  """Tell the result for the proposal from a joined evaluation process."""

  ng_mp_evaluation = ng_mp_evaluations[evaluation_slot_idx]
  ipc_state = ng_mp_evaluation.ipc_state()

  if not ipc_state.success:
    scheduler.optimizer.tell(ng_mp_evaluation.proposal, 1)
    return 0

  process_throughputs = ipc_state.throughputs[parsed_args.metric_to_measure]
  # Calculate the relative distance to peak: invert the throughput @90%
  # (i.e. 6th computed quantile).
  # Lower is better.
  # This matches the optimization process which is a minimization.
  throughput = compute_quantiles(process_throughputs)[6]
  relative_error = \
    (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
  scheduler.optimizer.tell(ng_mp_evaluation.proposal, relative_error)
  throughputs.append(throughput)
  return throughput


def finalize_parallel_search(scheduler: NGSchedulerInterface, \
                             throughputs: tp.Sequence[float],
                             parsed_args):
  """Report and save the best proposal after search finished."""

  # TODO: better handling of result saving, aggregation etc etc.
  final_module_filename = None
  if parsed_args.output_dir is not None:
    final_module_filename = f'{parsed_args.output_dir}/module.mlir'
  else:
    final_module_filename = '/tmp/module.mlir'

  recommendation = scheduler.optimizer.recommend()
  # TODO: extract information from saved and draw some graphs
  # TODO: extract info from final recommendation instead of an auxiliary `throughputs` list
  throughputs.sort()
  best = int(throughputs[-1])
  print(
      f'Best solution: {best} GUnits/s (peak is {parsed_args.machine_peak} GUnits/s)'
  )
  scheduler.save_proposal_as_module(proposal=recommendation,
                                    module_save_filename=final_module_filename,
                                    benefit=best)


################################################################################
### Multiprocess optimization loops.
################################################################################

def process_proposal(scheduler: NGSchedulerInterface, proposal, problem_definition: ProblemDefinition, n_iters):
  print("process_proposal")
  # Create problem instance, which holds the compiled module and the
  # ExecutionEngine.
  problem_types = [np.float32] * 3
  problem_instance = ProblemInstance(problem_definition, problem_types)
  return proposal, compile_and_run_checked_mp(problem_instance, scheduler, proposal, n_iters)

global interrupted

def async_optim_loop(problem_definition: ProblemDefinition, \
                     scheduler: NGSchedulerInterface,
                     parsed_args):
  print("running with #processes: " + str(parsed_args.num_compilation_processes))
  pool = mp.Pool(parsed_args.num_compilation_processes)
  enqueued = 0
  scheduler_lock = threading.Lock()

  def process_evaluation_result(result):
    nonlocal enqueued, pool, scheduler_lock
    print("process_evaluation_result")
    with scheduler_lock:
      if enqueued < parsed_args.search_budget:
        enqueue_proposal()

    proposal = result[0]
    throughputs = result[1]
    if not throughputs:
      scheduler.optimizer.tell(proposal, 1)
      return

    process_throughputs = throughputs[parsed_args.metric_to_measure]
    # Calculate the relative distance to peak: invert the throughput @90%
    # (i.e. 6th computed quantile).
    # Lower is better.
    # This matches the optimization process which is a minimization.
    throughput = compute_quantiles(process_throughputs)[6]
    relative_error = \
      (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
    scheduler.optimizer.tell(proposal, relative_error)

    print("Throughput of proposal: " + str(throughput))

  def enqueue_proposal(): 
    nonlocal enqueued, pool, scheduler_lock
    #def process_proposal():
    #  print("ASD!")
    print("enqueue_proposal")
    proposal = scheduler.optimizer.ask()
    #(scheduler, proposal, problem_definition, parsed_args.n_iters),
    #pool.apply_async(process_proposal, callback=process_evaluation_result)
    pool.apply(process_proposal, (scheduler, proposal, problem_definition, parsed_args.n_iters))
    enqueued = enqueued + 1
    print("enqueued")

  with scheduler_lock:
    for _ in range(2 * parsed_args.num_compilation_processes):
      enqueue_proposal()
  
  while 1:
    with scheduler_lock:
      if enqueued == parsed_args.search_budget:
        print("waiting for stop")
        pool.close()
        pool.join()
        break
    time.sleep(1)

