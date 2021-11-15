# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'depthwise_conv_1d_nwc_wc'
op_name = 'linalg.depthwise_conv_1d_nwc_wc'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    # LoweringOnlyExpert([], print_ir_after_all=False),
    SingleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        #      N  W   C  KW
        sizes=[1, 4, 16, 3],
        interchange=[],
        peel=[],
        pad=False,
        pack_paddings=[],
        hoist_paddings=[],
        print_ir_after_all=False)
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  #   N   W   C  KW   st  dil
  problem_size_list = [\
     [8, 16, 32,  3, [1],  [1]], \
     [8, 16, 32,  3, [1],  [2]], \
     [8, 16, 32,  3, [2],  [1]], \
     [8, 16, 32,  3, [2],  [2]],  \
     [8, 16, 32,  3, [2],  [3]],  \
     [8, 16, 32,  3, [3],  [2]]  \
  ]
  for np_types in [[np.float32, np.float32, np.float32]]:
    for problem_sizes in problem_size_list:
      compile_time_problem_sizes_dict = {
          k: v for k, v in zip(keys, problem_sizes)
      }
      runtime_problem_sizes_dict = compile_time_problem_sizes_dict
      # Init printing.
      print(
          f'\n###############################################################\n'
          f'Problem size {compile_time_problem_sizes_dict}\n'
          f'Problem types {np_types}')
      for expert in all_experts:
        problem = ProblemInstance(
            problem_definition=DepthwiseConvolutionProblem(
                'NWC',
                'WC',
                strides=compile_time_problem_sizes_dict['strides'],
                dilations=compile_time_problem_sizes_dict['dilations']),
            problem_sizes_keys=keys,
            np_types=np_types)
        assert problem.problem_definition.keys() == keys

        problem.compile(
            entry_point_name='main',
            fun_to_benchmark_name=fun_name,
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert,
            # Used to pipe through llvm-mca with mlir LLVM dialect.
            dump_ir_to_file='/tmp/abc.mlir')

        problem.run(
            n_iters=n_iters,
            entry_point_name='main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict,
            # Used to pipe through llvm-mca with the **actual JIT'ed object**.
            dump_obj_to_file='/tmp/abc.o')


if __name__ == '__main__':
  main()