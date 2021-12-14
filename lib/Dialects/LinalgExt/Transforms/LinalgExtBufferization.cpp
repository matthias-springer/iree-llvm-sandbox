//===-- LinalgExtBufferization.cpp - Linalg Extension bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/BuiltinOps.h>

#include "Dialects/LinalgExt/LinalgExtBufferization.h"
#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

using linalg::comprehensive_bufferize::BufferizationAliasInfo;
using linalg::comprehensive_bufferize::BufferizableOpInterface;
using linalg::comprehensive_bufferize::BufferizationState;
using linalg::comprehensive_bufferize::BufferRelation;

namespace linalg_ext {

static OpOperand *getInsertionDest(InParallelOp inParallelOp) {
  Operation *terminator = &inParallelOp.region().front().back();
  auto performConcOp = dyn_cast<PerformConcurrentlyOp>(terminator);
  assert(performConcOp && "expected PerformConcurrentlyOp as terminator");

  // TODO: What if there are multiple ops in the region?
  Operation *concOp = &performConcOp.region().front().front();
  auto parallelInsert = dyn_cast<ParallelInsertSliceOp>(concOp);
  assert(parallelInsert && "expected ParallelInsertSliceOp");
  return &parallelInsert->getOpOperand(1) /*dest*/;
}

struct InParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<InParallelOpInterface,
                                                    InParallelOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    auto inParallelOp = cast<InParallelOp>(op);
    return {getInsertionDest(inParallelOp)};
  }

  bool mustBufferizeInPlace(Operation *op, OpResult opResult) const {
    return true;
  }

  bool isMemoryWrite(Operation *op, OpResult opResult) const {
    // TODO: Return true only if there is actually a write inside the region.
    return true;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);

    auto inParallelOp = cast<InParallelOp>(op);
    Value insertionDest = getInsertionDest(inParallelOp)->get();

    TypeRange newResultTypes;
    auto newInParallelOp = b.create<InParallelOp>(
      inParallelOp.getLoc(), newResultTypes, inParallelOp.num_threads());
    newInParallelOp.getBody()->getTerminator()->erase();

    // Move over block of the old op.
    IRRewriter rewriter(op->getContext());
    rewriter.mergeBlocks(inParallelOp.getBody(), newInParallelOp.getBody(),
                         {newInParallelOp.getBody()->getArgument(0)});
    newInParallelOp.dump();

    if (failed(
        linalg::comprehensive_bufferize::bufferize(
            newInParallelOp.getBody(), state)))
      return failure();


    // TODO: This is wrong. Should use getResultBuffer here. But ops without
    // OpOperands (that can just yield anything that's in scope) cannot
    // bufferize out-of-place at the moment.
    // TODO: Support multiple results.
    state.mapBuffer(inParallelOp.results()[0], state.lookupBuffer(insertionDest));

    return success();
  }
};

struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<ParallelInsertSliceOpInterface,
                                                    ParallelInsertSliceOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/
        ? op->getOpResult(0) : OpResult();
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    return success();

    // TODO: Do something with the op.
    op->dump();
    state.markOpObsolete(op);
    return success();
  }  
};
} // namespace linalg_ext
} // namespace mlir

void mlir::linalg_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<InParallelOp, InParallelOpInterface>();
  // TODO: Following line causes a crash. (Later in the code...)
  registry.addOpInterface<linalg_ext::ParallelInsertSliceOp,
                          linalg_ext::ParallelInsertSliceOpInterface>();
}
