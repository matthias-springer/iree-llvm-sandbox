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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

using linalg::comprehensive_bufferize::BufferizableOpInterface;
using linalg::comprehensive_bufferize::BufferizationAliasInfo;
using linalg::comprehensive_bufferize::BufferizationState;
using linalg::comprehensive_bufferize::BufferRelation;
using linalg::comprehensive_bufferize::replaceOpWithBufferizedValues;
using linalg::comprehensive_bufferize::replaceOpWithNewBufferizedOp;
using linalg::comprehensive_bufferize::getDynamicMemRefType;
using tensor::ExtractSliceOp;

namespace linalg_ext {

static SmallVector<OpOperand *> getInsertionDest(InParallelOp inParallelOp) {
  Operation *terminator = inParallelOp.region().front().getTerminator();
  auto performConcOp = dyn_cast<PerformConcurrentlyOp>(terminator);
  assert(performConcOp && "expected PerformConcurrentlyOp as terminator");

  SmallVector<OpOperand *> result;
  performConcOp.walk([&](ParallelInsertSliceOp insertOp) {
    result.push_back(&insertOp->getOpOperand(1) /*dest*/);
  });

  return result;
}

struct InParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<InParallelOpInterface,
                                                    InParallelOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult,
      const BufferizationState &state) const {
    auto inParallelOp = cast<InParallelOp>(op);
    return {getInsertionDest(inParallelOp)[opResult.getResultNumber()]};
  }

/*
  bool mustBufferizeInPlace(
      Operation *op, OpResult opResult,
      const BufferizationState &state) const {
    return true;
  }
*/

  bool isMemoryWrite(
      Operation *op, OpResult opResult,
      const BufferizationState &state) const {
    // TODO: Return true only if there is actually a write inside the region.
    return true;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    llvm::errs() << "!!! InParallelOp\n";
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    auto inParallelOp = cast<InParallelOp>(op);

    // Create new InParallelOp.
    Block *body = &inParallelOp.region().front();
    SmallVector<Value> newResults;
    for (OpResult opResult : inParallelOp->getOpResults()) {
      SmallVector<OpOperand *> insertDestOperands =
          state.getAliasingOpOperand(opResult);
      assert(insertDestOperands.size() == 1 &&
             "expected exactly one aliasing OpOperand");
      Value buffer = *state.getBuffer(b, *insertDestOperands.front());
      newResults.push_back(buffer);
      Value destTensor = insertDestOperands.front()->get();

      // Replace all uses of the insert dest tensor inside the InParallelOp
      // with the result buffer.
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(body);
      Value toTensorOp = b.create<bufferization::ToTensorOp>(inParallelOp.getLoc(), buffer);
      for (OpOperand &use : destTensor.getUses())
        if (body->findAncestorOpInBlock(*use.getOwner()))
          // This is a use inside the InParallelOp.
          use.set(toTensorOp);
    }
    TypeRange newResultTypes;
    auto newInParallelOp = b.create<InParallelOp>(
        inParallelOp.getLoc(), newResultTypes, inParallelOp.num_threads());

    // Delete terminator.
    newInParallelOp.getBody()->getTerminator()->erase();

    // Move over block contents of the old op.
    IRRewriter rewriter(op->getContext());
    rewriter.mergeBlocks(inParallelOp.getBody(), newInParallelOp.getBody(),
                         {newInParallelOp.getBody()->getArgument(0)});

    // Bufferize terminator.
    auto performConcurrentlyOp = cast<PerformConcurrentlyOp>(
        newInParallelOp.getBody()->getTerminator());
    b.setInsertionPoint(performConcurrentlyOp);
    performConcurrentlyOp.walk([&](ParallelInsertSliceOp insertOp) {
      Type srcType = getDynamicMemRefType(insertOp.source().getType().cast<RankedTensorType>());
      Type destType = getDynamicMemRefType(insertOp.dest().getType().cast<RankedTensorType>());
      auto srcMemref = b.create<bufferization::ToMemrefOp>(insertOp.getLoc(), srcType, insertOp.source());
      auto destMemref = b.create<bufferization::ToMemrefOp>(insertOp.getLoc(), destType, insertOp.dest());
      Value subview = b.create<memref::SubViewOp>(insertOp.getLoc(), destMemref, insertOp.offsets(), insertOp.sizes(), insertOp.strides());
      state.createMemCpy(b, insertOp.getLoc(), srcMemref, subview);
      b.eraseOp(insertOp);
    });

    // Replace the op.
    replaceOpWithBufferizedValues(rewriter, op, newResults);

    newInParallelOp->getParentOp()->dump();
    return success();
  }
};

struct PerformConcurrentlyOpInterface
    : public BufferizableOpInterface::ExternalModel<
          PerformConcurrentlyOpInterface, PerformConcurrentlyOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    llvm_unreachable("op does not have any tensor OpOperands / OpResults");
    return failure();
  }
};

/// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
///
/// This is one particular type of relationship between ops on tensors that
/// reduce to an equivalence on buffers. This should be generalized and
/// exposed as interfaces on the proper types.
static bool
areEquivalentExtractSliceOps(const BufferizationAliasInfo &aliasInfo,
                             ExtractSliceOp st, ParallelInsertSliceOp sti) {
  if (!st || !sti)
    return false;
  if (!aliasInfo.areEquivalentBufferizedValues(st.source(), sti.dest()))
    return false;
  // TODO: ParallelInsertSliceOp should implement ViewLikeInterface
  // TODO: Enable this check again!
  //if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
  //  return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const BufferizationAliasInfo &aliasInfo,
                                      const BufferizationState &state,
                                      Value value, ParallelInsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(aliasInfo, extractOp, insertOp))
        return true;
    return false;
  };

  return llvm::all_of(state.findValueInReverseUseDefChain(value, condition),
                      condition);
}

struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, ParallelInsertSliceOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult,
      const BufferizationState &state) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  OpResult getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/
               // ParallelInsertSliceOp has not results, attempting to get the
               // OpResult form the parent.
               ? op->getParentOfType<InParallelOp>()->getResult(0)
               : OpResult();
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    llvm_unreachable("op is bufferized as part of InParallelOp");
    return failure();
  }

  // COPIED FROM TENSORINTERFACEIMPL. CAN WE SHARE THE CODE SOMEHOW?
  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const BufferizationState &state,
                        const BufferizationAliasInfo &aliasInfo) const {
    Operation *readingOp = uRead->getOwner();
    Operation *conflictingWritingOp = uConflictingWrite->getOwner();

    // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
    // uRead is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<ParallelInsertSliceOp>(readingOp)) {
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }

      // TODO: Use insertSliceOp.getDestOpOperand etc. when available.
      if (uRead == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(aliasInfo, state, uConflictingWrite->get(),
                                    insertSliceOp))
        // Case 1: The main insight is that InsertSliceOp reads only part of
        // the destination tensor. The overwritten area is not read. If
        // uConflictingWrite writes into exactly the memory location that is
        // being read by uRead, this is not a conflict.
        //
        // In the above example:
        // uRead             = OpOperand 1 (%t) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
        //
        // The read of %t does not conflict with the write of the FillOp
        // (same aliases!) because the area that the FillOp operates on is
        // exactly the one that is *not* read via %t.
        return true;

      if (uRead == &insertSliceOp->getOpOperand(0) /*source*/ &&
          uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(aliasInfo, state, uRead->get(),
                                    insertSliceOp))
        // Case 2: The read of the source tensor and the write to the dest
        // tensor via an InsertSliceOp is not a conflict if the read is
        // reading exactly that part of an equivalent tensor that the
        // InsertSliceOp is writing.
        //
        // In the above example:
        // uRead             = OpOperand 0 (%1) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
        return true;
    }

    // If uConflictingWrite is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<ParallelInsertSliceOp>(conflictingWritingOp))
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }
      // %3 = vector.transfer_read %1, %cst
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of vector.transfer_read
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      // lastWrite         = %1
      //
      // This is not a conflict because the InsertSliceOp overwrites the
      // memory segment of %1 with the exact same data. (Effectively, there
      // is no memory write here.)
      if (uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          aliasInfo.areEquivalentBufferizedValues(uRead->get(),
                                                  insertSliceOp.source()) &&
          hasMatchingExtractSliceOp(aliasInfo, state, insertSliceOp.source(),
                                    insertSliceOp))
        return true;

    return false;
  }
};
} // namespace linalg_ext
} // namespace mlir

void mlir::linalg_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<InParallelOp, InParallelOpInterface>();
  registry.addOpInterface<PerformConcurrentlyOp, PerformConcurrentlyOpInterface>();
  registry.addOpInterface<linalg_ext::ParallelInsertSliceOp,
                          linalg_ext::ParallelInsertSliceOpInterface>();
}
