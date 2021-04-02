//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

// Defined in the runners directory, no public header.
namespace mlir {
namespace linalg {
void registerLinalgComprehensiveBufferizePass();
void registerLinalgTensorCodegenStrategyPass();
void registerLinalgTiledLoopToSCFPass();
}  // namespace linalg
void registerConvertToGPUPass();
}  // namespace mlir

void registerCustomPasses() {
  registerLinalgComprehensiveBufferizePass();
  registerLinalgTensorCodegenStrategyPass();
  registerLinalgTiledLoopToSCFPass();
  registerConvertToGPUPass();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  registerCustomPasses();

  DialectRegistry registry;
  registerAllDialects(registry);

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
