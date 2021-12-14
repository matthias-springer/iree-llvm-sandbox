set(LLVM_OPTIONAL_SOURCES
  InParallelToAsync.cpp
  InParallelToSequentialFor.cpp
  LinalgExtBufferization.cpp
  TilingExternalModels.cpp
  TileToInParallel.cpp
  TileToSequentialFor.cpp
  Tiling.cpp
  TilingToTileOp.cpp
  Utils.cpp
)

add_mlir_library(MLIRLinalgExtTransforms
  InParallelToAsync.cpp
  InParallelToSequentialFor.cpp
  TilingExternalModels.cpp
  TileToSequentialFor.cpp
  TileToInParallel.cpp
  Tiling.cpp
  TilingToTileOp.cpp
  Utils.cpp

  DEPENDS
  MLIRLinalgExt

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRAsync
  MLIRSCFToStandard
  MLIRLinalgToLLVM
  MLIRVectorToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRIR
  MLIRMath
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLinalgExt
  MLIRPass
  MLIRSCF
  MLIRTransforms
)

add_mlir_library(MLIRLinalgExtOpInterfaceImpl
  LinalgExtBufferization.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRLinalgExt
)
