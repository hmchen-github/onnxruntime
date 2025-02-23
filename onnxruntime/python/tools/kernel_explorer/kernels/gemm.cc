// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm.h"
#include "python/tools/kernel_explorer/kernels/gemm_rocblas.h"
#include "python/tools/kernel_explorer/kernels/gemm_ck.h"

#include <type_traits>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace onnxruntime {

void InitGemm(py::module mod) {
  auto blas_op = mod.def_submodule("blas_op");

  py::enum_<BlasOp>(blas_op, "BlasOp")
      .value("N", BlasOp::N, "Passthrough")
      .value("T", BlasOp::T, "Transpose")
      .export_values();

  InitRocBlasGemm(mod);
  InitComposableKernelGemm(mod);
}

}  // namespace onnxruntime
