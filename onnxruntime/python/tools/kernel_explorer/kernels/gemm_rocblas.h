// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>

#include "core/common/common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "python/tools/kernel_explorer/kernels/gemm.h"

namespace py = pybind11;

namespace onnxruntime {

// TODO: move to onnxruntime once we have a monolithicly tunable gemm wrapper
template <typename T>
Status RocBlasGemmOp(const GemmParams<T>* params) {
  // NOTE: rocblas assumes the storage is column-majored, swapping A and B makes it have the same interface
  // as those with row-majored convention. That is, if you treat the storage as row-majored but view the matrices as
  // transposed, then by using the property Transpose(A*B) = Tranpose(B)*Transpose(A), the correctness is obvious.
  auto status = rocblasGemmHelper(
      params->handle,
      params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->n, params->m, params->k,
      &(params->alpha),
      params->b, params->ldb,
      params->a, params->lda,
      &(params->beta),
      params->c, params->ldc);
  // TODO: use ORT status wrapper
  return status == rocblas_status_success
             ? Status::OK()
             : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, rocblas_status_to_string(status));
}

void InitRocBlasGemm(py::module mod);

}
