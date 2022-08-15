// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>
#include <hip/hip_fp16.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

#include "core/common/common.h"
#include "python/tools/kernel_explorer/kernels/gemm.h"

namespace py = pybind11;

namespace onnxruntime {

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

// TODO: move to onnxruntime once we have a monolithicly tunable gemm wrapper
template <typename T, typename ALayout, typename BLayout>
class CKGemmOp {
  template <typename DT>
  struct DataTypeAdaptor {
    using type = DT;
  };

  template <>
  struct DataTypeAdaptor<half> {
    using type = ck::half_t;
  };

  using OpType = CKGemmOp<T, ALayout, BLayout>;

  using CKDataType = typename DataTypeAdaptor<T>::type;

  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKDataType, CKDataType, CKDataType,
      Nop, Nop, Nop>;

  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

 public:
  static std::vector<OpType> GetInstances() {
    std::vector<OpType> ret;
    for (auto&& impl : InstanceFactory::GetInstances()) {
      ret.emplace_back(CKGemmOp{std::move(impl)});
    }
    return ret;
  }

  CKGemmOp(std::unique_ptr<DeviceGemm>&& device_instance)
      : impl_{std::move(device_instance)},
        invoker_{impl_->MakeInvokerPointer()} {}

  Status operator()(const GemmParams<T>* params) {
    auto nop = Nop{};
    auto arg = impl_->MakeArgumentPointer(params->a, params->b, params->c,
                                          params->m, params->n, params->k,
                                          params->lda, params->ldb, params->ldc,
                                          nop, nop, nop);
    if (!impl_->IsSupportedArgument(arg.get())) {
      return TUNABLE_OP_MAKE_UNSUPPOTED_ARGUMENT_STATUS(
          impl_->GetTypeString(), " does not support ", params->Signature());
    }
    invoker_->Run(arg.get(), StreamConfig{params->stream});
    return Status::OK();
  }

  std::string GetTypeString() const {
    return impl_->GetTypeString();
  }

  std::unique_ptr<DeviceGemm> impl_;
  std::unique_ptr<ck::tensor_operation::device::BaseInvoker> invoker_;
};

void InitComposableKernelGemm(py::module mod);

}  // namespace onnxruntime
