// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/gemm_ck.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

#include "python/tools/kernel_explorer/kernels/gemm.h"

namespace py = pybind11;

namespace onnxruntime {

namespace {

template <typename T>
struct DataTypeAdaptor {
  using type = T;
};

template <>
struct DataTypeAdaptor<half> {
  using type = ck::half_t;
};

}  // namespace

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

// TODO: move to onnxruntime once we have a monolithicly tunable gemm wrapper
template <typename T, typename ALayout, typename BLayout>
class CKGemmOp {
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
    ORT_RETURN_IF(!impl_->IsSupportedArgument(arg.get()), impl_->GetTypeString(), " does not support ", params->Signature());
    invoker_->Run(arg.get(), StreamConfig{params->stream});
    return Status::OK();
  }

  std::string GetTypeString() const {
    return impl_->GetTypeString();
  }

  std::unique_ptr<DeviceGemm> impl_;
  std::unique_ptr<ck::tensor_operation::device::BaseInvoker> invoker_;
};

template <typename T, typename ALayout, typename BLayout>
class CKGemm : public IKernelExplorer {
 public:
  CKGemm(BlasOp opa, BlasOp opb,
         int64_t m, int64_t n, int64_t k,
         double alpha,
         DeviceArray& a, int64_t lda,
         DeviceArray& b, int64_t ldb,
         double beta,
         DeviceArray& c, int64_t ldc)
      : params_{},
        impls_{CKGemmOp<T, ALayout, BLayout>::GetInstances()} {
    auto supports_a = opa == BlasOp::N ? std::is_same_v<ALayout, Row> : std::is_same_v<ALayout, Col>;
    auto supports_b = opb == BlasOp::N ? std::is_same_v<BLayout, Row> : std::is_same_v<BLayout, Col>;
    ORT_ENFORCE(!impls_.empty());
    ORT_ENFORCE(supports_a && supports_b);

    // rocblas handle is not used for ck
    params_.handle = nullptr;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.alpha = alpha;
    params_.a = static_cast<T*>(a.ptr());
    params_.lda = lda;
    params_.b = static_cast<T*>(b.ptr());
    params_.ldb = ldb;
    params_.beta = beta;
    params_.c = static_cast<T*>(c.ptr());
    params_.ldc = ldc;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(impls_[selected_impl_](&params_));
  }

  std::vector<std::string> ListImpls() const {
    std::vector<std::string> results;
    std::transform(impls_.cbegin(), impls_.cend(), std::back_inserter(results),
                   [](const auto& it) { return it.GetTypeString(); });
    return results;
  }

  bool SelectImpl(const std::string& name) {
    for (size_t i = 0; i < impls_.size(); i++) {
      if (impls_[i].GetTypeString() == name) {
        selected_impl_ = i;
        Status status = impls_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = GemmParams<T>;
  ParamsT params_;

  std::vector<CKGemmOp<T, ALayout, BLayout>> impls_;
  size_t selected_impl_{};
};

#define REGISTER_OP(type, alayout, blayout, layout_string)                         \
  py::class_<CKGemm<type, alayout, blayout>>(m, "CKGemm_" #type "_" layout_string) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                     \
                    double,                                                        \
                    DeviceArray&, int64_t,                                         \
                    DeviceArray&, int64_t,                                         \
                    double,                                                        \
                    DeviceArray&, int64_t>())                                      \
      .def("SetRepeats", &CKGemm<type, alayout, blayout>::SetRepeats)              \
      .def("Profile", &CKGemm<type, alayout, blayout>::Profile)                    \
      .def("Run", &CKGemm<type, alayout, blayout>::Run)                            \
      .def("ListImpls", &CKGemm<type, alayout, blayout>::ListImpls)                \
      .def("SelectImpl", &CKGemm<type, alayout, blayout>::SelectImpl);

#define REGISTER_OP_FOR_ALL_TRANSAB(type) \
  REGISTER_OP(type, Row, Row, "NN");      \
  REGISTER_OP(type, Row, Col, "NT");      \
  REGISTER_OP(type, Col, Row, "TN");      \
  REGISTER_OP(type, Col, Col, "TT");

void InitComposableKernelGemm(py::module m) {
  REGISTER_OP_FOR_ALL_TRANSAB(float);
  REGISTER_OP_FOR_ALL_TRANSAB(half);
}

}  // namespace onnxruntime
