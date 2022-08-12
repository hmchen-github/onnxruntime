// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include "contrib_ops/rocm/bert/tunable_op.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct FastGeluParams : OpParams<FastGeluParams<T>> {
  std::string Signature() const {
    std::string sig = TypeToString<T>() + std::to_string(input_length) + "_" + std::to_string(bias_length);
    return sig;
  }

  const T* input{};
  const T* bias{};
  T* output{};
  int input_length{};
  int bias_length{};
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status FastGeluOp(const FastGeluParams<T>* params) {
  hipLaunchKernelGGL((FastGeluKernelVec<T, ThreadsPerBlock, VecSize>),
                     dim3(CeilingDivision(params->input_length, ThreadsPerBlock * VecSize)),
                     dim3(ThreadsPerBlock),
                     0, params->stream,
                     params->input_length, params->bias_length, params->input, params->bias, params->output);
  // TODO: use ORT status wrapper
  auto status = hipGetLastError();
  return status == hipSuccess ? Status::OK() : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, hipGetErrorName(status));
}

template <typename T>
class FastGeluTunableOp : public TunableOp<FastGeluParams<T>> {
 public:
  FastGeluTunableOp() {
#define ADD_SPECIALIZATIONS(threads_per_block)                  \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 1>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 2>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 4>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 8>); \
  this->ops_.emplace_back(FastGeluOp<T, threads_per_block, 16>);

    ADD_SPECIALIZATIONS(64);
    ADD_SPECIALIZATIONS(128);
    ADD_SPECIALIZATIONS(192);
    ADD_SPECIALIZATIONS(256);
    ADD_SPECIALIZATIONS(320);
    ADD_SPECIALIZATIONS(384);
    ADD_SPECIALIZATIONS(448);
    ADD_SPECIALIZATIONS(512);

#undef ADD_SPECIALIZATIONS
  }

 private:
  bool Condition(const FastGeluParams<T>* params) override {
    bool condition = (params->bias_length > 0) && (params->bias_length % 16 == 0);
    return condition;
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
