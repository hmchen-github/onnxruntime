// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <numeric>
#include <type_traits>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/common/common.h"
#include "contrib_ops/rocm/bert/util.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct OpParams {
  std::string Signature() { return static_cast<T*>(this)->Signature(); }
  hipStream_t stream{};
};

template <typename ParamsT>
class TunableFunction {
  // using type = std::function<Status(const std::remove_cv_t<ParamsT>*)>;
public:
  template <typename T>


};

template <typename ParamsT>
using OpT = typename TunableFunction<ParamsT>::type;

template <typename ParamsT, int NumIter = 5>
struct DefaultWarmUp {
  void operator()(const OpT<ParamsT>& op, const ParamsT* param) {
    ORT_ENFORCE(NumIter >= 0);
    for (int i = 0; i < NumIter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
  };
};

template <typename ParamsT, int NumIter = 100>
struct DefaultProfile {
  double operator()(const OpT<ParamsT>& op, const ParamsT* param) {
    ORT_ENFORCE(NumIter >= 0);
    Timer timer{};
    timer.Start();
    for (int i = 0; i < NumIter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
    timer.End();
    return timer.Duration() / NumIter;
  };
};

// NOTE: onnxruntime's status currently do not have a StatusCode::UNSUPPORTED. Currently, we do not want to extend the
// enum. So we reuse StatusCode::INVALID_ARGUMENT for this purpose. It can be interpreted as "The input argument is not
// valid for this specialized kernel implementation.". This semantic is crucial for the tuning mechanism.
#define TUNABLE_OP_MAKE_UNSUPPOTED_ARGUMENT_STATUS(...) ORT_MAKE_STATUS(NONE, INVALID_ARGUMENT, __VA_ARGS__)

template <typename ParamsT>
struct IsSupported {
  bool operator()(const OpT<ParamsT>& op, const ParamsT* param) {
    Status status = op(param);
    if (status.Category() == common::StatusCategory::NONE && status.Code() == common::StatusCode::INVALID_ARGUMENT) {
      return false;
    }
    ORT_THROW_IF_ERROR(status);
    return true;
  }
};

template <typename ParamsT,
          typename WarmUp = DefaultWarmUp<ParamsT>,
          typename Profile = DefaultProfile<ParamsT>>
class TunableOp {
 public:
  Status operator()(const ParamsT* params) {
    int id;
    if (tuning_ == true && Condition(params)) {
      if (kernel_map_.find(params->Signature()) == kernel_map_.end()) {
        id = FindFastest(params);
        kernel_map_.insert({params->Signature(), id});
      } else {
        id = kernel_map_[params->Signature()];
      }
    } else {
      id = default_id_;
    }
    ORT_RETURN_IF_ERROR(ops_[id](params));
    return Status::OK();
  }

  void EnableTuning() {
    tuning_ = true;
  }

  void DisableTuning() {
    tuning_ = false;
  }

  virtual ~TunableOp() = default;

 protected:
  std::vector<OpT<ParamsT>> ops_;

 private:
  // Whether we should tune for this input
  virtual bool Condition(const ParamsT* /*params*/) {
    return true;
  }

  int FindFastest(const ParamsT* params) {
    auto min_time = std::numeric_limits<double>::max();
    int id = -1;
    for (size_t i = 0; i < this->ops_.size(); i++) {
      if (!IsSupported<ParamsT>{}(ops_[i], params)) {
        continue;
      }

      WarmUp{}(ops_[i], params);
      auto time = Profile{}(ops_[i], params);
      if (time < min_time) {
        min_time = time;
        id = static_cast<int>(i);
      }
    }
    ORT_ENFORCE(id >= 0, "Cannot found viable op");
    return id;
  }

  // mapping from Signature to best impl
  std::map<std::string, int> kernel_map_;

  // the default impl to use when tuning is disabled
  int default_id_{0};

  bool tuning_{false};
};

}  // namespace rocm
}  // namespace contrib

template <typename T>
std::string TypeToString();

template <>
inline std::string TypeToString<half>() { return "f16"; };

template <>
inline std::string TypeToString<float>() { return "f32"; };

template <>
inline std::string TypeToString<double>() { return "f64"; };

template <>
inline std::string TypeToString<int8_t>() { return "i8"; };

template <>
inline std::string TypeToString<int32_t>() { return "i32"; };

template <>
inline std::string TypeToString<int64_t>() { return "i64"; };

}  // namespace onnxruntime
