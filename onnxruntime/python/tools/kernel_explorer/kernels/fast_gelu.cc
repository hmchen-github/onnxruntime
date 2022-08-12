// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"
#include "contrib_ops/rocm/bert/fast_gelu_tunable_op.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/device_array.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int TPB, int Vec>
class FastGelu : public IKernelExplorer {
 public:
  FastGelu(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length) {
    params_.stream = this->Stream();
    params_.input = static_cast<T*>(input.ptr());
    params_.bias = static_cast<T*>(bias.ptr());
    params_.output = static_cast<T*>(output.ptr());
    params_.input_length = input_length;
    params_.bias_length = bias_length;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::FastGeluOp<T, TPB, Vec>(&params_)));
  }

 private:
  using ParamsT = contrib::rocm::FastGeluParams<T>;
  ParamsT params_{};
};

template <typename T>
class FastGeluTunable : public IKernelExplorer {
 public:
  FastGeluTunable(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length) {
    params_.stream = Stream();
    params_.input = static_cast<T*>(input.ptr());
    params_.bias = static_cast<T*>(bias.ptr());
    params_.output = static_cast<T*>(output.ptr());
    params_.input_length = input_length;
    params_.bias_length = bias_length;

    op_.EnableTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

 private:
  using ParamsT = contrib::rocm::FastGeluParams<T>;
  ParamsT params_{};
  contrib::rocm::FastGeluTunableOp<T> op_{};
};

#define REGISTER_OP(type, threads_per_block, vec_size)                                                               \
  py::class_<FastGelu<type, threads_per_block, vec_size>>(m, "FastGelu_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>())                                           \
      .def("SetRepeats", &FastGelu<type, threads_per_block, vec_size>::SetRepeats)                                   \
      .def("Profile", &FastGelu<type, threads_per_block, vec_size>::Profile)                                         \
      .def("Run", &FastGelu<type, threads_per_block, vec_size>::Run);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(type, threads_per_block) \
  REGISTER_OP(type, threads_per_block, 1)                     \
  REGISTER_OP(type, threads_per_block, 2)                     \
  REGISTER_OP(type, threads_per_block, 4)                     \
  REGISTER_OP(type, threads_per_block, 8)                     \
  REGISTER_OP(type, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(type) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 64)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 128)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 192)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 256)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 320)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 384)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 448)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 512)

#define REGISTER_TUNABLE_OP(type)                                          \
  py::class_<FastGeluTunable<type>>(m, "FastGelu_" #type "_Tunable")       \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>()) \
      .def("SetRepeats", &FastGeluTunable<type>::SetRepeats)               \
      .def("Profile", &FastGeluTunable<type>::Profile)                     \
      .def("Run", &FastGeluTunable<type>::Run);

void InitFastGelu(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(float);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(double);

  REGISTER_TUNABLE_OP(half);
  REGISTER_TUNABLE_OP(float);
  REGISTER_TUNABLE_OP(double);
}

}  // namespace onnxruntime
