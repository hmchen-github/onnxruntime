// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include "contrib_ops/rocm/bert/tunable_op.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/vector_add.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.h"

namespace py = pybind11;

namespace onnxruntime {

//#####################################################################################################################
// In practice, VectorAddParam, VectorAddOp and VectorAddTunableOp should be tightly integrated to onnxruntime.
// We place it here purely for demo purpose.
//#####################################################################################################################

// Extend the OpParams so that all specialization have the same parameter passing interface
template <typename T>
struct VectorAddParams : contrib::rocm::OpParams<VectorAddParams<T>> {
  std::string Signature() const {
    return TypeToString<T>() + "_" + std::to_string(n);
  }

  T* x;
  T* y;
  T* z;
  int n;
};

// Wraps the kernel function, so that we have a unified launch interface. If the kernel has state, the state can also
// be managed at this level via a functor
template <typename T, int TPB, int Vec>
Status VectorAddOp(const VectorAddParams<T>* params) {
  return LaunchVectorAdd<T, TPB, Vec>(
      params->stream,
      params->x,
      params->y,
      params->z,
      params->n);
}

// A Tunable VectorAddOp is a collection of non-tunable VectorAddOps implementation that have variable performance
// characteristic. Thoese implementations may be put into a C++ container. So we use the wrapped
// Op object here as a proxy.
template <typename T>
class VectorAddTunableOp : public contrib::rocm::TunableOp<VectorAddParams<T>> {
 public:
  VectorAddTunableOp() {
#define ADD_SPECIALIZATIONS(threads_per_block)                   \
  this->ops_.emplace_back(VectorAddOp<T, threads_per_block, 1>); \
  this->ops_.emplace_back(VectorAddOp<T, threads_per_block, 2>); \
  this->ops_.emplace_back(VectorAddOp<T, threads_per_block, 4>); \
  this->ops_.emplace_back(VectorAddOp<T, threads_per_block, 8>);

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
  // This Op is always tunable, you generally don't need to implement it.
  virtual bool Condition(const VectorAddParams<T>* /*params*/) {
    return true;
  }
};

//#####################################################################################################################
// Following code just wraps our kernel implementation and expose them as python interface. This is what the code
// kernel_explorer should contains.
//#####################################################################################################################

template <typename T, int TPB, int Vec>
class VectorAdd : public IKernelExplorer {
 public:
  VectorAdd(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((VectorAddOp<T, TPB, Vec>(&params_)));
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = VectorAddParams<T>;
  ParamsT params_{};
};

template <typename T>
class VectorAddTunable : public IKernelExplorer {
 public:
  VectorAddTunable(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;

    impl_.EnableTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(impl_(&params_));
  }

 private:
  using ParamsT = VectorAddParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  VectorAddTunableOp<T> impl_;
};

#define REGISTER_OP(type, threads_per_block, vec_size)                                                                 \
  py::class_<VectorAdd<type, threads_per_block, vec_size>>(m, "VectorAdd_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())                                                  \
      .def("SetRepeats", &VectorAdd<type, threads_per_block, vec_size>::SetRepeats)                                    \
      .def("Profile", &VectorAdd<type, threads_per_block, vec_size>::Profile)                                          \
      .def("Run", &VectorAdd<type, threads_per_block, vec_size>::Run);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(type, threads_per_block) \
  REGISTER_OP(type, threads_per_block, 1)                     \
  REGISTER_OP(type, threads_per_block, 2)                     \
  REGISTER_OP(type, threads_per_block, 4)                     \
  REGISTER_OP(type, threads_per_block, 8)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(type) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 64)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 128)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 192)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 256)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 320)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 384)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 448)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(type, 512)

#define REGISTER_TUNABLE_OP(type)                                      \
  py::class_<VectorAddTunable<type>>(m, "VectorAdd_" #type "_Tunable") \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())  \
      .def("SetRepeats", &VectorAddTunable<type>::SetRepeats)          \
      .def("Profile", &VectorAddTunable<type>::Profile)                \
      .def("Run", &VectorAddTunable<type>::Run);

void InitVectorAdd(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(float);

  REGISTER_TUNABLE_OP(half);
  REGISTER_TUNABLE_OP(float)
}

}  // namespace onnxruntime
