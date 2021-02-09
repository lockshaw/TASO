/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _CNN_OPS_H_
#define _CNN_OPS_H_

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef TRT
#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;
#endif

#ifdef USE_DNNL
#include "dnnl.hpp"

// FIXME: check DNNL_VERSION_MAJOR/MINOR
#define DNNL_NO_MATMUL

using DNNLNet = std::vector<std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>>;
#endif

#include <cassert>
#include <map>
#include <array>
#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <fstream>
#include <memory>
#include "flexflow/legion_mock.h"
#include "taso/tensor.h"
#include "flexflow/ffconst.h"
#include "taso/graph.h"
//#include "flexflow/simulator.h"
#include "flexflow/config.h"

using namespace std;

class Simulator;

#define BATCH_SIZE 1
#define MAX_TENSOR_SIZE 512 * 1024 * 1024 // 512MB
#define REPEAT_TIMES 32
#define WARMUP_TIMES 8
const size_t WORK_SPACE_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2GB
typedef float DATATYPE;

class Model;

class OpBase {
protected:
  void inner_measure_compute_time(Simulator *sim,
                                  std::function<void()> const &forward,
                                  std::function<void()> const &backward,
                                  float &forward_time,
                                  float &backward_time);
public:
  OpBase(Model* _model, OpType _type); // No inputs
  OpBase(const Tensor& input, Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1,
         Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1, const Tensor& input2,
         Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1, const Tensor& input2,
         const Tensor& input3, Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1,
         const Tensor& input2, const Tensor& input3,
         const Tensor& input4, Model* _model, OpType _type);
  OpBase(int n, Tensor* inputs, Model* _model, OpType _type);
  virtual bool get_input_parameter(TNParameter, DIMParameter, int*);
  virtual bool get_int_parameter(PMParameter, int*);
  virtual bool get_float_parameter(PMParameter, float*);
  //virtual bool get_ints_parameter(PMParameter, std::vector<int>*);
  /* virtual void forward(bool block = false) = 0; */
  /* virtual void map(void) = 0; */
  /* virtual void unmap(void) = 0; */
  /* virtual void collect_costs(float& exe_time, float& flops, */
  /*                            float& mem_acc, int& num_kernels) = 0; */
  virtual bool measure_compute_time(Simulator* sim,
      const ParallelConfig& pc, float& forward, float& backward) = 0;
  virtual ParallelConfig get_data_parallel_config(FFConfig const &conf) const;
  virtual ParallelConfig get_random_parallel_config(FFConfig const &conf) const;
  virtual Domain get_output_tensor_shape(ParallelConfig const &pc, int output_idx, int part_idx);
  virtual Domain get_input_tensor_shape(ParallelConfig const &pc, int input_idx, int part_idx);
  virtual Domain get_weight_tensor_shape(const ParallelConfig& pc, int weight_idx, int part_idx);
public:
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  char name[MAX_OPNAME];
  int numInputs, numOutputs, numWeights;
  Model *model;
  OpType type;
  float runtime;
#ifdef USE_DNNL
  DNNLNet net;
#endif
};

/* class Constant : public OpBase { */
/* public: */
/*   Constant(Model* _model, int ndim, int* dims, OpType _type); */
/*   ~Constant(void); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

class Conv2D : public OpBase {
public:
  Conv2D(Model* _model,
         Tensor const &_input,
         int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         int groups,
         ActiMode activation,
         bool use_bias,
         OpBase const *shared_op,
         const char* name);
  ~Conv2D(void);

  bool get_int_parameter(PMParameter para, int*);
  /* void get_padding(int* padH, int* padW); */

  void forward_kernel(const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr);
  void backward_kernel(const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);

#ifdef USE_CUDNN
  cudnnConvolutionFwdAlgo_t selectForwardAlgorithm(void);
#endif
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu;
#endif
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  // int strideH, strideW;
  // PaddingMode padding;
  ActiMode activation;
};

/* class Matmul : public OpBase { */
/* public: */
/*   Matmul(Model* _model, Tensor _input, Tensor _weight, */
/*          ActiMode _actiMode); */
/*   ~Matmul(void); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void set_layout(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   int outputC; */
/*   ActiMode activation; */
/* #ifdef USE_CUDNN */
/*   cudnnTensorDescriptor_t outputTensor; */
/*   cudnnActivationDescriptor_t actiDesc; */
/* #endif */
/* #ifdef USE_DNNL */
/* #ifdef DNNL_NO_MATMUL */
/*   struct BLASGEMMParams { */
/*     int batch; */
/*     int m; */
/*     int n; */
/*     int k; */
/*     char transA; */
/*     char transB; */
/*     int lda; */
/*     int ldb; */
/*     int ldc; */
/*   }; */
/*   BLASGEMMParams params; */
/* #endif */
/* #endif */
/* }; */

/* class Mul : public OpBase { */
/* public: */
/*   Mul(Model* _model, const Tensor& x, const Tensor& y); */
/*   ~Mul(void); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

class Pool2D : public OpBase {
public:
  Pool2D(Model* _model,
         Tensor const &_input,
         int _kernelH, int _kernelW,
         int _strideH, int _strideW,
         int _paddingH, int _paddingW,
         PoolType type,
         ActiMode _activation);
  ~Pool2D(void);

  bool get_int_parameter(PMParameter para, int*);

  void forward_kernel(const float* input_ptr,
                      float* output_ptr);
  void backward_kernel(const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       const float* output_grad_ptr);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
#endif
  int kernelH, kernelW, strideH, strideW;
  PaddingMode padding;
  ActiMode activation;
};

class ElementUnary : public OpBase {
public:
  ElementUnary(Model* _model, Tensor _input, OpType _type, bool _inPlace);
  ~ElementUnary(void);
  bool get_int_parameter(PMParameter para, int*);
public:
  void forward_kernel(const float* in_ptr,
                      float* out_ptr,
                      size_t num_elements);
  void backward_kernel(const float* in_ptr,
                       float* in_grad_ptr,
                       const float* out_ptr,
                       const float* out_grad_ptr,
                       size_t num_elements);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  bool use_cudnn(OpType type);
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
};

class ElementBinary : public OpBase {
public:
  ElementBinary(Model *model,
                OperatorType type,
                const Tensor& x,
                const Tensor& y,
                const char* name);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  void forward_kernel(const float* in1_ptr,
                      const float* in2_ptr,
                      float* out_ptr);
  void backward_kernel(const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr);
public:
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
};

/* class BatchNorm : public OpBase { */
/* public: */
/*   BatchNorm(Model* _model, const Tensor& _input, const Tensor& _scale, */
/*             const Tensor& _bias, const Tensor& _mean, const Tensor& _var, */
/*             const float _epsilon); */
/*   ~BatchNorm(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   bool get_float_parameter(PMParameter para, float*); */
/*   float get_min_epsilon(void); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   float epsilon; */
/* #ifdef USE_CUDNN */
/*   cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor; */
/* #endif */
/* #ifdef USE_DNNL */
/*   void* scaleShiftPtr; */
/* #endif */
/*   //DATATYPE *biasPtr, *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar; */
/* }; */

/* class Cast : public OpBase { */
/* public: */
/*   Cast(Model* _model, const Tensor& _input, DataType _datatype); */
/*   ~Cast(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

class Concat : public OpBase {
public:
  Concat(Model* _model, int _axis, int _n, Tensor* _inputs, bool* _needCopy, char const *name);
  ~Concat(void);
  bool get_int_parameter(PMParameter para, int*);

  //void init_meta(ConcatMeta *meta) const;
  void forward_kernel(float* output,
                      float const * const *inputs,
                      int num_inputs,
                      int axis,
                      const Domain& out_domain,
                      const Domain* in_domain);
  void backward_kernel(const float* output_grad,
                       float** input_grads,
                       int num_inputs,
                       int axis,
                       const Domain& out_grad_domain,
                       const Domain* in_grad_domain);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);

public:
  int axis;
  //bool needCopy[MAX_NUM_INPUTS];
};

/* class Element : public OpBase { */
/* public: */
/*   Element(Model* _model, OpType _type, const Tensor& _t1, const Tensor& _t2); */
/*   ~Element(void); */
/*   bool use_kernel(void) const; */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/* #ifdef USE_CUDNN */
/*   cudnnTensorDescriptor_t in1Tensor, in2Tensor, outTensor; */
/*   cudnnOpTensorDescriptor_t opDesc; */
/* #endif */
/* }; */

/* class ElementWiseUnary : public OpBase { */
/* public: */
/*   ElementWiseUnary(Model* _model, const Tensor& _input, OpType _type); */
/*   ~ElementWiseUnary(void); */
/*   bool use_kernel(void) const; */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class Enlarge : public OpBase { */
/* public: */
/*   Enlarge(Model* _model, Tensor _w1, Tensor _w2); */
/*   ~Enlarge(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class FuseConvBatchNorm : public OpBase { */
/* public: */
/*   FuseConvBatchNorm(Model* _model, const Tensor& _conv_w, const Tensor& _scale, */
/*                     const Tensor& _bias, const Tensor& _mean, const Tensor& _var); */
/*   ~FuseConvBatchNorm(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class FuseConvBatchNormAlphaVar : public OpBase { */
/* public: */
/*   FuseConvBatchNormAlphaVar(Model* _model, const Tensor& _conv_w, const Tensor& _scale, const Tensor& _var); */
/*   ~FuseConvBatchNormAlphaVar(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class FuseConvBatchNormBias : public OpBase { */
/* public: */
/*   FuseConvBatchNormBias(Model* _model, const Tensor& _scale, */
/*                     const Tensor& _bias, const Tensor& _mean, const Tensor& _var); */
/*   ~FuseConvBatchNormBias(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class BroadcastAdd : public OpBase { */
/* public: */
/*   BroadcastAdd(Model* _model, const Tensor& _data, const Tensor& _bias); */
/*   ~BroadcastAdd(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class MergeGConv : public OpBase { */
/* public: */
/*   MergeGConv(Model* _model, const Tensor& _weight, int count); */
/*   ~MergeGConv(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   int count; */
/* }; */

/* class NoOp : public OpBase { */
/* public: */
/*   NoOp(Model* _model, Tensor _input, OpType _type); */
/*   ~NoOp(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class Pad : public OpBase { */
/* public: */
/*   Pad(Model* _model, const Tensor& _input, */
/*       const std::vector<int>& _pad_before, */
/*       const std::vector<int>& _pad_after, */
/*       float _pad_value); */
/*   ~Pad(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   std::vector<int> pad_before, pad_after; */
/*   float pad_value; */
/* }; */

/* class Reduce : public OpBase { */
/* public: */
/*   Reduce(Model* _model, const Tensor& _input, OpType _type, */
/*          const std::vector<int>& _axes, bool _keepdims); */
/*   ~Reduce(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   bool keepdims; */
/*   std::vector<int> axes; */
/* }; */

/* class Reshape : public OpBase { */
/* public: */
/*   Reshape(Model* _model, Tensor _input, const std::vector<int>& shape); */
/*   ~Reshape(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class Resize : public OpBase { */
/* public: */
/*   Resize(Model* _model, const Tensor& _input, const std::vector<int>& _shape); */
/*   ~Resize(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   std::vector<int> shape; */
/* }; */

/* class Shape : public OpBase { */
/* public: */
/*   Shape(Model* _model, const Tensor& _input, OpType _type); */
/*   ~Shape(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

/* class Slice : public OpBase { */
/* public: */
/*   Slice(Model* _model, const Tensor& _input, */
/*         const std::vector<int>& _start, */
/*         const std::vector<int>& _end, */
/*         const std::vector<int>& _axes, */
/*         const std::vector<int>& _steps); */
/*   ~Slice(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   std::vector<int> start, end, axes, steps; */
/* }; */

class Split : public OpBase {
public:
  Split(Model* _model,
        Tensor const &_input,
        std::vector<int> const &_sizes,
        int axis,
        char const *name);

  ~Split(void);
  bool get_int_parameter(PMParameter para, int*);

  void forward_kernel(float **out_ptrs,
                      float const *in_ptr,
                      coord_t const *out_blk_sizes,
                      coord_t in_blk_size,
                      coord_t num_blks,
                      int numOutputs);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int axis;
  std::vector<int> sizes;
};

/* class Squeeze : public OpBase { */
/* public: */
/*   Squeeze(Model* _model, const Tensor& input, const std::vector<int>& axes); */
/*   ~Squeeze(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   std::vector<int> axes; */
/* }; */

/* class TopK : public OpBase { */
/* public: */
/*   TopK(Model* _model, const Tensor& _input, */
/*        int _axis, int _numk, */
/*        bool _largest, bool _sorted); */
/*   ~TopK(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   int axis; */
/*   bool largest, sorted; */
/* }; */

/* class Transpose : public OpBase { */
/* public: */
/*   Transpose(Model* _model, Tensor _input, */
/*             const std::vector<int>& perm, */
/*             bool _shuffle); */
/*   ~Transpose(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   int permIdx; */
/*   bool shuffle; */
/* }; */

/* class Unsqueeze : public OpBase { */
/* public: */
/*   Unsqueeze(Model* _model, const Tensor& input, const std::vector<int>& axes); */
/*   ~Unsqueeze(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* public: */
/*   std::vector<int> axes; */
/* }; */

/* class Where : public OpBase { */
/* public: */
/*   Where(Model* _model, const Tensor& _input, const Tensor& _x, const Tensor& _y); */
/*   ~Where(void); */
/*   bool get_int_parameter(PMParameter para, int*); */
/*   void forward(bool block); */
/*   void map(void); */
/*   void unmap(void); */
/*   void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels); */
/* }; */

template<typename T>
struct KeyCompare {
  bool operator()(const T& a, const T& b) const {
    for (int i = 0; i < T::KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

/* struct ActivationKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2; */
/*   ActivationKey(Tensor, OpType, bool); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* // key is (inputN, inputC, inputH, inputW) */
/* struct BatchNormKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH; */
/*   BatchNormKey(const Tensor& _input); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct CastKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1; */
/*   CastKey(const Tensor& _input, DataType _datatype); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ConcatKey { */
/*   static const int KEY_LENGTH = MAX_NUM_INPUTS * Tensor::MAX_KEY_LENGTH + 3; */
/*   ConcatKey(int, int, Tensor*, bool*); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* //keys are (ndim, dims[0..ndims-1], constant_mode */
/* struct ConstantKey { */
/*   static const int KEY_LENGTH = MAX_DIM + 2; */
/*   ConstantKey(int, int*, OpType); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* // keys are (strideH, strideW, padding, activation, input, weight) */
/* struct Conv2DKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2 + 4; */
/*   Conv2DKey(Tensor, Tensor, int, int, */
/*             PaddingMode, ActiMode); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ElementKey { */
/*   static const int KEY_LENGTH = 2*Tensor::MAX_KEY_LENGTH + 1; */
/*   ElementKey(const Tensor& t1, const Tensor& t2, OpType type); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ElementWiseUnaryKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1; */
/*   ElementWiseUnaryKey(const Tensor& _input, OpType _type); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct EnlargeKey { */
/*   static const int KEY_LENGTH = 2 * Tensor::MAX_KEY_LENGTH; */
/*   EnlargeKey(Tensor w1, Tensor w2); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct FuseConvBatchNormKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH; */
/*   FuseConvBatchNormKey(const Tensor& conv_w); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct FuseConvBatchNormBiasKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH; */
/*   FuseConvBatchNormBiasKey(const Tensor& _scale); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct FuseConvBatchNormAlphaVarKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH; */
/*   FuseConvBatchNormAlphaVarKey(const Tensor& conv_w); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct BroadcastAddKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH; */
/*   BroadcastAddKey(const Tensor& data); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct TopKKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 4; */
/*   TopKKey(const Tensor& _input, int _axis, int _numk, bool _largest, bool _sorted); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* // keys are (inputX, inputN, inputC, outputC, acti) */
/* // */
/* struct MatmulKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2 + 1; */
/*   MatmulKey(Tensor, Tensor, ActiMode); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct MergeGConvKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1; */
/*   MergeGConvKey(const Tensor& weight, int count); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* // keys are (inputX, inputN, inputC, outputC, acti) */
/* struct MulKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2; */
/*   MulKey(const Tensor&, const Tensor&); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct NoopKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1; */
/*   NoopKey(Tensor input, OpType typee); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct PadKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2 * MAX_DIM + 1; */
/*   PadKey(const Tensor& _input, */
/*          const std::vector<int>& _pad_before, */
/*          const std::vector<int>& _pad_after, */
/*          float _pad_value); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* // keys are (inputN, inputC, inputH, inputW, kernelH, kernelW, */
/* //           strideH, strideW, padding, activation, type, */
/* //           input.split[0], input.split[1] */
/* struct Pool2DKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 7; */
/*   Pool2DKey(Tensor, OpType, int, int, int, int, */
/*             PaddingMode, ActiMode); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ReduceKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 3; */
/*   ReduceKey(const Tensor&, OpType, const std::vector<int>&, bool); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ReshapeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 1; */
/*   ReshapeKey(Tensor, const std::vector<int>&); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ResizeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 1; */
/*   ResizeKey(const Tensor&, const std::vector<int>&); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct ShapeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1; */
/*   ShapeKey(const Tensor& _input, OpType _type); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct SliceKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM * 4 + 1; */
/*   SliceKey(const Tensor& _input, */
/*            const std::vector<int>& _start, */
/*            const std::vector<int>& _end, */
/*            const std::vector<int>& _axes, */
/*            const std::vector<int>& _steps); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct SqueezeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM; */
/*   SqueezeKey(const Tensor& input, const std::vector<int>& axes); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct SplitKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_NUM_OUTPUTS + 2; */
/*   SplitKey(const Tensor& _input, int _axis, const std::vector<int>& _sizes); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct TransposeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2; */
/*   TransposeKey(Tensor, const std::vector<int>&, bool); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct UnsqueezeKey { */
/*   static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM; */
/*   UnsqueezeKey(const Tensor& input, const std::vector<int>& axes); */
/*   int keys[KEY_LENGTH]; */
/* }; */

/* struct WhereKey { */
/*   static const int KEY_LENGTH = 3 * Tensor::MAX_KEY_LENGTH; */
/*   WhereKey(const Tensor& _cond, const Tensor& _x, const Tensor& _y); */
/*   int keys[KEY_LENGTH]; */
/* }; */

class Model {
  public:
    Model(FFConfig &config);
    // C++ APIs for constructing models
    // Add an exp layer
    /* Tensor exp(const Tensor& x, */
    /*            const char *name = NULL); */
    // Add an add layer
    Tensor add(const Tensor& x,
               const Tensor& y,
               char const *name = NULL);
    Tensor pool2d(const Tensor& input,
                  int kernelH, int kernelW,
                  int strideH, int strideW,
                  int paddingH, int paddingW,
                  PoolType type = POOL_MAX,
                  ActiMode activation = AC_MODE_NONE,
                  const char* name = NULL);
    Tensor conv2d(const Tensor& input,
                  int outChannels,
                  int kernelH, int kernelW,
                  int strideH, int strideW,
                  int paddingH, int paddingW,
                  int groups = 1,
                  ActiMode activation = AC_MODE_NONE,
                  bool use_bias = true,
                  OpBase const *shared_op = NULL,
                  char const *name = NULL);
    Tensor conv2d(Tensor const &input,
                  int outChannels,
                  int kernelH, int kernelW,
                  int strideH, int strideW,
                  PaddingMode padding,
                  int groups = 1,
                  ActiMode activation = AC_MODE_NONE,
                  bool use_bias = true,
                  OpBase const *shared_op = NULL,
                  char const *name = NULL);
    // Add a subtract layer
    /* Tensor subtract(const Tensor& x, */
    /*                 const Tensor& y, */
    /*                 char const *name = NULL); */
    /* // Add a multiply layer */
    /* Tensor multiply(const Tensor& x, */
    /*                 const Tensor& y, */
    /*                 char const *name = NULL); */
    /* // Add a divide layer */
    /* Tensor divide(const Tensor& x, */
    /*               const Tensor& y, */
    /*               char const *name = NULL); */
    // Add an activation layer
    Tensor unary(OpType type,
                 Tensor const &x,
                 char const *name = NULL);
    Tensor relu(const Tensor& x,
                const char *name = NULL);
    /* Tensor sigmoid(const Tensor& x, */
    /*                const char *name = NULL); */
    /* Tensor tanh(const Tensor& x, */
    /*             const char *name = NULL); */
    /* Tensor elu(const Tensor& x, */
    /*            const char *name = NULL); */
    // Add a dense layer
    /* Tensor dense(const Tensor& input, */
    /*              int outDim, */
    /*              ActiMode activation = AC_MODE_NONE, */
    /*              bool use_bias = true, */
    /*              const OpBase* shared_op = NULL, */
    /*              const char *name = NULL); */
    // Add a concat layer
    Tensor concat(int n,
                  const Tensor* tensors,
                  int axis,
                  const char *name = NULL);
    // Add a split layer
    void split(const Tensor& input, Tensor* outputs,
               const std::vector<int>& split, int axis,
               const char *name = NULL);
    // Create input tensors and constants
    template<int NDIM>
    Tensor create_tensor(const int dims[],
                         DataType data_type,
                         const OpBase* owner_op = NULL,
                         bool create_grad = true);
    template<int NDIM>
    Tensor create_constant(const int dims[],
                           float value,
                           DataType date_type);
    void optimize(Simulator* simulator,
                  std::map<OpBase*, ParallelConfig>& best,
                  size_t budget, float alpha) const;
    void rewrite(const std::map<OpBase*, ParallelConfig>& current,
                 std::map<OpBase*, ParallelConfig>& next) const;
    std::string get_operator_type_name(OperatorType type) const;
    // Internal funcitons
    Tensor get_tensor_from_guid(int guid);
  public:
    int op_global_guid;
    FFConfig config;
    Tensor label_tensor;
    //std::vector<Tensor> input_tensors;

    std::vector<OpBase*> layers;
    FFHandler *handler;
    //DataLoader *dataLoader;
  private:
    bool debug;

    ElementBinary * binary(OperatorType op,
                           char const *name = NULL);
    ElementUnary * unary(OperatorType op,
                         char const *name = NULL);
};

/* class Model { */
/* public: */
/*   Model(); */
/*   Op get_or_create_activation(Tensor _input, OpType _type, */
/*                               bool _inPlace); */
/*   Op get_or_create_batchnorm(const Tensor& _input, */
/*                              const Tensor& _scale, */
/*                              const Tensor& _bias, */
/*                              const Tensor& _mean, */
/*                              const Tensor& _var, */
/*                              const float _epsilon); */
/*   Op get_or_create_cast(const Tensor& _input, DataType _datatype); */
/*   Op get_or_create_concat(int axis, int n, Tensor* _inputs, bool* _needCopy); */
/*   Op get_or_create_constant(int ndim, int* dims, OpType type); */
/*   Op get_or_create_conv2d(Tensor _input, Tensor _weight, */
/*                           int _strideH, int _strideW, */
/*                           PaddingMode _padding, */
/*                           ActiMode _activation); */
/*   Op get_or_create_element(OpType type, const Tensor& t1, const Tensor& t2); */
/*   Op get_or_create_elementwise_unary(const Tensor& _input, OpType _type); */
/*   Op get_or_create_enlarge(Tensor _w1, Tensor _w2); */
/*   Op get_or_create_fuse_conv_batchnorm(const Tensor& _conv_w, */
/*                                        const Tensor& _scale, */
/*                                        const Tensor& _bias, */
/*                                        const Tensor& _mean, */
/*                                        const Tensor& _var); */
/*   Op get_or_create_fuse_conv_batchnorm_alpha_var(const Tensor& _conv_w, */
/*                                        const Tensor& _scale, */
/*                                        const Tensor& _var); */
/*   Op get_or_create_fuse_conv_batchnorm_bias(const Tensor& _scale, */
/*                                        const Tensor& _bias, */
/*                                        const Tensor& _mean, */
/*                                        const Tensor& _var); */
/*   Op get_or_create_broadcast_add(const Tensor& _data, */
/*                                  const Tensor& _bias); */
/*   Op get_or_create_matmul(Tensor _input, Tensor _weight, */
/*                           ActiMode _actimode); */
/*   Op get_or_create_mul(const Tensor& x, */
/*                        const Tensor& y); */
/*   Op get_or_create_pad(const Tensor& _input, */
/*                        const std::vector<int>& _pad_before, */
/*                        const std::vector<int>& _pad_after, */
/*                        float _pad_value); */
/*   Op get_or_create_pool2d(Tensor _input, Tensor _weight, */
/*                           OpType _type, */
/*                           int _kernelH, int _kernelW, */
/*                           int _strideH, int _strideW, */
/*                           PaddingMode _padding, */
/*                           ActiMode _activation); */
/*   Op get_or_create_reduce(const Tensor& _input, OpType _type, */
/*                           const std::vector<int>& _axes, bool _keepdims); */
/*   Op get_or_create_reshape(Tensor _input, const std::vector<int>& shape); */
/*   Op get_or_create_resize(const Tensor& _input, */
/*                           const std::vector<int>& _shape); */
/*   Op get_or_create_shape(const Tensor& _input, OpType _type); */
/*   Op get_or_create_slice(const Tensor& _input, */
/*                          const std::vector<int>& _start, */
/*                          const std::vector<int>& _end, */
/*                          const std::vector<int>& _axes, */
/*                          const std::vector<int>& _steps); */
/*   Op get_or_create_squeeze(const Tensor& input, const std::vector<int>& axes); */
/*   Op get_or_create_split(const Tensor& _input, int _axis, const std::vector<int>& _sizes); */
/*   Op get_or_create_split(const Tensor& _input, int axis, int n); */
/*   Op get_or_create_topk(const Tensor& _input, int _axis, int _numk, */
/*                         bool _largest, bool _sorted); */
/*   Op get_or_create_transpose(Tensor _input, const std::vector<int>& _perm, */
/*                              bool _shuffle); */
/*   Op get_or_create_transpose(Tensor _input, int permIdx, */
/*                              bool _shuffle); */
/*   Op get_or_create_noop(Tensor _input, OpType _type); */
/*   Op get_or_create_merge_gconv(const Tensor& _weight, */
/*                                int count); */
/*   Op get_or_create_unsqueeze(const Tensor& input, const std::vector<int>& axes); */
/*   Op get_or_create_where(const Tensor& _cond, const Tensor& _x, const Tensor& _y); */
/*   // Special API for creating weight and input operator */
/*   Op create_input(Tensor _input, OpType _type); */
/*   Op create_weight(Tensor _weight, OpType _type); */
/*   /1* void measure_conv2d_cost(Conv2D*); *1/ */
/*   /1* void measure_matmul_cost(Matmul*); *1/ */
/*   /1* void measure_mul_cost(Mul*); *1/ */
/*   /1* void measure_pad_cost(Pad*); *1/ */
/*   /1* void measure_pool2d_cost(Pool2D*); *1/ */
/*   /1* void measure_topk_cost(TopK*); *1/ */
/*   /1* void measure_transpose_cost(Transpose*); *1/ */
/*   /1* void measure_reduce_cost(Reduce*); *1/ */
/*   /1* void measure_reshape_cost(Reshape*); *1/ */
/*   /1* void measure_resize_cost(Resize*); *1/ */
/*   /1* void measure_activation_cost(Activation*); *1/ */
/*   /1* void measure_batchnorm_cost(BatchNorm*); *1/ */
/*   /1* void measure_cast_cost(Cast*); *1/ */
/*   /1* void measure_concat_cost(Concat*); *1/ */
/*   /1* void measure_shape_cost(Shape*); *1/ */
/*   /1* void measure_slice_cost(Slice*); *1/ */
/*   /1* void measure_split_cost(Split*); *1/ */
/*   /1* void measure_element_cost(Element*); *1/ */
/*   /1* void measure_elementwise_unary_cost(ElementWiseUnary*); *1/ */
/*   /1* void measure_enlarge_cost(Enlarge*); *1/ */
/*   /1* void measure_squeeze_cost(Squeeze*); *1/ */
/*   /1* void measure_unsqueeze_cost(Unsqueeze*); *1/ */
/*   /1* void measure_where_cost(Where*); *1/ */
/*   void* allocate_memory(size_t size, const DATATYPE* initial_data= NULL); */
/*   bool copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size); */
/*   float measure_oplist_runtime(const std::vector<OpBase*>& list); */
/*   bool broadcastable(const Tensor& t1, const Tensor& t2); */
/* public: */
/*   bool isTraining; */
/*   bool print_cost; */
/*   size_t global_unique_id; */
/*   size_t workSpaceSize; */
/*   void* workSpace; */
/* #ifdef USE_CUDNN */
/*   cudnnHandle_t dnn; */
/*   cublasHandle_t blas; */
/*   cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor; */
/*   cudnnFilterDescriptor_t filterDesc; */
/*   // Note that actiDesc is set when we construct Model since */
/*   // all relus are identical. */
/*   cudnnActivationDescriptor_t actiDesc; */
/*   cudnnPoolingDescriptor_t poolDesc; */
/*   cudnnConvolutionDescriptor_t convDesc; */
/*   cudaEvent_t startEvent, endEvent; */
/*   // variables for batch norm */
/*   cudnnTensorDescriptor_t scaleTensor; */
/*   // variables for element wise */
/*   cudnnOpTensorDescriptor_t opDesc; */
/* #endif */
/* #ifdef USE_DNNL */
/*   DNNLNet net; */
/*   dnnl::engine eng; */
/*   dnnl::stream strm; */
/* #endif */
/*   std::vector<OpBase*> layers; */
/*   /1* std::map<ActivationKey, Activation*, KeyCompare<ActivationKey> > activation; *1/ */
/*   /1* std::map<BatchNormKey, BatchNorm*, KeyCompare<BatchNormKey> > batchnorm; *1/ */
/*   /1* std::map<CastKey, Cast*, KeyCompare<CastKey> > cast; *1/ */
/*   /1* std::map<ConcatKey, Concat*, KeyCompare<ConcatKey> > concat; *1/ */
/*   /1* std::map<ConstantKey, Constant*, KeyCompare<ConstantKey> > constant; *1/ */
/*   /1* std::map<Conv2DKey, Conv2D*, KeyCompare<Conv2DKey> > conv2d; *1/ */
/*   /1* std::map<ElementKey, Element*, KeyCompare<ElementKey> > element; *1/ */
/*   /1* std::map<ElementWiseUnaryKey, ElementWiseUnary*, KeyCompare<ElementWiseUnaryKey> > element_unary; *1/ */
/*   /1* std::map<EnlargeKey, Enlarge*, KeyCompare<EnlargeKey> > enlarge; *1/ */
/*   /1* std::map<FuseConvBatchNormKey, FuseConvBatchNorm*, KeyCompare<FuseConvBatchNormKey> > fuse_conv_batchnorm; *1/ */
/*   /1* std::map<FuseConvBatchNormAlphaVarKey, FuseConvBatchNormAlphaVar*, KeyCompare<FuseConvBatchNormAlphaVarKey> > fuse_conv_batchnorm_alpha_var; *1/ */
/*   /1* std::map<FuseConvBatchNormBiasKey, FuseConvBatchNormBias*, KeyCompare<FuseConvBatchNormBiasKey> > fuse_conv_batchnorm_bias; *1/ */
/*   /1* std::map<BroadcastAddKey, BroadcastAdd*, KeyCompare<BroadcastAddKey> > broadcast_add; *1/ */
/*   /1* std::map<MatmulKey, Matmul*, KeyCompare<MatmulKey> > matmul; *1/ */
/*   /1* std::map<MergeGConvKey, MergeGConv*, KeyCompare<MergeGConvKey> > merge_gconv; *1/ */
/*   /1* std::map<MulKey, Mul*, KeyCompare<MulKey> > mul; *1/ */
/*   /1* std::map<NoopKey, NoOpBase*, KeyCompare<NoopKey> > noop; *1/ */
/*   /1* std::map<PadKey, Pad*, KeyCompare<PadKey> > pad; *1/ */
/*   /1* std::map<Pool2DKey, Pool2D*, KeyCompare<Pool2DKey> > pool2d; *1/ */
/*   /1* std::map<ReduceKey, Reduce*, KeyCompare<ReduceKey> > reduce; *1/ */
/*   /1* std::map<ReshapeKey, Reshape*, KeyCompare<ReshapeKey> > reshape; *1/ */
/*   /1* std::map<ResizeKey, Resize*, KeyCompare<ResizeKey> > resize; *1/ */
/*   /1* std::map<ShapeKey, Shape*, KeyCompare<ShapeKey> > shape; *1/ */
/*   /1* std::map<SliceKey, Slice*, KeyCompare<SliceKey> > slice; *1/ */
/*   /1* std::map<SplitKey, Split*, KeyCompare<SplitKey> > split; *1/ */
/*   /1* std::map<SqueezeKey, Squeeze*, KeyCompare<SqueezeKey> > squeeze; *1/ */
/*   /1* std::map<TopKKey, TopK*, KeyCompare<TopKKey> > topk; *1/ */
/*   /1* std::map<TransposeKey, Transpose*, KeyCompare<TransposeKey> > transpose; *1/ */
/*   /1* std::map<UnsqueezeKey, Unsqueeze*, KeyCompare<UnsqueezeKey> > unsqueeze; *1/ */
/*   /1* std::map<WhereKey, Where*, KeyCompare<WhereKey> > where; *1/ */
/*   /1* DATATYPE *inputPtr, *biasPtr, *outputPtr, *filterPtr; *1/ */
/*   // variables for batch norm */
/*   /1* DATATYPE *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar; *1/ */
/* }; */

#endif
