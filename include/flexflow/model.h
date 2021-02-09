/* Copyright 2020 Stanford
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
#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_

#include "flexflow/config.h"
#include "flexflow/tensor.h"
#include "flexflow/simulator.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <unistd.h>
#include <functional>
#include "flexflow/ffconst.h"
#include "flexflow/legion_mock.h"

using namespace Legion;

namespace flexflow {

  class FFModel;
  class Op;
  class DataLoader;

  class OpMeta {
  public:
    OpMeta(FFHandler _handle);
  public:
    FFHandler handle;
  };

  class Op {
  protected:
    void inner_measure_compute_time(Simulator *sim,
                                    std::function<void()> const &forward,
                                    std::function<void()> const &backward,
                                    float &forward_time,
                                    float &backward_time);
  public:
    Op(FFModel& model, OperatorType type, const char* _name, const Tensor& input);
    Op(FFModel& model, OperatorType type, const char* _name, const Tensor& input1, const Tensor& input2);
    Op(FFModel& model, OperatorType type, const char* _name, const Tensor& input1, const Tensor& input2, const Tensor& input3);
    Op(FFModel& model, OperatorType type, const char* _name, int num, const Tensor* inputs);
    Op(FFModel& model, OperatorType type, const char* _name, int num);
    Op(FFModel& model, OperatorType type, const Op* shared_op, const char* _name, const Tensor& input);
    // Pure virtual functions that must be implemented
    virtual bool measure_compute_time(Simulator* sim,
        const ParallelConfig& pc, float& forward, float& backward) = 0;
    // Other virtual functions that can be optionally overwritten
    virtual ParallelConfig get_random_parallel_config(const FFModel& ff) const;
    virtual ParallelConfig get_data_parallel_config(const FFModel& ff) const;
    virtual Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
    virtual Domain get_output_tensor_shape(const ParallelConfig& pc, int output_idx, int part_idx);
    virtual Domain get_weight_tensor_shape(const ParallelConfig& pc, int weight_idx, int part_idx);
    // Helper functions
    Parameter* get_parameter(int index);
  public:
    OperatorType op_type;
    char name[MAX_OPNAME];
    Tensor outputs[MAX_NUM_OUTPUTS];
    Tensor inputs[MAX_NUM_INPUTS];
    Parameter weights[MAX_NUM_WEIGHTS];
    //bool trainableInputs[MAX_NUM_INPUTS];
    //bool resetInputGrads[MAX_NUM_INPUTS];
    //Tensor locals[MAX_NUM_LOCALS];
    OpMeta* meta[MAX_NUM_WORKERS];
    int numInputs, numWeights, numOutputs;
  };

  class ElementBinary;
  class ElementUnary;
  class Conv2D;
  class Pool2D;
  class Flat;
  class Linear;
  class Embedding;

  class FFModel {
  public:
    FFModel(FFConfig &config);
    // C++ APIs for constructing models
    // Add an exp layer
    Tensor exp(const Tensor& x,
               const char *name = NULL);
    // Add an add layer
    Tensor add(const Tensor& x,
               const Tensor& y,
               char const *name = NULL);
    // Add a subtract layer
    Tensor subtract(const Tensor& x,
                    const Tensor& y,
                    char const *name = NULL);
    // Add a multiply layer
    Tensor multiply(const Tensor& x,
                    const Tensor& y,
                    char const *name = NULL);
    // Add a divide layer
    Tensor divide(const Tensor& x,
                  const Tensor& y,
                  char const *name = NULL);
    // Add an activation layer
    Tensor relu(const Tensor& x,
                const char *name = NULL);
    Tensor sigmoid(const Tensor& x,
                   const char *name = NULL);
    Tensor tanh(const Tensor& x,
                const char *name = NULL);
    Tensor elu(const Tensor& x,
               const char *name = NULL);
    // Add a 2D convolutional layer
    Tensor conv2d(const Tensor& input,
                  int outChannels,
                  int kernelH, int kernelW,
                  int strideH, int strideW,
                  int paddingH, int paddingW,
                  int groups = 1,
                  ActiMode activation = AC_MODE_NONE,
                  bool use_bias = true,
                  const Op* shared_op = NULL,
                  const char* name = NULL);
    // Add a dropout layer
    Tensor dropout(const Tensor& input,
                   float rate,
                   unsigned long long seed = 0,
                   const char* name = NULL);
    // Add an embedding layer
    Tensor embedding(const Tensor& input,
                     int num_entires, int outDim,
                     AggrMode aggr,
                     const Op* shared_op = NULL,
                     const char* name = NULL);
    // Add a 2D pooling layer
    Tensor pool2d(const Tensor& input,
                  int kernelH, int kernelW,
                  int strideH, int strideW,
                  int paddingH, int paddingW,
                  PoolType type = POOL_MAX,
                  ActiMode activation = AC_MODE_NONE,
                  const char* name = NULL);
    // Add a batch_norm layer
    Tensor batch_norm(const Tensor& input,
                      bool relu = true,
                      const char* name = NULL);
    // Add a batch_matmul layer
    Tensor batch_matmul(const Tensor& A,
                        const Tensor& B,
                        const char *name = NULL);
    // Add a dense layer
    Tensor dense(const Tensor& input,
                 int outDim,
                 ActiMode activation = AC_MODE_NONE,
                 bool use_bias = true,
                 const Op* shared_op = NULL,
                 const char *name = NULL);
    // Add a concat layer
    Tensor concat(int n,
                  const Tensor* tensors,
                  int axis,
                  const char *name = NULL);
    // Add a split layer
    void split(const Tensor& input, Tensor* outputs,
               const std::vector<int>& split, int axis,
               const char *name = NULL);
    // Add a flat layer
    Tensor flat(const Tensor& input, const char *name = NULL);
    // Add a softmax layer
    Tensor softmax(const Tensor& input,
                   const char *name = NULL);
    // Create input tensors and constants
    Tensor transpose(const Tensor& input,
                     const std::vector<int>& perm,
                     const char *name = NULL);
    Tensor reshape(const Tensor& input,
                   const std::vector<int>& shape,
                   const char *name = NULL);
    Tensor reverse(const Tensor& input,
                   int axis,
                   const char *name = NULL);
    Tensor multihead_attention(const Tensor& query,
                               const Tensor& key,
                               const Tensor& value,
                               int embed_dim,
                               int num_heads,
                               int kdim = 0,
                               int vdim = 0,
                               float dropout = 0.0f,
                               bool bias = true,
                               bool add_bias_kv = false,
                               bool add_zero_attn = false,
                               const char *name = NULL);
    // ========================================
    // Internal APIs that should not be invoked from applications
    // ========================================
    // Deprecated API --- to be removed
    //template<int NDIM>
    //Tensor create_tensor(const int* dims,
    //                     const IndexSpaceT<NDIM>& part_is,
    //                     DataType data_type,
    //                     bool create_grad = true);
    void optimize(Simulator* simulator,
                  std::map<Op*, ParallelConfig>& best,
                  size_t budget, float alpha) const;
    void rewrite(const std::map<Op*, ParallelConfig>& current,
                 std::map<Op*, ParallelConfig>& next) const;
    std::string get_operator_type_name(OperatorType type) const;
    // Internal funcitons
    Tensor get_tensor_from_guid(int guid);
  public:
    int op_global_guid;
    FFConfig config;
    Tensor label_tensor;
    //std::vector<Tensor> input_tensors;

    std::vector<Op*> layers;
    std::vector<Parameter> parameters;
    FFHandler handlers[MAX_NUM_WORKERS];
    //DataLoader *dataLoader;
  private:
    bool debug;

    Tensor binary(OperatorType op,
                  Tensor const &x,
                  Tensor const &y,
                  char const *name = NULL);
    ElementBinary * binary(OperatorType op,
                           char const *name = NULL);
    Tensor unary(OperatorType op,
                 Tensor const &x,
                 char const *name = NULL);
    ElementUnary * unary(OperatorType op,
                         char const *name = NULL);
  };

  class ElementBinaryMeta : public OpMeta {
  public:
    ElementBinaryMeta(FFHandler handle);
    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnOpTensorDescriptor_t opDesc;
    OperatorType op_type;
  };

  class ElementBinary : public Op {
  public:
    ElementBinary(FFModel& model,
                  OperatorType type,
                  const Tensor& x,
                  const Tensor& y,
                  const char* name);

    bool measure_compute_time(Simulator* sim,
                              const ParallelConfig& pc,
                              float& forward_time,
                              float& backward_time);
    static void forward_kernel(const ElementBinaryMeta* m,
                        const float* in1_ptr,
                        const float* in2_ptr,
                        float* out_ptr);
    static void backward_kernel(const ElementBinaryMeta* m,
                         const float* out_grad_ptr,
                         const float* in1_ptr,
                         const float* in2_ptr,
                         float* in1_grad_ptr,
                         float* in2_grad_ptr);
  public:
    //IndexSpace task_is;
    OperatorType op_type;
    bool profiling;
  };

  class ElementUnaryMeta : public OpMeta {
  public:
    ElementUnaryMeta(FFHandler handle);
    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnActivationDescriptor_t actiDesc;
    OperatorType op_type;
  };

  class ElementUnary : public Op {
  public:
    ElementUnary(FFModel& model,
                 OperatorType type,
                 const Tensor& x,
                 const char* name);
    static void forward_kernel(const ElementUnaryMeta* m,
                        const float* in_ptr,
                        float* out_ptr,
                        size_t num_elements);
    static void backward_kernel(const ElementUnaryMeta* m,
                         const float* in_ptr,
                         float* in_grad_ptr,
                         const float* out_ptr,
                         const float* out_grad_ptr,
                         size_t num_elements);
    bool measure_compute_time(Simulator* sim,
                              const ParallelConfig& pc,
                              float& forward_time,
                              float& backward_time);
    static bool use_cudnn(OperatorType type);
  };

  class Conv2DMeta : public OpMeta {
  public:
    Conv2DMeta(FFHandler handler);
    cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
    cudnnFilterDescriptor_t filterDesc;
    cudnnActivationDescriptor_t actiDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
    bool relu;
  };

  class Conv2D : public Op {
  public:
    Conv2D(FFModel& model,
           const Tensor& input,
           int out_dim,
           int kernelH, int kernelW,
           int strideH, int strideW,
           int paddingH, int paddingW,
           int groups,
           ActiMode activation,
           bool use_bias,
           const Op* shared_op,
           const char* name);
    void init(const FFModel&);
    //void update(const FFModel&);
    //Parameter* get_parameter(int index);

    static void forward_kernel(const Conv2DMeta* m,
                        const float* input_ptr,
                        float* output_ptr,
                        const float* filter_ptr,
                        const float* bias_ptr);
    static void backward_kernel(const Conv2DMeta* m,
                         const float* input_ptr,
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
  public:
    int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
    bool profiling, use_bias;
    ActiMode activation;
  };

  class Pool2D : public Op {
  public:
    Pool2D(FFModel& model,
           const Tensor& input,
           int kernelH, int kernelW,
           int strideH, int strideW,
           int paddingH, int paddingW,
           PoolType type, ActiMode _activation,
           const char* name);
    static void forward_kernel(const Pool2DMeta* m,
                               const float* input_ptr,
                               float* output_ptr);
    static void backward_kernel(const Pool2DMeta* m,
                                const float* input_ptr,
                                float* input_grad_ptr,
                                const float* output_ptr,
                                const float* output_grad_ptr);
    bool measure_compute_time(Simulator* sim,
                              const ParallelConfig& pc,
                              float& forward_time,
                              float& backward_time);
  public:
    int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
    PoolType pool_type;
    ActiMode activation;
    bool profiling;
  };

  class Pool2DMeta : public OpMeta {
  public:
    Pool2DMeta(FFHandler handle);
    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnActivationDescriptor_t actiDesc;
    cudnnPoolingDescriptor_t poolDesc;
    bool relu;
  };

  class LinearMeta : public OpMeta {
  public:
    LinearMeta(FFHandler handle, int batch_size);
    cudnnTensorDescriptor_t outputTensor;
    cudnnActivationDescriptor_t actiDesc;
    const float *one_ptr;
    ActiMode activation;
  };

  class Linear : public Op {
  public:
    Linear(FFModel& model,
           const Tensor& input,
           int outChannels,
           ActiMode activation,
           bool use_bias,
           const Op* shared_op,
           const char* name);

    static void forward_kernel(const LinearMeta* m,
                        const float* input_ptr,
                        float* output_ptr,
                        const float* filter_ptr,
                        const float* bias_ptr,
                        int in_dim, int out_dim, int batch_size);
    static void backward_kernel(const LinearMeta* m,
                         const float* input_ptr,
                         float* input_grad_ptr,
                         const float* output_ptr,
                         float* output_grad_ptr,
                         const float* kernel_ptr,
                         float* kernel_grad_ptr,
                         float* bias_ptr,
                         int in_dim, int out_dim, int batch_size);
    bool measure_compute_time(Simulator* sim,
                              const ParallelConfig& pc,
                              float& forward_time,
                              float& backward_time);
    ParallelConfig get_random_parallel_config(const FFModel& ff) const override;
  public:
    int in_channels, out_channels;
    //Tensor replica;
    bool profiling, use_bias;
    ActiMode activation;
  };

  class ConcatMeta : public OpMeta {
  public:
    ConcatMeta(FFHandler handle) : OpMeta(handle) {};
    int axis;
  };

  class Concat : public Op {
  public:
    Concat(FFModel& model,
           int n,
           const Tensor* inputs,
           int axis,
           const char* name);

    void init_meta(ConcatMeta *meta) const;
    static void forward_kernel(float* output,
                               float const * const *inputs,
                               int num_inputs,
                               int axis,
                               const Domain& out_domain,
                               const Domain* in_domain);
    static void backward_kernel(const float* output_grad,
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
    bool profiling;
  };

  class Split : public Op {
  public:
    Split(FFModel& model,
          const Tensor& input,
          const std::vector<int>& split,
          int axis,
          const char* name);

    static void forward_kernel(float **out_ptrs,
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
    bool profiling;
  };

  class Flat : public Op {
  public:
    Flat(FFModel& model,
         const Tensor& input,
         const char* name);

    static void forward_kernel(const float* input_ptr,
                               float* output_ptr,
                               size_t num_elements);
    static void backward_kernel(float* input_grad_ptr,
                                const float* output_grad_ptr,
                                size_t num_elements);
    bool measure_compute_time(Simulator* sim,
                              const ParallelConfig& pc,
                              float& forward_time,
                              float& backward_time);

    Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
  };

  class FlatMeta : public OpMeta {
  public:
    FlatMeta(FFHandler handle) : OpMeta(handle) {};
  };
}

#endif//_FLEXFLOW_MODEL_H_
