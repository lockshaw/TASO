/* Copyright 2020 Stanford, NVIDIA, Facebook
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

#include "flexflow/model.h"
#include "flexflow/cuda_helper.h"

using namespace flexflow;

Tensor FFModel::dense(const Tensor& input,
                      int outDim,
                      ActiMode activation,
                      bool use_bias,
                      const Op* shared_op,
                      const char *name)
{
  layers.push_back(
    std::unique_ptr<Op>(
      new Linear(*this, input, outDim, activation, use_bias, shared_op, name)
    )
  );
  return layers.back()->outputs[0];
}

Linear::Linear(FFModel& model,
               const Tensor& _input,
               int out_dim,
               ActiMode _activation,
               bool _use_bias,
               const Op* shared_op,
               const char* name)
: Op(model, OP_LINEAR, shared_op, name, _input),
  in_channels(_input.adim[0]), out_channels(out_dim),
  activation(_activation), use_bias(_use_bias),
  profiling(model.config.profiling)
{
  numInputs = 1;
  numOutputs = 1;
  outputs[0].numDim = _input.numDim;
  for (int i = 1; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = _input.adim[i];
  outputs[0].adim[0] = out_dim;
  weights[0].numDim = 2;
  weights[0].adim[0] = in_channels;
  weights[0].adim[1] = out_channels;
  numWeights = 1;
  if (use_bias) {
    weights[1].numDim = 1;
    weights[1].adim[0] = out_channels;
    numWeights = 2;
  }
  /* printf("Op: %s\n", name); */
  /* printf("Input: "); */
  /* for (int i = 0; i < _input.numDim; i++) { */
  /*   printf("%d ", _input.adim[i]); */
  /* } */
  /* printf("\nOutput: "); */
  /* for (int i = 0; i < outputs[0].numDim; i++) { */
  /*   printf("%d ", outputs[0].adim[i]); */
  /* } */
  /* printf("\nWeights: "); */
  /* for (int i = 0; i < weights[0].numDim; i++) { */
  /*   printf("%d ", weights[0].adim[i]); */
  /* } */
  /* printf("\n"); */
}

/*static*/
void Linear::forward_kernel(const LinearMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            const float* kernel_ptr,
                            const float* bias_ptr,
                            int in_dim, int out_dim, int batch_size)
{
  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        out_dim, batch_size, in_dim,
                        &alpha, kernel_ptr, in_dim,
                        input_ptr, in_dim, &beta,
                        output_ptr, out_dim));
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        out_dim, batch_size, 1,
                        &alpha, bias_ptr, 1,
                        m->one_ptr, 1, &alpha,
                        output_ptr, out_dim));
  if (m->activation != AC_MODE_NONE) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  }
}

__global__
void sigmoid_backward(float *grad_ptr, const float *output, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = grad_ptr[i] * output[i] * (1 - output[i]);
  }
}

/*static*/
void Linear::backward_kernel(const LinearMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr,
                             int in_dim, int out_dim, int batch_size)
{
  float alpha = 1.0f;
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    reluBackward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        in_dim, out_dim, batch_size,
                        &alpha, input_ptr, in_dim,
                        output_grad_ptr, out_dim,
                        &alpha, kernel_grad_ptr, in_dim));
  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                        out_dim, batch_size,
                        &alpha, output_grad_ptr, out_dim,
                        m->one_ptr, 1,
                        &alpha, bias_grad_ptr, 1));
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        in_dim, batch_size, out_dim,
                        &alpha, kernel_ptr, in_dim,
                        output_grad_ptr, out_dim,
                        &alpha, input_grad_ptr, in_dim));
}

LinearMeta::LinearMeta(FFHandler handler, int batch_size)
: OpMeta(handler)
{
  // Allocate an all-one's vector
  float* dram_one_ptr = (float *) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  float* fb_one_ptr;
  checkCUDA(cudaMalloc(&fb_one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(fb_one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  one_ptr = (const float*) fb_one_ptr;
  // Allocate descriptors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

bool Linear::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  Tensor sub_output, sub_input;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, OP_LINEAR))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, OP_LINEAR))
    return false;
  int input_c = sub_input.adim[0];
  int input_n = sub_input.get_volume() / input_c;
  int output_c = sub_output.adim[0];
  int output_n = sub_output.get_volume() / output_c;
  LinearMeta* m = sim->linear_meta;
  m->activation = activation;
  if (activation != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, 1, 1));
  }
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* kernel_ptr = (float*)sim->allocate((size_t)output_c * input_c, DT_FLOAT);
  assert(kernel_ptr != NULL);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, kernel_ptr, bias_ptr,
        input_c, output_c, input_n);
  };
  auto backward = [&] {
    backward_kernel(m, input_ptr, input_ptr, output_ptr, output_ptr,
        kernel_ptr, kernel_ptr, bias_ptr, input_c, output_c, input_n);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  if (sim->verbosity >= SimulationVerbosity::ALL) {
    printf("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
           name, input_n, input_c, output_n, output_c, forward_time, backward_time);
  }

  return true;
}

ParallelConfig Linear::get_random_parallel_config(const FFModel& ff) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::get_random_parallel_config(ff);
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0].adim[outputs[0].numDim-1];
  int channel = outputs[0].adim[0];
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++)
    if (channel % i == 0)
      for (int j = 1; i * j <= total_devices; j++)
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0].numDim;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims-1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++)
    pc.dim[i] = 1;
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

