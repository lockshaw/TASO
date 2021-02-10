/* Copyright 2018 Stanford
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

Tensor FFModel::pool2d(const Tensor& input,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       PoolType type, ActiMode activation,
                       char const *name)
{
  assert(input.numDim == 4); /*NCHW*/
  layers.push_back(
    std::unique_ptr<Op>(
      new Pool2D(*this, input, kernelH, kernelW,
                 strideH, strideW, paddingH, paddingW,
                 type, activation, name)

    )
  );
  return layers.back()->outputs[0];
}

Pool2D::Pool2D(FFModel& model,
               const Tensor& _input,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               PoolType _type, ActiMode _activation,
               const char* name)
: Op(model, OP_POOL2D, name, _input),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  pool_type(_type), activation(_activation),
  profiling(model.config.profiling)
{
  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = inputs[0].adim[2];
  int output_n = inputs[0].adim[3];
  outputs[0].numDim = 4;
  outputs[0].adim[0] = output_w;
  outputs[0].adim[1] = output_h;
  outputs[0].adim[2] = output_c;
  outputs[0].adim[3] = output_n;
}

/*static*/
void Pool2D::forward_kernel(const Pool2DMeta* m,
                            const float* input_ptr,
                            float* output_ptr)
{
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->outputTensor, output_ptr));
}

/*static*/
void Pool2D::backward_kernel(const Pool2DMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             const float* output_grad_ptr)
{
  float alpha = 1.0f;
  checkCUDNN(cudnnPoolingBackward(m->handle.dnn, m->poolDesc,
                                  &alpha, m->outputTensor, output_ptr,
                                  m->outputTensor, output_grad_ptr,
                                  m->inputTensor, input_ptr,
                                  &alpha, m->inputTensor, input_grad_ptr));
}

Pool2DMeta::Pool2DMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
}

bool Pool2D::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  Tensor sub_output, sub_input;
  if(!outputs[0].get_output_sub_tensor(pc, sub_output, OP_CONV2D))
    return false;
  if(!inputs[0].get_input_sub_tensor(pc, sub_input, OP_CONV2D))
    return false;
  int input_w = sub_input.adim[0];
  int input_h = sub_input.adim[1];
  int input_c = sub_input.adim[2];
  int input_n = sub_input.adim[3];
  int output_w = sub_output.adim[0];
  int output_h = sub_output.adim[1];
  int output_c = sub_output.adim[2];
  int output_n = sub_output.adim[3];
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;
  Pool2DMeta* m = sim->pool2d_meta;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  cudnnPoolingMode_t mode;
  if (pool_type == POOL_MAX)
    mode = CUDNN_POOLING_MAX;
  else {
    assert(pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,//pool->padding_h,
                                         pad_w,//pool->padding_w,
                                         stride_h,
                                         stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_grad_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_grad_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input_ptr, output_ptr);
  };
  auto backward = [&] {
    backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Pool2D] name(%s) input(%d %d %d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
      name,
      input_n, input_c, input_h, input_w,
      output_n, output_c, output_h, output_w,
      stride_h, stride_w,
      padding_h, padding_w,
      forward_time, backward_time);

  return true;
}
