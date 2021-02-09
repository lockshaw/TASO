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

#include "flexflow/model.h"
#include "flexflow/cuda_helper.h"

using namespace flexflow;

Tensor FFModel::unary(OperatorType op,
                      const Tensor& x,
                      const char *name)
{
  ElementUnary *ele = new ElementUnary(*this, op, x, name);
  layers.push_back(ele);
  return ele->outputs[0];
}

Tensor FFModel::exp(const Tensor& x,
                    const char *name)
{
  return this->unary(OP_EXP, x, name);
}

Tensor FFModel::relu(const Tensor& x, const char *name)
{
  return this->unary(OP_RELU, x, name);
}

Tensor FFModel::sigmoid(const Tensor& x, const char *name)
{
  return this->unary(OP_SIGMOID, x, name);
}

Tensor FFModel::tanh(const Tensor& x, const char *name)
{
  return this->unary(OP_TANH, x, name);
}

Tensor FFModel::elu(const Tensor& x, const char *name)
{
  return this->unary(OP_ELU, x, name);
}

ElementUnary::ElementUnary(FFModel& model,
                           OperatorType _op_type,
                           const Tensor& x,
                           const char* name)
: Op(model, _op_type, name, x)
{
  outputs[0].numDim = inputs[0].numDim;
  printf("Op: %s\n", name);
  for (int i = 0; i < outputs[0].numDim; i++) {
    printf("%d ", inputs[0].adim[i]);;
  }
  printf("\n");
  for (int i = 0; i < outputs[0].numDim; i++) {
    outputs[0].adim[i] = inputs[0].adim[i];
  }
}

bool ElementUnary::use_cudnn(OperatorType type)
{
  if (type == OP_RELU)
    return true;
  if (type == OP_SIGMOID)
    return true;
  if (type == OP_TANH)
    return true;
  if (type == OP_ELU)
    return true;
  return false;
}

__global__
void elewise_unary_forward_kernel(coord_t volume,
                                  const float alpha,
                                  const float beta,
                                  OperatorType type,
                                  const float* in,
                                  float* out)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        out[i] = alpha * exp(in[i]) + beta * out[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementUnary::forward_kernel(const ElementUnaryMeta* m,
                                  const float* input_ptr,
                                  float* output_ptr,
                                  size_t num_elements)
{
  float alpha = 1.0f, beta = 0.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->inputTensor, input_ptr,
        &beta, m->outputTensor, output_ptr));
  } else {
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
        num_elements, alpha, beta, m->op_type, input_ptr, output_ptr);
  }
}

__global__
void elewise_unary_backward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   OperatorType type,
                                   const float* output_grad,
                                   const float* input,
                                   float* input_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        //TODO: change to use output instead of recomputing
        input_grad[i] = alpha * output_grad[i] * exp(input[i]) + beta * input_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementUnary::backward_kernel(const ElementUnaryMeta* m,
                                   const float* input_ptr,
                                   float* input_grad_ptr,
                                   const float* output_ptr,
                                   const float* output_grad_ptr,
                                   size_t num_elements)
{
  float alpha = 1.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
        m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
  } else {
    elewise_unary_backward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
        num_elements, alpha, alpha, m->op_type, output_grad_ptr, input_ptr, input_grad_ptr);
  }
}

ElementUnaryMeta::ElementUnaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

bool ElementUnary::measure_compute_time(Simulator* sim,
                                        const ParallelConfig& pc,
                                        float& forward_time,
                                        float& backward_time)
{
  Tensor sub_output, sub_input;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, op_type))
    return false;
  ElementUnaryMeta* m = sim->ele_unary_meta;
  m->op_type = op_type;
  if (use_cudnn(m->op_type))
  {
    cudnnActivationMode_t mode;
    switch (op_type) {
      case OP_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OP_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OP_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OP_ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    Domain input_domain, output_domain;
    input_domain.dim = sub_input.numDim;
    for (int i = 0; i < sub_input.numDim; i++) {
      input_domain.rect_data[i] = 0;
      input_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_input.adim[i]-1;
    }
    output_domain.dim = sub_output.numDim;
    for (int i = 0; i < sub_output.numDim; i++) {
      output_domain.rect_data[i] = 0;
      output_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_output.adim[i]-1;
    }
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  }
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_grad_ptr != NULL);
  float* output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_grad_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_output.get_volume());
  };
  auto backward = [&] {
    backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr,
        sub_output.get_volume());
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Elewise Unary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
         name, sub_output.get_volume(), forward_time, backward_time);
  return true;
}
