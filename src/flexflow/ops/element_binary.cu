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

Tensor FFModel::binary(OperatorType op,
                       const Tensor& in1,
                       const Tensor& in2,
                       char const *name)
{
  ElementBinary *ele = new ElementBinary(*this, op, in1, in2, name);
  layers.push_back(ele);
  return ele->outputs[0];
}

Tensor FFModel::add(const Tensor& in1,
                    const Tensor& in2,
                    char const *name)
{
  return this->binary(OP_EW_ADD, in1, in2, name);
}

Tensor FFModel::subtract(const Tensor& in1,
                         const Tensor& in2,
                         char const *name)
{
  return this->binary(OP_EW_SUB, in1, in2, name);
}

Tensor FFModel::multiply(const Tensor& in1,
                         const Tensor& in2,
                         char const *name)
{
  return this->binary(OP_EW_MUL, in1, in2, name);
}

Tensor FFModel::divide(const Tensor& in1,
                       const Tensor& in2,
                       char const *name)
{
  return this->binary(OP_EW_DIV, in1, in2, name);
}

ElementBinary::ElementBinary(FFModel& model,
                             OperatorType _op_type,
                             const Tensor& in1,
                             const Tensor& in2,
                             const char* name)
: Op(
    model,
    _op_type,
    name,
    in1,
    in2
  ),
  op_type(_op_type),
  profiling(model.config.profiling)
{
  //TODO: implement broadcast op
  numOutputs = 1;
  numWeights = 0;
  assert(in1.numDim == in2.numDim);
  int dim = in1.numDim;
  outputs[0].numDim = in1.numDim;
  printf("Op: %s\n", name);
  for (int i = 0; i < dim; i++) {
    printf("%d %d\n", in1.adim[i], in2.adim[i]);
  }
  for (int i = 0; i < dim; i++) {
    assert(in1.adim[i] == in2.adim[i]);
    outputs[0].adim[i] = in1.adim[i];
  }
}

__global__
void elewise_binary_forward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   OperatorType type,
                                   const float* in1,
                                   const float* in2,
                                   float* out)
{
    switch (type) {
      case OP_EW_ADD:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
	}
        break;
      }
      case OP_EW_SUB:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
        }
        break;
      }
      case OP_EW_MUL:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * in1[i] * in2[i] + beta * out[i];
        }
        break;
      }
      case OP_EW_DIV:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] / in2[i]) + beta * out[i];
        }
        break;
      }
      default:
        assert(false);
    }
}

/*static*/
void ElementBinary::forward_kernel(const ElementBinaryMeta* m,
                                   const float* in1_ptr,
                                   const float* in2_ptr,
                                   float* out_ptr)
{
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (m->op_type) {
    case OP_EW_SUB:
      alpha2 = -1.0f;
      break;
    case OP_EW_ADD:
    case OP_EW_MUL:
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->inputTensor, in1_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->outputTensor, out_ptr));
}

__global__
void elewise_binary_backward_kernel(coord_t volume,
                                    const float alpha,
                                    const float beta,
                                    OperatorType type,
                                    const float* out_grad,
                                    const float* in1,
                                    const float* in2,
                                    float* in1_grad,
                                    float* in2_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EW_ADD:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_SUB:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_MUL:
      {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_DIV:
      {
        in1_grad[i] = alpha * out_grad[i] / in2[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] * in1[i] / (in2[i] * in2[i]) + beta * in2_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementBinary::backward_kernel(const ElementBinaryMeta* m,
                                    const float* out_grad_ptr,
                                    const float* in1_ptr,
                                    const float* in2_ptr,
                                    float* in1_grad_ptr,
                                    float* in2_grad_ptr)
{
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
  switch (m->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->inputTensor, in1_grad_ptr));
  switch (m->op_type) {
    case OP_EW_ADD:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_SUB:
      alpha1 = -1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in1_ptr,
      &beta, m->inputTensor, in2_grad_ptr));
}

ElementBinaryMeta::ElementBinaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  op_type = OP_ANY;
}

bool ElementBinary::measure_compute_time(Simulator* sim,
                                         const ParallelConfig& pc,
                                         float& forward_time,
                                         float& backward_time)
{
  Tensor sub_output, sub_input1, sub_input0;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input0, op_type))
    return false;
  if (!inputs[1].get_input_sub_tensor(pc, sub_input1, op_type))
    return false;
  ElementBinaryMeta* m = sim->ele_binary_meta;
  m->op_type = op_type;
  cudnnOpTensorOp_t mode;
  switch (op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc, mode,
      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  Domain input_domain = sub_input0.get_domain();
  Domain output_domain = sub_output.get_domain();
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  sim->free_all();
  float* input0_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_ptr != NULL);
  float* input0_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_grad_ptr != NULL);
  float* input1_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  float* input1_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input1_grad_ptr != NULL);
  float* output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input0_ptr, input1_ptr, output_ptr);
  };
  auto backward = [&] {
    backward_kernel(m, output_ptr, input0_ptr, input1_ptr, input0_grad_ptr, input1_grad_ptr);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Elewise Binary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
         name, sub_output.get_volume(), forward_time, backward_time);

  return true;
}
