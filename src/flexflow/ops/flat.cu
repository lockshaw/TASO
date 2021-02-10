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

Tensor FFModel::flat(const Tensor& input,
                     const char* name)
{
  assert(input.numDim == 4);
  layers.push_back(
    std::unique_ptr<Op>(
      new Flat(*this, input, name)
    )
  );
  return layers.back()->outputs[0];
}

Flat::Flat(FFModel& model,
           const Tensor& _input,
           const char* name)
: Op(model, OP_FLAT, name, _input)
{
  assert(_input.numDim == 4);
  int out_dim = _input.adim[0] * _input.adim[1] * _input.adim[2];
  int batch_size = _input.adim[3];
  outputs[0].numDim = 2;
  outputs[0].adim[0] = out_dim;
  outputs[0].adim[1] = batch_size;
}

/*static*/
void Flat::forward_kernel(const float* input_ptr,
                          float* output_ptr,
                          size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
}

void Flat::backward_kernel(float* input_grad_ptr,
                           const float* output_grad_ptr,
                           size_t num_elements)
{
  float alpha = 1.0f;
  apply_add_with_scale<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input_grad_ptr, output_grad_ptr, num_elements, alpha);
}

bool Flat::measure_compute_time(Simulator* sim,
                                const ParallelConfig& pc,
                                float& forward_time,
                                float& backward_time)
{
  Tensor sub_input, sub_output;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_grad_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_grad_ptr != NULL);
  size_t num_elements = sub_output.get_volume();

  auto forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements);
  };
  auto backward = [&] {
    backward_kernel(input_grad_ptr, output_grad_ptr, num_elements);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Flat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
      name,
      forward_time,
      backward_time);

  return true;
}

Domain Flat::get_input_tensor_shape(const ParallelConfig& pc,
                                  int input_idx, int part_idx)
{
  assert(input_idx < numInputs);
  assert(pc.nDims == 2);
  // Currently assume data parallelism for Flat
  assert(pc.dim[0] == 1);
  Domain d;
  d.dim = inputs[input_idx].numDim;
  for (int i = 0; i < d.dim-1; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = inputs[input_idx].adim[i] - 1;
  }
  assert(inputs[input_idx].adim[d.dim-1] % pc.num_parts() == 0);
  int dim_size = inputs[input_idx].adim[d.dim-1] / pc.num_parts();
  d.rect_data[d.dim-1] = part_idx * dim_size;
  d.rect_data[2*d.dim-1] = d.rect_data[d.dim-1] + dim_size - 1;
  return d;
}
