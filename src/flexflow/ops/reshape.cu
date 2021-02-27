/* Copyright 2020 Stanford, Facebook
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

Tensor FFModel::reshape(const Tensor& input,
                        const std::vector<int>& shape,
                        const char* name)
{
  layers.push_back(
    std::unique_ptr<Op>(
      new Reshape(*this, input, shape, name)
    )
  );
  return layers.back()->outputs[0];
}

Reshape::Reshape(FFModel& model,
                 const Tensor& input,
                 const std::vector<int>& shape,
                 const char* name)
: Op(model, OP_RESHAPE, name, input)
{
  numOutputs = 1;
  numWeights = 0;
  outputs[0].numDim = (int)shape.size();
  size_t volume = 1;
  for (int i = 0; i < outputs[0].numDim; i++) {
    outputs[0].adim[i] = shape[outputs[0].numDim-1-i];
    volume *= (size_t)outputs[0].adim[i];
  }
  assert(volume == inputs[0].get_volume());
}

/*static*/
void Reshape::forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Reshape::backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements)
{
  float alpha = 1.0f;
  apply_add_with_scale<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input_grad_ptr, output_grad_ptr, num_elements, alpha);

}

bool Reshape::measure_compute_time(Simulator* sim,
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
  assert(input_ptr != NULL);
  float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_grad_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_grad_ptr != NULL);
  assert (sub_output.get_volume() == sub_input.get_volume());
  size_t num_elements = sub_input.get_volume();

  auto forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements);
  };
  auto backward = [&] {
    backward_kernel(input_grad_ptr, output_grad_ptr, num_elements);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  if (sim->verbosity >= SimulationVerbosity::ALL) {
    printf("[Measure Reshape] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        forward_time,
        backward_time);
  }

  return true;
}

