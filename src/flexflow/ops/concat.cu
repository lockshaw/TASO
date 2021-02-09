/* Copyright 2017 Stanford, NVIDIA
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

Tensor FFModel::concat(int n,
                       const Tensor* tensors,
                       int axis,
                       const char *name)
{
  Concat *cat = new Concat(*this, n, tensors, axis, name);
  layers.push_back(cat);
  return cat->outputs[0];
}

Concat::Concat(FFModel& model,
               int _n, const Tensor* _tensors,
               int _axis,
               const char* name)
: Op(model, OP_CONCAT, name, _n, _tensors), axis(_axis),
   profiling(model.config.profiling)
{
  //TODO: swich to use the Legion dim ordering
  int num_dim = inputs[0].numDim;
  outputs[0].numDim = num_dim;
  for (int i = 0; i < num_dim; i++)
    outputs[0].adim[i] = inputs[0].adim[i];
  for (int i = 1; i < numInputs; i++)
    for (int j = 0; j < num_dim; j++) {
      if (j != num_dim - 1 - axis)
        assert(inputs[i].adim[j] == outputs[0].adim[j]);
      else
        outputs[0].adim[j] += inputs[i].adim[j];
    }
  numOutputs = 1;
  numWeights = 0;
  printf("Op: %s\n", name);
  printf("Input: ");
  for (int i = 0; i < inputs[0].numDim; i++) {
    printf("%d ", inputs[0].adim[i]);
  }
  printf("\nOutput: ");
  for (int i = 0; i < outputs[0].numDim; i++) {
    printf("%d ", outputs[0].adim[i]);
  }
  printf("\n");
}

void Concat::init_meta(ConcatMeta *m) const
{
  m->axis = this->outputs[0].numDim - 1 - this->axis;
}

void calc_blk_size(coord_t& num_blocks,
                   coord_t& blk_size,
                   Domain const &domain,
                   int axis)
{
  num_blocks = 1;
  blk_size = 1;
  coord_t const *lo = domain.get_lo();
  coord_t const *hi = domain.get_hi();
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis)
      blk_size *= (hi[d] - lo[d] + 1);
    else
      num_blocks *= (hi[d] - lo[d] + 1);
  }
}

/*static*/
void Concat::forward_kernel(float* output,
                            float const * const *inputs,
                            int num_inputs,
                            int axis,
                            const Domain& out_domain,
                            const Domain* in_domain)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);

  calc_blk_size(num_blocks, output_blk_size, out_domain, axis);

  coord_t input_num_blocks;
  for (int i = 0; i < num_inputs; i++) {
    calc_blk_size(input_num_blocks, input_blk_sizes[i], in_domain[i], axis);
    assert (input_num_blocks == num_blocks);
  }

  for (int i = 0; i < num_inputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
        output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
    //printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
    //       output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    output += input_blk_sizes[i];
  }
}

void Concat::backward_kernel(const float* output_grad,
                             float** input_grads,
                             int num_inputs,
                             int axis,
                             const Domain& out_grad_domain,
                             const Domain* in_grad_domain)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);

  calc_blk_size(num_blocks, output_blk_size, out_grad_domain, axis);

  coord_t input_num_blocks;
  for (int i = 0; i < num_inputs; i++) {
    calc_blk_size(input_num_blocks, input_blk_sizes[i], in_grad_domain[i], axis);
    assert (input_num_blocks == num_blocks);
  }

  for (int i = 0; i < num_inputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
        input_grads[i], output_grad, num_blocks, input_blk_sizes[i], output_blk_size);
    output_grad += input_blk_sizes[i];
  }

  //Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
  //Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
  //print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
  //print_tensor<2, float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

bool Concat::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  assert (numInputs <= MAX_NUM_INPUTS);
  Tensor sub_inputs[MAX_NUM_INPUTS], sub_output;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  for (int i = 0; i < numInputs; i++) {
    if (!inputs[i].get_input_sub_tensor(pc, sub_inputs[i], op_type)) {
      return false;
    }
  }

  ConcatMeta *m = sim->concat_meta;
  this->init_meta(m);

  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  float *input_grad_ptrs[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    input_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    assert (input_ptrs[i] != NULL);
    input_grad_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    assert (input_grad_ptrs[i] != NULL);
  }
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_grad_ptr != NULL);

  int axis = outputs[0].numDim - 1 - this->axis;

  Domain out_domain = sub_output.get_domain();
  Domain in_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    in_domains[i] = sub_inputs[i].get_domain();
  }

  auto forward = [&] {
    forward_kernel(output_ptr, input_ptrs, numInputs, axis, out_domain, in_domains);
  };
  auto backward = [&] {
    backward_kernel(output_grad_ptr, input_grad_ptrs, numInputs, axis, out_domain, in_domains);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Concat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
      name,
      forward_time,
      backward_time);

  return true;
}
