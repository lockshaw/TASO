/* Copyright 2020 Facebook
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

void FFModel::split(const Tensor& input,
                    Tensor* outputs,
                    const std::vector<int>& splits,
                    int axis,
                    const char* name)
{
  Split* split = new Split(*this, input, splits, axis, name);
  layers.push_back(split);
  for (size_t i = 0; i < splits.size(); i++)
    outputs[i] = split->outputs[i];
}

Split::Split(FFModel& model,
             const Tensor& input,
             const std::vector<int>& splits,
             int _axis,
             const char* name)
: Op(model, OP_SPLIT, name, input)
{
  numOutputs = splits.size();
  // Use the Legion dim ordering
  axis = input.numDim - 1 - _axis;
  assert(axis >= 0);
  numWeights = 0;
  int split_size = 0;
  for (int i = 0; i < numOutputs; i++) {
    split_size += splits[i];
    outputs[i].numDim = input.numDim;
    for (int j = 0; j < input.numDim; j++)
      outputs[i].adim[j] = input.adim[j];
    outputs[i].adim[axis] = splits[i];
  }
  // Check split sizes
  assert(split_size == input.adim[axis]);
}

void calc_block_size(coord_t& num_blks,
                     coord_t& blk_size,
                     const Domain& domain,
                     int axis)
{
  num_blks = 1;
  blk_size = 1;
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis)
      blk_size *= (domain.get_hi()[d] - domain.get_lo()[d] + 1);
    else
      num_blks *= (domain.get_hi()[d] - domain.get_lo()[d] + 1);
  }
}

void Split::forward_kernel(float **out_ptrs,
                           float const *in_ptr,
                           coord_t const *out_blk_sizes,
                           coord_t in_blk_size,
                           coord_t num_blks,
                           int numOutputs)
{
  for (int i = 0; i < numOutputs; i++) {
    copy_with_stride<<<GET_BLOCKS(out_blk_sizes[i]*num_blks), CUDA_NUM_THREADS>>>(
        out_ptrs[i], in_ptr, num_blks, out_blk_sizes[i], in_blk_size);
    in_ptr += out_blk_sizes[i];
  }
}

bool Split::measure_compute_time(Simulator* sim,
                                 const ParallelConfig& pc,
                                 float& forward_time,
                                 float& backward_time)
{
  //TODO: implement measure_forward
  forward_time = 0.0f;
  backward_time = 0.0f;
  return false;
}
