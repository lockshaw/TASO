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

Tensor FFModel::transpose(const Tensor& input,
                          const std::vector<int>& perm,
                          const char* name)
{
  layers.push_back(
    std::unique_ptr<Op>(
      new Transpose(*this, input, perm, name)
    )
  );
  return layers.back()->outputs[0];
}

Transpose::Transpose(FFModel& model,
                     const Tensor& input,
                     const std::vector<int>& _perm,
                     const char* name)
: Op(model, OP_TRANSPOSE, name, input)
{
  assert(_perm.size() == input.numDim);
  // Use Legion indexing to store perm
  for (int i = 0; i < input.numDim; i++)
    perm[i] = input.numDim - 1 - _perm[input.numDim - 1 - i];
  outputs[0].numDim = input.numDim;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = input.adim[perm[i]];
  numOutputs = 1;
  numWeights = 0;
}

void Transpose::init_meta(TransposeMeta *m, Domain const &in_domain, Domain const &out_domain) const
{
  /* for (int i = 0; i < out_domain.get_dim(); i++) { */
  /*   /1* assert(out_domain.get_hi()[i] == in_domain.get_hi()[this->perm[i]]); *1/ */
  /*   /1* assert(out_domain.get_lo()[i] == in_domain.get_lo()[this->perm[i]]); *1/ */
  /* } */
  m->num_dim = out_domain.get_dim();
  for (int i = 0; i < m->num_dim; i++)
    m->perm[i] = this->perm[i];
}

struct TransposeStrides
{
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM], perm[MAX_TENSOR_DIM];
};

__global__
void transpose_simple_kernel(coord_t volume,
                             const float* in_ptr,
                             float* out_ptr,
                             const TransposeStrides info,
                             const float beta)
{
  CUDA_KERNEL_LOOP(o_idx, volume)
  {
    coord_t i_idx = 0;
    coord_t t = o_idx;
    for (int i = info.num_dim-1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

/*static*/
void Transpose::forward_kernel(const TransposeMeta* m,
                               const float* input_ptr,
                               float* output_ptr,
                               Domain in_domain,
                               Domain out_domain)
{
  TransposeStrides info;
  info.num_dim = out_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (in_domain.get_hi()[i] - in_domain.get_lo()[i] + 1);
    int out_dim_size = (out_domain.get_hi()[i] - out_domain.get_lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[i] = m->perm[i];
  }
  transpose_simple_kernel<<<GET_BLOCKS(out_domain.get_volume()), CUDA_NUM_THREADS>>>(
      out_domain.get_volume(), input_ptr, output_ptr, info, 0.0f/*beta*/);
}

/*static*/
void Transpose::backward_kernel(const TransposeMeta* m,
                                float* input_grad_ptr,
                                const float* output_grad_ptr,
                                Domain in_grad_domain,
                                Domain out_grad_domain)
{
  TransposeStrides info;
  info.num_dim = in_grad_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (out_grad_domain.get_hi()[i] - out_grad_domain.get_lo()[i] + 1);
    int out_dim_size = (in_grad_domain.get_hi()[i] - in_grad_domain.get_lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[m->perm[i]] = i;
  }
  transpose_simple_kernel<<<GET_BLOCKS(in_grad_domain.get_volume()), CUDA_NUM_THREADS>>>(
      in_grad_domain.get_volume(), output_grad_ptr, input_grad_ptr, info, 1.0f/*beta*/);
}

bool Transpose::measure_compute_time(Simulator* sim,
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

  TransposeMeta *m = sim->transpose_meta;
  this->init_meta(m, sub_input.get_domain(), sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_grad_ptr != NULL);
  float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_grad_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_input.get_domain(), sub_output.get_domain());
  };
  auto backward = [&] {
    backward_kernel(m, input_grad_ptr, output_grad_ptr, sub_input.get_domain(), sub_output.get_domain());
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  if (sim->verbosity >= SimulationVerbosity::ALL) {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        forward_time,
        backward_time);
  }

  return true;
}
