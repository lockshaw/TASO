/* Copyright 2019 Stanford
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

#include "taso/ops.h"
#include "taso/substitution.h"
#include "taso/boundedpq.h"
#include <iostream>
#include <fstream>
#include <memory>
#include "flexflow/config.h"

using namespace std;

const SplitInfo SplitInfo::NO_SPLIT = SplitInfo();

OpBase::OpBase(Model* _model, OpType _type)
: numInputs(0), model(_model), type(_type), runtime(0.0f)
{
  // Assume only constant operator can take no inputs
  assert(type == OP_CONSTANT_POOL);
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input,
               Model* _model, OpType _type)
: numInputs(1), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               Model* _model, OpType _type)
: numInputs(2), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               const Tensor& _input2,
               Model* _model, OpType _type)
: numInputs(3), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               const Tensor& _input2,
               const Tensor& _input3,
               Model* _model, OpType _type)
: numInputs(5), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  inputs[3] = _input3;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}


OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               const Tensor& _input2,
               const Tensor& _input3,
               const Tensor& _input4,
               Model* _model, OpType _type)
: numInputs(5), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  inputs[3] = _input3;
  inputs[4] = _input4;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(int n, Tensor* _inputs, Model* _model, OpType _type)
: numInputs(n), model(_model), type(_type), runtime(0.0f)
{
  assert(n <= MAX_NUM_INPUTS);
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

bool OpBase::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    default:
      return false;
  }
}

bool OpBase::get_float_parameter(PMParameter para, float* value)
{
  switch (para) {
    default:
      return false;
  }
}

bool OpBase::get_input_parameter(TNParameter tnp, DIMParameter dim, int* value)
{
  int inputIdx = 0, dimIdx = 0;
  switch (tnp) {
    case IN_5:
      inputIdx++;
    case IN_4:
      inputIdx++;
    case IN_3:
      inputIdx++;
    case IN_2:
      inputIdx++;
    case IN_1:
      inputIdx++;
    case IN_0:
      break;
    default:
      return false;
  }
  if (inputIdx >= numInputs) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx].numDim;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx].numDim) return false;
  *value = inputs[inputIdx].dim[dimIdx];
  return true;
}

ParallelConfig OpBase::get_data_parallel_config(FFConfig const &conf) const {
  int num_parts = conf.workersPerNode * conf.numNodes;
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0].numDim;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = i;
  return pc;
}

ParallelConfig OpBase::get_random_parallel_config(FFConfig const &conf) const {
  std::vector<int> candidates;
  int batch_size = outputs[0].adim[outputs[0].numDim-1];
  for (int i = 1; i <= conf.workersPerNode; i++)
    if (conf.workersPerNode % i == 0) {
      if (batch_size % i != 0)
        continue;
      candidates.push_back(i);
    }
  for (int i = 1; i <= conf.numNodes; i++)
    if (conf.numNodes % i == 0) {
      if (batch_size % (i * conf.workersPerNode) != 0)
        continue;
      candidates.push_back(i * conf.workersPerNode);
    }
  assert(candidates.size() > 0);
  int idx = std::rand() % candidates.size();
  int num_parts = candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0].numDim;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  int total_num_devices = conf.workersPerNode * conf.numNodes;
  int start_idx = std::rand() % (total_num_devices - num_parts + 1);
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

Domain OpBase::get_input_tensor_shape(const ParallelConfig& pc,
                                  int input_idx, int part_idx)
{
  assert(input_idx < numInputs);
  Domain d;
  d.dim = inputs[input_idx].numDim;
  if (pc.nDims == d.dim) {
    for (int i = 0; i < d.dim; i++) {
      // Assume an equal partitioning
      assert(inputs[input_idx].adim[i] % pc.dim[i] == 0);
      int dim_size = inputs[input_idx].adim[i] / pc.dim[i];
      d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
      part_idx = part_idx / pc.dim[i];
    }
  } else {
    // Require data parallel when dims mismatch
    for (int i = 0; i < pc.nDims-1; i++)
      assert(pc.dim[i] == 1);
    for (int i = 0; i < d.dim-1; i++) {
      int dim_size = inputs[input_idx].adim[i];
      d.rect_data[i] = 0;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    }
    // Assume an equal partitioning
    assert(inputs[input_idx].adim[d.dim-1] % pc.dim[pc.nDims-1] == 0);
    assert(part_idx < pc.dim[pc.nDims-1]);
    int dim_size = inputs[input_idx].adim[d.dim-1] / pc.dim[pc.nDims-1];
    d.rect_data[d.dim - 1] = part_idx * dim_size;
    d.rect_data[2*d.dim - 1] = d.rect_data[d.dim-1] + dim_size - 1;
    part_idx = part_idx / pc.dim[pc.nDims-1];
  }
  assert(part_idx == 0);
  return d;
}

Domain OpBase::get_output_tensor_shape(const ParallelConfig& pc,
                                   int output_idx, int part_idx)
{
  assert(output_idx < numOutputs);
  Domain d;
  d.dim = outputs[output_idx].numDim;
  // Assume pc dim matches output dim
  assert(d.dim == pc.nDims);
  for (int i = 0; i < d.dim; i++) {
    // Assume an equal partitioning
    assert(outputs[output_idx].adim[i] % pc.dim[i] == 0);
    int dim_size = outputs[output_idx].adim[i] / pc.dim[i];
    d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
    d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    part_idx = part_idx / pc.dim[i];
  }
  assert(part_idx == 0);
  return d;
}

