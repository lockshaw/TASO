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
//#include "flexflow/model.h"
#include "taso/ops.h"
#include "flexflow/cuda_helper.h"
#include "flexflow/simulator.h"

using namespace std;


/* Tensor::Tensor(void) */
/* { */
/*   numDim = 0; */
/*   for (int i = 0; i < MAX_TENSOR_DIM; i++) { */
/*     adim[i] = 0; */
/*     //pdim[i] = 0; */
/*   } */
/*   region = LogicalRegion::NO_REGION; */
/*   region_grad = LogicalRegion::NO_REGION; */
/*   part = LogicalPartition::NO_PART; */
/*   part_grad = LogicalPartition::NO_PART; */
/*   owner_op = NULL; */
/*   owner_idx = 0; */
/*   //physical_region.impl = NULL; */
/* } */

/* Tensor& Tensor::operator=(const Tensor& rhs) */
/* { */
/*   numDim = rhs.numDim; */
/*   for (int i = 0; i < numDim; i++) */
/*     adim[i] = rhs.adim[i]; */
/*   data_type = rhs.data_type; */
/*   owner_op = rhs.owner_op; */
/*   owner_idx = rhs.owner_idx; */
/*   region = rhs.region; */
/*   region_grad = rhs.region_grad; */
/*   part = rhs.part; */
/*   part_grad = rhs.part_grad; */
/*   physical_region = rhs.physical_region; */
/*   return *this; */
/* } */

/* template<typename T> */
/* T* Tensor::get_raw_ptr(FFConfig &config) */
/* { */
/*   Context ctx = config.lg_ctx; */
/*   Runtime* runtime = config.lg_hlr; */
/*   RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region); */
/*   region_req.add_field(FID_DATA); */
/*   T *raw_ptr = NULL; */
/*   if (numDim == 1) { */
/*     TensorAccessorW<T, 1> acc(physical_region, region_req, FID_DATA, ctx, runtime, true); */
/*     raw_ptr = (T*)acc.ptr; */
/*   } else if (numDim == 2) { */
/*     TensorAccessorW<T, 2> acc(physical_region, region_req, FID_DATA, ctx, runtime, true); */
/*     raw_ptr = (T*)acc.ptr; */
/*   } else if (numDim == 3) { */
/*     TensorAccessorW<T, 3> acc(physical_region, region_req, FID_DATA, ctx, runtime, true); */
/*     raw_ptr = (T*)acc.ptr; */
/*   } else if (numDim == 4) { */
/*     TensorAccessorW<T, 4> acc(physical_region, region_req, FID_DATA, ctx, runtime, true); */
/*     raw_ptr = (T*)acc.ptr; */
/*   } else { */
/*     printf("wrong numDim %d", numDim); */
/*     assert(0); */
/*   } */
/*   return raw_ptr; */
/* } */

/* void Tensor::attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major) */
/* { */
/*   Context ctx = config.lg_ctx; */
/*   Runtime* runtime = config.lg_hlr; */
/*   AttachLauncher launcher(EXTERNAL_INSTANCE, region, region); */
/*   std::vector<FieldID> fields(1, FID_DATA); */
/*   const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine()) */
/*        .has_affinity_to(runtime->get_executing_processor(ctx)) */
/*        .only_kind(Memory::SYSTEM_MEM) */
/*        .first(); */
/*   launcher.attach_array_soa(raw_ptr, column_major, */
/*                             fields, local_sysmem); */
/*   physical_region = runtime->attach_external_resource(ctx, launcher); */
/* } */

/* void Tensor::detach_raw_ptr(FFConfig &config) */
/* { */
/*   Context ctx = config.lg_ctx; */
/*   Runtime* runtime = config.lg_hlr; */
/*   runtime->detach_external_resource(ctx, physical_region); */
/* } */

bool Tensor::get_input_sub_tensor(const ParallelConfig& pc,
                                  Tensor& tensor,
                                  OperatorType type)
{
  //TODO: consider reduction dim for conv2d and linear
  switch (type) {
    case OP_FLAT:
    case OP_RESHAPE:
      {
        assert (pc.nDims == 2 && "Invalid dimension for parallel config of OP_FLAT");
        int nonBatchDim = pc.dim[0];
        tensor.numDim = numDim;
        assert (nonBatchDim == 1 && "I'm not sure this is correct otherwise");
        if (adim[numDim - 1] % nonBatchDim != 0) {
          printf("Could not get input subtensor because the dimension is not divisiable: %d %% %d != 0\n", adim[numDim - 1], nonBatchDim);
          return false;
        }
        tensor.adim[numDim - 1] = adim[numDim - 1] / pc.dim[1];
        for (int i = numDim - 2; i >= 0; i--) {
          tensor.adim[i] = adim[i];
        }
      }
      break;
    default:
      {
        if (pc.nDims != numDim) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, numDim);
          return false;
        }
        for (int i = 0; i < numDim; i++) {
          if (adim[i] % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", adim[i], pc.dim[i]);
            return false;
          }
        }
        tensor.numDim = numDim;
        for (int i = 0; i < numDim; i++) {
          tensor.adim[i] = adim[i] / pc.dim[i];
        }
      }
      break;
  }
  return true;
}

bool Tensor::get_output_sub_tensor(const ParallelConfig& pc,
                                   Tensor& tensor,
                                   OperatorType type)
{
  if (pc.nDims != numDim) {
    printf("Could not get output subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, numDim);
    return false;
  }
  for (int i = 0; i < numDim; i++) {
    if (adim[i] % pc.dim[i] != 0) {
      printf("Could not get output subtensor because the given dimension is not divisible: %d %% %d != 0\n", adim[i], pc.dim[i]);
      return false;
    }
  }
  tensor.numDim = numDim;
  for (int i = 0; i < numDim; i++)
    tensor.adim[i] = adim[i] / pc.dim[i];
  return true;
}

Domain Tensor::get_domain() const
{
  Domain d;
  d.dim = this->numDim;
  for (int i = 0; i < this->numDim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+Domain::MAX_RECT_DIM] = this->adim[i] - 1;
  }
  return d;
}

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const char* _name, */
/*        const Tensor& _input) */
/* : op_type(_op_type), numInputs(1), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   std::strcpy(name, pcname.c_str()); */
/*   inputs[0] = _input; */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const OpBase* shared_op, */
/*        const char* _name, */
/*        const Tensor& _input) */
/* : op_type(_op_type), numInputs(1), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   if (shared_op == NULL) { */
/*     pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   } else { */
/*     pcname = std::string(shared_op->name); */
/*   } */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   std::strcpy(name, pcname.c_str()); */
/*   inputs[0] = _input; */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const char* _name, */
/*        const Tensor& _input1, */
/*        const Tensor& _input2) */
/* : op_type(_op_type), numInputs(2), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   std::strcpy(name, pcname.c_str()); */
/*   inputs[0] = _input1; */
/*   inputs[1] = _input2; */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const char* _name, */
/*        const Tensor& _input1, */
/*        const Tensor& _input2, */
/*        const Tensor& _input3) */
/* : op_type(_op_type), numInputs(3), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   std::strcpy(name, pcname.c_str()); */
/*   inputs[0] = _input1; */
/*   inputs[1] = _input2; */
/*   inputs[2] = _input3; */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const char* _name, */
/*        int n, const Tensor* _inputs) */
/* : op_type(_op_type), numInputs(n), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   assert(n <= MAX_NUM_INPUTS); */
/*   std::strcpy(name, pcname.c_str()); */
/*   for (int i = 0; i < n; i++) */
/*     inputs[i] = _inputs[i]; */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Op::Op(Model& model, */
/*        OperatorType _op_type, */
/*        const char* _name, */
/*        int _numInputs) */
/* : op_type(_op_type), numInputs(_numInputs), numWeights(0), numOutputs(1) */
/* { */
/*   std::string pcname; */
/*   if (_name == NULL) { */
/*     pcname = model.get_operator_type_name(op_type); */
/*   } else { */
/*     pcname = std::string(_name); */
/*   } */
/*   pcname = pcname + "_" + std::to_string(model.op_global_guid++); */
/*   assert(pcname.length() < MAX_OPNAME); */
/*   std::strcpy(name, pcname.c_str()); */
/*   //for (int i = 0; i < numInputs; i++) { */
/*   //  trainableInputs[i] = true; */
/*   //  resetInputGrads[i] = true; */
/*   //} */
/*   for (int i = 0; i < MAX_NUM_OUTPUTS; i++) { */
/*     outputs[i].owner_op = this; */
/*     outputs[i].owner_idx = i; */
/*   } */
/*   for (int i = 0; i < MAX_NUM_WORKERS; i++) */
/*     meta[i] = NULL; */
/* } */

/* Parameter* Op::get_parameter(int index) */
/* { */
/*   assert(index < numWeights); */
/*   return &weights[index]; */
/* } */

/* ParallelConfig Op::get_data_parallel_config(const Model& ff) const */
/* { */
/*   int num_parts = ff.config.workersPerNode * ff.config.numNodes; */
/*   ParallelConfig pc; */
/*   pc.device_type = ParallelConfig::GPU; */
/*   pc.nDims = outputs[0].numDim; */
/*   for (int i = 0; i < pc.nDims; i++) */
/*     pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1; */
/*   for (int i = 0; i < num_parts; i++) */
/*     pc.device_ids[i] = i; */
/*   return pc; */
/* } */

/* ParallelConfig Op::get_random_parallel_config(const Model& ff) const */
/* { */
/*   std::vector<int> candidates; */
/*   int batch_size = outputs[0].adim[outputs[0].numDim-1]; */
/*   for (int i = 1; i <= ff.config.workersPerNode; i++) */
/*     if (ff.config.workersPerNode % i == 0) { */
/*       if (batch_size % i != 0) */
/*         continue; */
/*       candidates.push_back(i); */
/*     } */
/*   for (int i = 1; i <= ff.config.numNodes; i++) */
/*     if (ff.config.numNodes % i == 0) { */
/*       if (batch_size % (i * ff.config.workersPerNode) != 0) */
/*         continue; */
/*       candidates.push_back(i * ff.config.workersPerNode); */
/*     } */
/*   assert(candidates.size() > 0); */
/*   int idx = std::rand() % candidates.size(); */
/*   int num_parts = candidates[idx]; */
/*   ParallelConfig pc; */
/*   pc.device_type = ParallelConfig::GPU; */
/*   pc.nDims = outputs[0].numDim; */
/*   for (int i = 0; i < pc.nDims; i++) */
/*     pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1; */
/*   int total_num_devices = ff.config.workersPerNode * ff.config.numNodes; */
/*   int start_idx = std::rand() % (total_num_devices - num_parts + 1); */
/*   for (int i = 0; i < num_parts; i++) */
/*     pc.device_ids[i] = start_idx + i; */
/*   return pc; */
/* } */


/* Domain Op::get_weight_tensor_shape(const ParallelConfig& pc, */
/*                                    int weight_idx, int part_idx) */
/* { */
/*   // Default data parallel weight replication */
/*   assert(weight_idx < numWeights); */
/*   Domain d; */
/*   d.dim = weights[weight_idx].numDim; */
/*   for (int i = 0; i < d.dim; i++) { */
/*     d.rect_data[i] = 0; */
/*     d.rect_data[i+d.dim] = weights[weight_idx].adim[i] - 1; */
/*   } */
/*   return d; */
/* } */

Model::Model(FFConfig& _config)
: op_global_guid(100), config(_config)
{
  // Load strategy file
  int start_dim = 1, end_dim = 4;
#if MAX_TENSOR_DIM >= 5
  end_dim = 5;
#endif
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::GPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = config.workersPerNode * config.numNodes;
    for (int j = 0; j < pc.dim[pc.nDims-1]; j++)
      pc.device_ids[j] = j;
    config.strategies[FFConfig::DataParallelism_GPU_1D+i-1] = pc;
  }
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::CPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = config.cpusPerNode * config.numNodes;
    for (int j = 0; j < pc.dim[pc.nDims-1]; j++)
      pc.device_ids[j] = j;
    config.strategies[FFConfig::DataParallelism_CPU_1D+i-1] = pc;
  }

  // Build training dataset
  //if (config.datasetPath.length() == 0) {
  //  dataLoader = NULL;
  //} else {
  //  dataLoader = new DataLoader(config.datasetPath);
  //}
}

/*
template<int NDIM>
Tensor Model::create_tensor(const int dims[],
                              DataType data_type,
                              OpBase* owner_op,
                              bool create_grad)
{
  ParallelConfig pc;
  assert(config.find_parallel_config(NDIM, pc_name, pc));
  IndexSpaceT<NDIM> task_is = IndexSpaceT<NDIM>(get_or_create_task_is(pc));
  return create_tensor<NDIM>(dims, task_is, data_type, create_grad);
}
*/

void Model::optimize(Simulator* simulator,
                       std::map<OpBase*, ParallelConfig>& best,
                       size_t budget, float alpha) const
{
  // Start from data parallel
  std::map<OpBase*, ParallelConfig> current, next;
  float best_runtime = simulator->simulate_runtime(this, best);
  current = best;
  float current_runtime = best_runtime;
  for (size_t iter = 0; iter < budget; iter++) {
    rewrite(current, next);
    float next_runtime = simulator->simulate_runtime(this, next);
    if (iter % 100 == 0) {
      printf("iter(%zu) cur(%.2lf) next(%.2lf) best(%.2lf)\n", iter,
             current_runtime, next_runtime, best_runtime);
    }
    float rn = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    //float ratio = (next_runtime - current_runtime) / current_runtime;
    float diff = (next_runtime - current_runtime);
    if (next_runtime < best_runtime) {
      best_runtime = next_runtime;
      best = next;
    }
    if (next_runtime < current_runtime) {
      current = next;
      current_runtime = next_runtime;
    } else if (rn < std::exp(-alpha * diff)) {
      current = next;
      current_runtime = next_runtime;
    }
  }
  printf("=========== Best Discovered Strategy ==========\n");
  simulator->simulate_runtime(this, best, this->config.export_strategy_task_graph_file);
  std::map<OpBase*, ParallelConfig>::const_iterator it;
  for (it = best.begin(); it != best.end(); it++) {
    printf("[%s] num_dims(%d) dims[", it->first->name, it->second.nDims);
    for (int i = 0; i < it->second.nDims; i++)
      if (i < it->second.nDims - 1)
        printf("%d,", it->second.dim[i]);
      else
        printf("%d", it->second.dim[i]);
    printf("] device_ids[");
    for (int i = 0; i < it->second.num_parts(); i++)
      if (i < it->second.num_parts() - 1)
        printf("%d,", it->second.device_ids[i]);
      else
        printf("%d", it->second.device_ids[i]);
    printf("]\n");
  }
  printf("============= MCMC Search Finished ============\n\n");
}

std::string Model::get_operator_type_name(OperatorType type) const
{
  switch(type) {
    case OP_CONV2D: return "Conv2D";
    case OP_DROPOUT: return "Dropout";
    case OP_LINEAR: return "Dense";
    case OP_BATCHMATMUL: return "BatchMatMul";
    case OP_POOL2D: return "Pool2D";
    case OP_RELU: return "ReLU";
    case OP_SIGMOID: return "Sigmoid";
    case OP_TANH: return "Tanh";
    case OP_ELU: return "Elu";
    case OP_FLAT: return "Flat";
    case OP_SOFTMAX: return "Softmax";
    case OP_BATCHNORM: return "BatchNorm";
    case OP_CONCAT: return "Concat";
    case OP_SPLIT: return "Split";
    case OP_EMBEDDING: return "Embedding";
    case OP_RESHAPE: return "Reshape";
    case OP_REVERSE: return "Reverse";
    case OP_TRANSPOSE: return "Transpose";
    case OP_EW_ADD: return "Add";
    case OP_EW_MUL: return "Mul";
    case OP_MATMUL: return "Matmul";
    case OP_MUL: return "Mul";
    case OP_ENLARGE: return "Enlarge";
    case OP_SQUEEZE: return "Squeeze";
    case OP_UNSQUEEZE: return "Unsqueeze";
    case OP_EW_SUB: return "Sub";
    case OP_EW_DIV: return "Div";
    case OP_EW_EQUAL: return "Equal";
    case OP_EW_GREATER: return "Greater";
    case OP_EW_LESS: return "Less";
    case OP_EW_MAX: return "Max";
    case OP_EW_MIN: return "Min";
    case OP_REDUCE_ARGMAX: return "ReduceArgMax";
    case OP_REDUCE_ARGMIN: return "ReduceArgMin";
    case OP_REDUCE_MAX: return "ReduceMax";
    case OP_REDUCE_MEAN: return "ReduceMean";
    case OP_REDUCE_MIN: return "ReduceMin";
    case OP_REDUCE_PROD: return "ReduceProd";
    case OP_REDUCE_SUM: return "ReduceSum";
    case OP_PAD: return "Pad";
    case OP_SHAPE: return "Shape";
    case OP_SIZE: return "Size";
    case OP_TOPK: return "TopK";
    case OP_WHERE: return "Where";
    case OP_CEIL: return "Ceil";
    case OP_CAST: return "Cast";
    case OP_EXP: return "Exp";
    case OP_ROUND: return "Round";
    case OP_LOG: return "Log";
    case OP_LOGICAL_NOT: return "LogicalNot";
    case OP_SQRT: return "Sqrt";
    case OP_LEAKYRELU: return "LeakyReLU";
    case OP_SLICE: return "Slice";
    case OP_RESIZE: return "Resize";
    case OP_PRELU: return "PReLU";
    case OP_MULTIHEAD_ATTENTION: return "MultiHeadAttention";
    default: assert(false && "Not supported Operator type"); return "Unsupported";
  }
}

// ========================================================
// class FFConfig
// ========================================================

// Default Config Parameters
struct DefaultConfig {
  const static int epochs = 1;
  const static int iterations = 1;
  const static int batchSize = 64;
  const static bool profiling = false;
  const static bool debug = false;
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int cpusPerNode = 0;
  const static size_t searchBudget = 0;
  const static size_t simulatorWorkSpaceSize = (size_t)2 * 1024 * 1024 * 1024; //2GB
  constexpr static float searchAlpha = 1.0f;
  const static bool searchOverlapBackwardUpdate = false;
  const static bool enableSampleParallel = true;
  const static bool enableParameterParallel = false;
  const static bool enableAttributeParallel = false;
};

FFConfig::FFConfig()
{
  epochs = DefaultConfig::epochs;
  iterations = DefaultConfig::iterations;
  batchSize = DefaultConfig::batchSize;
  profiling = DefaultConfig::profiling;
  learningRate = DefaultConfig::learningRate;
  weightDecay = DefaultConfig::weightDecay;
  workSpaceSize = DefaultConfig::workSpaceSize;
  numNodes = DefaultConfig::numNodes;
  cpusPerNode = DefaultConfig::cpusPerNode;
  workersPerNode = DefaultConfig::workersPerNode;
  simulator_work_space_size = DefaultConfig::simulatorWorkSpaceSize;
  search_budget = DefaultConfig::searchBudget;
  search_alpha = DefaultConfig::searchAlpha;
  search_overlap_backward_update = DefaultConfig::searchOverlapBackwardUpdate;
  enable_sample_parallel = DefaultConfig::enableSampleParallel;
  enable_parameter_parallel = DefaultConfig::enableParameterParallel;
  enable_attribute_parallel = DefaultConfig::enableAttributeParallel;

  import_strategy_file = "";
  export_strategy_file = "";
  export_strategy_task_graph_file = "";
  dataset_path = "";
  syntheticInput = false;
  perform_fusion = false;
}

void FFConfig::parse_args(char **argv, int argc)
{
  for (int i = 1; i < argc; i++)
  {
    if ((!strcmp(argv[i], "-e")) || (!strcmp(argv[i], "--epochs"))) {
      epochs = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-i")) || (!strcmp(argv[i], "--iterations"))) {
      iterations = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-b")) || (!strcmp(argv[i], "--batch-size"))) {
      batchSize = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--lr")) || (!strcmp(argv[i], "--learning-rate"))) {
      learningRate = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--wd")) || (!strcmp(argv[i], "--weight-decay"))) {
      weightDecay = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-p")) || (!strcmp(argv[i], "--print-freq"))) {
      printFreq = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-d")) || (!strcmp(argv[i], "--dataset"))) {
      dataset_path = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--budget")) || (!strcmp(argv[i], "--search-budget"))) {
      search_budget =(size_t) atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--alpha")) || (!strcmp(argv[i], "--search-alpha"))) {
      search_alpha = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--import")) || (!strcmp(argv[i], "--import-strategy"))) {
      import_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--export")) || (!strcmp(argv[i], "--export-strategy"))) {
      export_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--enable-parameter-parallel"))) {
      enable_parameter_parallel = true;
      continue;
    }
    if ((!strcmp(argv[i], "--enable-attribute-parallel"))) {
      enable_parameter_parallel = true;
      continue;
    }
    if (!strcmp(argv[i], "-ll:gpu"))
    {
      workersPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--nodes"))
    {
      numNodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:cpu"))
    {
      cpusPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--profiling"))
    {
      profiling = true;
      continue;
    }
    if (!strcmp(argv[i], "--fusion"))
    {
      perform_fusion = true;
      continue;
    }
    if (!strcmp(argv[i], "--overlap"))
    {
      search_overlap_backward_update = true;
      continue;
    }
    if (!strcmp(argv[i], "--taskgraph")) {
      export_strategy_task_graph_file = std::string(argv[++i]);
      continue;
    }
  }
}
