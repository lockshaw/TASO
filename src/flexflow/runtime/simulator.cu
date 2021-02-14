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

#include "flexflow/simulator.h"
#include "flexflow/model.h"
#include "flexflow/cuda_helper.h"

using namespace flexflow;

Simulator::Simulator(FFConfig const &config,
                     FFHandler _handler)
: handler(_handler), offset(0), warmup_times(5), repeat_times(10)
{
  // Allocate simulator memory
  cudaMalloc(&base_ptr, config.simulator_work_space_size);
  capacity = config.simulator_work_space_size;

  float inter_gpu_bandwidth = 12 * 1024 * 1024.0f; /* B/ms*/
  float inter_node_bandwidth = 12 * 1024 * 1024.0f / config.numNodes; /* B/ms*/
  float gpu_dram_bandwidth = 16 * 1024 * 1024.0f; /* B/ms*/
  size_t max_num_tasks = 1024 * 1024;

  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  conv2d_meta = new Conv2DMeta(handler);
  linear_meta = new LinearMeta(handler, 4096);
  pool2d_meta = new Pool2DMeta(handler);
  ele_unary_meta = new ElementUnaryMeta(handler);
  ele_binary_meta = new ElementBinaryMeta(handler);
  concat_meta = new ConcatMeta(handler);
  num_nodes = config.numNodes;
  gpus_per_node = config.workersPerNode;
  total_num_gpus = num_nodes * gpus_per_node;
  verbose = config.simulationVerbose;
  // Create GPU compute device
  for (int i = 0; i < num_nodes; i++)
    for (int j = 0; j < gpus_per_node; j++) {
      id_to_compute_device[i*gpus_per_node+j] = new Device(Device::DEVICE_GPU,
          i, i*gpus_per_node+j);
    }
  // Create inter GPU comm devices:
  for (int i = 0; i < total_num_gpus; i++)
    for (int j = 0; j < total_num_gpus; j++) {
      Device* src = id_to_compute_device[i];
      Device* dst = id_to_compute_device[j];
      if (src->node_id == dst->node_id && src != dst) {
        int hash = i * total_num_gpus + j;
        ids_to_inter_gpu_comm_device[hash] = new Device(Device::DEVICE_COMM,
            inter_gpu_bandwidth);
      }
    }
  // Create gpu<->dram comm devices
  for (int i = 0; i < total_num_gpus; i++) {
    id_to_gputodram_comm_device[i] = new Device(Device::DEVICE_COMM,
        gpu_dram_bandwidth);
    id_to_dramtogpu_comm_device[i] = new Device(Device::DEVICE_COMM,
        gpu_dram_bandwidth);
  }
  // Create inter node comm devices
  for (int i = 0; i < num_nodes; i++)
    for (int j = 0; j < num_nodes; j++)
      if (i != j) {
        int hash = i * total_num_gpus + j;
        ids_to_inter_node_comm_device[hash] = new Device(Device::DEVICE_COMM,
            inter_node_bandwidth);
      }
  // Initialize task manager
  task_manager = new TaskManager(max_num_tasks);
}
