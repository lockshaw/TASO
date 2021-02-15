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
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "flexflow/ffconst.h"
#include "flexflow/config.h"
#include <memory>
#include <fstream>
#include <vector>
#include <sstream>

namespace flexflow {
  class Conv2DMeta;
  class LinearMeta;
  class Pool2DMeta;
  class ElementUnaryMeta;
  class ElementBinaryMeta;
  class ConcatMeta;
  class Op;
  class FFModel;

  class Device {
  public:
    enum DeviceType {
      DEVICE_GPU,
      DEVICE_CPU,
      DEVICE_COMM,
    };
    Device(DeviceType type, int node_id, int gpu_id);
    Device(DeviceType type, float bandwidth);
  public:
    int node_id, gpu_id;
    float bandwidth;
    DeviceType type;
  };

  class SimTask {
  public:
    enum SimTaskType {
      TASK_FORWARD,
      TASK_BACKWARD,
      TASK_COMM,
      TASK_UPDATE,
      TASK_BARRIER,
    };
    SimTask();
    void add_next_task(SimTask* task);
  public:
    float ready_time, run_time;
    SimTaskType type;
    Device* device;
    int counter;
    std::vector<SimTask*> next_tasks;
    char const *op_name;
    std::string get_type_str() const;
    Device *src, *dst;
  };

  template <typename T>
  class DotFile {
  private:
    size_t node_id;
    bool is_closed;
    std::map<T,size_t> node_ids;
    std::set<T> described_nodes;
    std::unique_ptr<std::ostream> out;
    std::string get_node_name(size_t node_id) const {
      std::ostringstream s;
      s << "node" << node_id;
      return s.str();
    }
  public:
    DotFile() : node_id(0), is_closed(false) {}
    DotFile(std::string const &filename) : DotFile(std::unique_ptr<std::ostream>(new std::ofstream(filename))) {}
    DotFile(std::unique_ptr<std::ostream> s)
      : node_id(0), out(std::move(s)), is_closed(false)
    {
      *out << "digraph taskgraph {" << std::endl;
    }
    ~DotFile() {
      if (this->out && !this->is_closed) {
        this->close();
      }
    }

    void set_filename(std::string filename) {
      assert (!this->out);
      this->out = std::unique_ptr<std::ostream>(new std::ofstream(filename));
      *out << "digraph taskgraph {";
    }
    bool reserve_node(T const &t) {
      assert (!this->is_closed);
      if (this->node_ids.find(t) == this->node_ids.end()) {
        this->node_ids[t] = this->node_id++;
        return true;
      }
      return false;
    }
    void add_node(T const &t, std::map<std::string, std::string> const &params) {
      assert (!this->is_closed);
      this->reserve_node(t);
      if (this->described_nodes.find(t) == this->described_nodes.end()) {
        *out << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
        for (auto it = params.begin(); it != params.end(); ++it)  {
          *out << it->first << "=" << it->second;
          if (std::next(it) != params.end()) {
            *out << ",";
          }
        }
        *out << "];" << std::endl;
        this->described_nodes.insert(t);
      }
    }
    void add_edge(T const &src, T const &dst) {
      assert (!this->is_closed);
      this->reserve_node(src);
      this->reserve_node(dst);
      auto src_name = this->get_node_name(this->node_ids.at(src));
      auto dst_name = this->get_node_name(this->node_ids.at(dst));
      *out << "  " << src_name << " -> " << dst_name << ";" << std::endl;
    }
    void close() {
      *out << "}";
      out->flush();
      this->is_closed = true;
    }
  };

  class SimTaskCompare {
  public:
    bool operator() (SimTask* lhs, SimTask* rhs) {
      return lhs->ready_time > rhs->ready_time;
    }
  };

  class TaskManager {
  public:
    TaskManager(size_t max_num_tasks);
    void reset();
    SimTask* new_barrier_task();
    SimTask* new_update_task();
    SimTask* new_comm_task();
    SimTask* new_forward_task(Op* op, int idx);
    SimTask* new_backward_task(Op* op, int idx);
    SimTask* get_forward_task(Op* op, int idx);
    SimTask* get_backward_task(Op* op, int idx);
  private:
    SimTask* new_task();
  public:
    size_t global_task_id, max_num_tasks;
    SimTask** tasks;
    std::map<size_t, SimTask*> hash_to_forward_task, hash_to_backward_task;
  };

  class Simulator {
  public:
    Simulator(FFConfig const &config,
              FFHandler handler);
    void free_all();
    void* allocate(size_t num_elements, DataType type);
    Device* get_compute_device_by_id(int device_id);
    Device* get_inter_gpu_comm_device_by_ids(int src_id, int dst_id);
    Device* get_inter_node_comm_device_by_ids(int src_id, int dst_id);
    Device* get_gpu_to_dram_comm_device_by_id(int gpu_id);
    Device* get_dram_to_gpu_comm_device_by_id(int gpu_id);
    void add_task_dependencies_with_xfer(
        SimTask* src_task, SimTask* dst_task, size_t intersect);
    float measure_op_forward_time(Op* op, const ParallelConfig& config);
    float measure_op_backward_time(Op* op, const ParallelConfig& config);
    float simulate_runtime(const FFModel* model,
        const std::map<Op*, ParallelConfig>& global);
    float simulate_runtime(const FFModel* model,
        const std::map<Op*, ParallelConfig>& global,
        std::string const &export_file_name);
  public:
    FFHandler handler;
    char* base_ptr;
    size_t capacity;
    off_t offset;
    int warmup_times, repeat_times;
    int num_nodes, gpus_per_node, total_num_gpus;
    int cache_hits, cache_misses;
    SimulationVerbosity verbosity;
    TaskManager* task_manager;
    cudaEvent_t start_event, end_event;
    std::map<int, Device*> id_to_compute_device;
    std::map<int, Device*> id_to_gputodram_comm_device;
    std::map<int, Device*> id_to_dramtogpu_comm_device;
    std::map<size_t, Device*> ids_to_inter_gpu_comm_device;
    std::map<size_t, Device*> ids_to_inter_node_comm_device;
    std::map<size_t, float> hash_to_op_forward_time;
    std::map<size_t, float> hash_to_op_backward_time;
    std::map<size_t, float> seen_graphs;
  public:
    Conv2DMeta* conv2d_meta;
    LinearMeta* linear_meta;
    Pool2DMeta* pool2d_meta;
    ElementUnaryMeta* ele_unary_meta;
    ElementBinaryMeta* ele_binary_meta;
    ConcatMeta *concat_meta;
  };
}

#endif
