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
#include "model.h"
#include "cuda_helper.h"
//#include "realm/runtime_impl.h"
//#include "realm/cuda/cuda_module.h"

void Op::inner_measure_compute_time(Simulator *sim,
                                    std::function<void()> const &forward,
                                    std::function<void()> const &backward,
                                    float &forward_time,
                                    float &backward_time)
{
  // measure forward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    forward();
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  forward_time = milliseconds / sim->repeat_times;

  // measure backward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    backward();
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  backward_time = milliseconds / sim->repeat_times;
}


/* FFHandler UtilityTasks::init_cuda_task( */
/*               const Task *task, */
/*               const std::vector<PhysicalRegion> &regions, */
/*               Context ctx, HighLevelRuntime *runtime) */
/* { */
/*   assert(regions.size() == 0); */
/*   assert(task->local_arglen == sizeof(FFInitInfo)); */
/*   const FFInitInfo* info = (FFInitInfo*) task->local_args; */
/*   //assert(task->arglen == sizeof(size_t)); */
/*   //size_t workSpaceSize = *(const size_t*) task->args; */
/*   printf("workSpaceSize (%d MB)\n", info->workSpaceSize / 1024 / 1024); */
/*   FFHandler handle; */
/*   handle.workSpaceSize = info->workSpaceSize; */
/*   checkCUDA(cublasCreate(&handle.blas)); */
/*   checkCUDNN(cudnnCreate(&handle.dnn)); */
/* //#ifdef FF_ENABLE_NCCL */
/* //  checkNCCL(ncclCommInitRank(&handle.nccl, info->allRanks, info->ncclId, info->myRank)); */
/* //  fprintf(stderr, "handle.nccl(%p)\n", handle.nccl); */
/* //#endif */
/*   //std::set<Memory> memFB; */
/*   //assert(memFB.size() == 1); */
/*   //assert(memFB.begin()->kind() == Memory::GPU_FB_MEM); */
/*   //Realm::MemoryImpl* memImpl = */
/*   //    Realm::get_runtime()->get_memory_impl(*memFB.begin()); */
/*   //Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl; */
/*   //off_t offset = memFBImpl->alloc_bytes(workSpaceSize); */
/*   //handle.workSpace = memFBImpl->get_direct_ptr(offset, 0); */
/*   checkCUDA(cudaMalloc(&handle.workSpace, handle.workSpaceSize)); */
/*   return handle; */
/* } */

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}

void nearest_neighbor(unsigned char* image,
                      unsigned char* buffer,
                      int height, int width,
                      int orig_height, int orig_width,
                      float height_scale, float width_scale)
{
  // Note buffer is in HWC layout while image is in CHW layout
  for (int y = 0; y < height; y++) {
    int y0 = std::min(static_cast<int>(roundf(y * height_scale)), orig_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 = std::min(static_cast<int>(roundf(x * width_scale)), orig_width - 1);
      for (int c = 0; c < 3; c++) {
        int origOffset = calc_offset(y0, x0, c, orig_width, 3);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[origOffset];
      }
    }
  }
}

__global__
void apply_normalize(float *tensor_ptr, const unsigned char *rgb_ptr,
                     size_t size, size_t hxw)
{
  const float mean[3] = {0.485, 0.456, 0.406};
  const float var[3] = {0.229, 0.224, 0.225};

  CUDA_KERNEL_LOOP(i, size)
  {
    // decide the color of the current position by assuming NCHW layout
    int c = (i / hxw) % 3;
    tensor_ptr[i] = (static_cast<float>(rgb_ptr[i]) / 256 - mean[c]) / var[c];
  }
}


__global__
void init_image_kernel(float* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1.0f;
  }
}

__global__
void init_label_kernel(int* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1;
  }
}

void FFModel::prefetch()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->prefetch(*this);
}

//void FFModel::update()
//{
//  for (int i = layers.size() - 1; i >= 0; i--)
//    layers[i]->update(*this);
//}


template <typename T>
bool Parameter::set_weights(const FFModel* ff,
                            const std::vector<int>& dims,
                            const T* data)
{
  Context ctx = ff->config.lg_ctx;
  Runtime* runtime = ff->config.lg_hlr;
  //TODO: check data type matches
  //TODO: Currently we use a task launch, change to index launch for NCCL parameter
  size_t volume = 1, num_replicas = 0;
  if (type == Parameter::NCCL) {
    Domain domain = runtime->get_index_space_domain(ctx, owner_op->task_is);
    num_replicas = domain.get_volume();
  } else if (type == Parameter::PS) {
    num_replicas = 1;
  } else {
    assert(false);
  }
  // Check dimensions
  if (numDim != (int)dims.size())
    return false;
  for (int i = 0; i < numDim; i++) {
    if (adim[numDim-1-i] != dims[i])
      return false;
    volume = volume * dims[i];
  }
  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (numDim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorW<T, DIM> acc(pr, req, FID_DATA, ctx, runtime, true); \
      assert(acc.rect.volume() == volume * num_replicas); \
      T* ptr = acc.ptr; \
      for (size_t i = 0; i < num_replicas; i++) { \
        memcpy(ptr, data, volume * sizeof(T)); \
        ptr += volume; \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template <typename T>
bool Parameter::get_weights(const FFModel* ff,
                            T* data)
{
  Context ctx = ff->config.lg_ctx;
  Runtime* runtime = ff->config.lg_hlr;
  LogicalRegion weight_lr = LogicalRegion::NO_REGION;
  if (type == CommType::PS) {
    weight_lr = region;
  } else {
    assert(owner_op != NULL);
    Domain domain = runtime->get_index_space_domain(ctx, owner_op->task_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
      case DIM: \
      { \
        DomainPoint point = Point<DIM>::ZEROES(); \
        weight_lr = runtime->get_logical_subregion_by_color( \
            ctx, part, point); \
        break; \
      }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }
  }
  //TODO: check data type matches
  size_t volume = 1;
  for (int i = 0; i < numDim; i++) {
    volume = volume * adim[i];
  }
  RegionRequirement req(weight_lr, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (numDim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorR<T, DIM> acc(pr, req, FID_DATA, ctx, runtime); \
      assert(acc.rect.volume() == volume); \
      memcpy(data, acc.ptr, volume * sizeof(T)); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template bool Parameter::set_weights<float>(const FFModel* ff, const std::vector<int>& dims, const float* data);
template bool Parameter::get_weights<float>(const FFModel* ff, float* data);
