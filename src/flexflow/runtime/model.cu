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
//#include "realm/runtime_impl.h"
//#include "realm/cuda/cuda_module.h"

using namespace flexflow;

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

