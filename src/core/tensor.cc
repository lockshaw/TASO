#include "taso/tensor.h"
#include <cassert>
#include "flexflow/config.h"

Tensor::Tensor(void)
: numDim(0), owner_idx(0), owner_op(NULL) {
  adim = dim;
  for (int i = 0; i < MAX_DIM; i++)
    split[i].num = 0;
}

Tensor::Tensor(int ndim, const int* dims)
: numDim(ndim), owner_idx(0), owner_op(NULL) {
  assert(ndim <= MAX_DIM);
  int count = 1;
  for (int i = ndim-1; i >= 0; i--) {
    dim[i] = dims[i];
    stride[i] = count;
    count *= dim[i];
    split[i]  = SplitInfo::NO_SPLIT;
  }
}

Tensor& Tensor::operator=(const Tensor& src) {
  numDim = src.numDim;
  for (int i = 0; i < numDim; i++) {
    dim[i] = src.dim[i];
    stride[i] = src.stride[i];
    split[i] = src.split[i];
  }
  owner_idx = src.owner_idx;
  owner_op = src.owner_op;
  return *this;
}

int Tensor::volume(void) const {
  int ret = 1;
  for (int i = 0; i < numDim; i++)
    ret *= dim[i];
  return ret;
}

size_t Tensor::get_volume() const {
  size_t volume = 1;
  for (int i = 0; i < numDim; i++)
    volume *= adim[i];
  return volume;
}


std::string Tensor::to_string(std::string name) const {
  name = name + "(";
  for (int i = 0; i < numDim; i++) {
    std::string suffix = (i == numDim -1) ? ")" : " ";
    name = name + std::to_string(dim[i]) + ":"
         + std::to_string(stride[i]) + suffix;
  }
  return name;
}

void Tensor::serialize(int* keys, int& idx) const {
  keys[idx++] = MAGIC_NUMBER;
  keys[idx++] = numDim;
  for (int i = 0; i < numDim; i++)
    keys[idx++] = dim[i];
  for (int i = 0; i < numDim; i++)
    keys[idx++] = stride[i];
  for (int i = 0; i < numDim; i++)
    split[i].serialize(keys, idx);
}

bool Tensor::has_same_shape_stride_split(const Tensor& tensor) const
{
  if (numDim != tensor.numDim)
    return false;
  for (int i = 0; i < numDim; i++) {
    if (dim[i] != tensor.dim[i])
      return false;
    if (stride[i] != tensor.stride[i])
      return false;
    if (split[i] != tensor.split[i])
      return false;
  }
  return true;
}

bool Tensor::default_layout(void) const
{
  int cnt = 1;
  for (int i = numDim-1; i >= 0; i--) {
    if (stride[i] != cnt) return false;
    cnt *= dim[i];
  }
  return true;
}

