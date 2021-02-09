#ifndef _TENSOR_H
#define _TENSOR_H

#include "flexflow/ffconst.h"
//#include "taso/graph_ops.h"
#include "flexflow/config.h"
#include "flexflow/legion_mock.h"

#define MAX_DIM 8
#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define MAX_NUM_SPLITS 32

class ParallelConfig;
class OpBase;

struct SplitInfo {
  SplitInfo(void) {num = 0;}

  bool operator==(const SplitInfo& rhs) const;
  void merge(int offset, const SplitInfo& next);
  bool operator!=(const SplitInfo& rhs) const;
  SplitInfo& operator=(const SplitInfo& st);
  void divide(SplitInfo& left, SplitInfo& right, int &mid);
  void combine(const SplitInfo& next);
  void serialize(int* keys, int& idx) const;

  static const SplitInfo NO_SPLIT;
  int num;
  int pos[MAX_NUM_SPLITS];
};

struct Tensor {
  static const int MAX_KEY_LENGTH = (MAX_NUM_SPLITS + 2) * MAX_DIM + 2;
  static const int MAGIC_NUMBER = 23333;

  Tensor(void);
  Tensor(int ndim, int const *dims);

  Tensor& operator=(const Tensor& src);

  int volume(void) const;
  size_t get_volume() const;

  std::string to_string(std::string name) const;

  void serialize(int* keys, int& idx) const;

  bool has_same_shape_stride_split(const Tensor& tensor) const;

  bool default_layout(void) const;

  bool get_input_sub_tensor(ParallelConfig const &,
                            Tensor &,
                            OperatorType);
  bool get_output_sub_tensor(ParallelConfig const &,
                             Tensor &,
                             OperatorType);
  Domain get_domain() const;

  int numDim, dim[MAX_DIM], stride[MAX_DIM];
  int owner_idx; // idx is used for Ops with multiple outputs (e.g., split)
  OpBase *owner_op;

  // Meta data for splits
  SplitInfo split[MAX_DIM];

  // Flexflow information
  int numInputs, numWeights, numOutputs;
  int *adim;
};

using TensorHandle = Tensor*;

#endif // _TENSOR_H
