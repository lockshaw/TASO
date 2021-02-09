#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <map>
#include <set>
#include <fstream>
#include "taso/graph_ops.h"
#include "taso/tensor.h"

using namespace std;

class Model;

class Graph {
public:
  Graph();
  TensorHandle new_input(int dim, const int* dims);
  TensorHandle new_weight(int dim, const int* dims);
  TensorHandle new_weight(const Tensor& input);
  void add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  void remove_edge(Edge e);
  bool has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  void replace_node(Op oldOp, Op newOp);
  void remove_node(Op oldOp);
  void export_to_file(std::string file_name);
  // This conv2ds will create a weight tensor
  TensorHandle group_conv2d(int groups,
                            const TensorHandle _input,
                            int _outputC,
                            int _kernelH, int _kernelW,
                            int _strideH, int strideW,
                            PaddingMode _padding,
                            ActiMode _activation = AC_MODE_NONE);
  TensorHandle batchnorm(const TensorHandle _input,
                         const TensorHandle _scale,
                         const TensorHandle _bias,
                         const TensorHandle _mean,
                         const TensorHandle _var,
                         float _epsilon);
  TensorHandle cast(const TensorHandle _input, DataType _datatype);
  TensorHandle ceil(const TensorHandle _input);
  TensorHandle concat(int axis, int n, const TensorHandle* _inputs);
  TensorHandle constant(int ndim, int* dims, OpType _type);
  TensorHandle conv2d(const TensorHandle _input,
                      int _outputC,
                      int _kernelH, int _kernelW,
                      int _strideH, int _strideW,
                      PaddingMode _padding,
                      ActiMode _activation = AC_MODE_NONE);
  TensorHandle conv2d(const TensorHandle _input,
                      const TensorHandle _weight,
                      int _strideH, int _strideW,
                      PaddingMode _padding,
                      ActiMode _activation = AC_MODE_NONE);
  TensorHandle dropout(const TensorHandle _input);
  TensorHandle element(OpType type,
                       const TensorHandle _t1,
                       const TensorHandle _t2);
  TensorHandle elementwise_unary(const TensorHandle _input, OpType _type);
  TensorHandle enlarge(const TensorHandle _w1, const TensorHandle _w2);
  TensorHandle exp(const TensorHandle _input);
  TensorHandle fc(const TensorHandle _input,
                  int _outputC,
                  ActiMode _actiMode = AC_MODE_NONE);
  TensorHandle fuse_conv_batchnorm(const TensorHandle _conv_w,
                                   const TensorHandle _scale,
                                   const TensorHandle _bias,
                                   const TensorHandle _mean,
                                   const TensorHandle _var);
  // TensorHandle fuse_conv_batchnorm_alpha_var(const TensorHandle _conv_w,
  //                                  const TensorHandle _scale,
  //                                  const TensorHandle _var);
  TensorHandle fuse_conv_batchnorm_bias(const TensorHandle _scale,
                                   const TensorHandle _bias,
                                   const TensorHandle _mean,
                                   const TensorHandle _var);
  TensorHandle broadcast_add(const TensorHandle _data,
                                   const TensorHandle _bias);

  TensorHandle leakyrelu(const TensorHandle _input, float _alpha,
                         bool _inplace=true);
  TensorHandle log(const TensorHandle _input);
  TensorHandle logical_not(const TensorHandle _input);
  TensorHandle matmul(const TensorHandle _input,
                      const TensorHandle _weight,
                      ActiMode _actiMode = AC_MODE_NONE);
  TensorHandle merge_gconv(const TensorHandle _weight, int count);
  TensorHandle mul(const TensorHandle _x,
                   const TensorHandle _y);
  TensorHandle pad(const TensorHandle _input,
                   const std::vector<int>& _pad_before,
                   const std::vector<int>& _pad_after,
                   float _pad_value);
  TensorHandle pool2d_max(const TensorHandle _input,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation = AC_MODE_NONE);
  TensorHandle pool2d_avg(const TensorHandle _input,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation = AC_MODE_NONE);
  TensorHandle reduce(const TensorHandle _input,
                      OpType _type,
                      const std::vector<int>& axes,
                      bool keepdims);
  TensorHandle reduce_argmax(const TensorHandle _input,
                             const std::vector<int>& axes,
                             bool keepdims);
  TensorHandle reduce_argmin(const TensorHandle _input,
                             const std::vector<int>& axes,
                             bool keepdims);
  TensorHandle reduce_max(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle reduce_mean(const TensorHandle _input,
                           const std::vector<int>& axes,
                           bool keepdims);
  TensorHandle reduce_min(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle reduce_prod(const TensorHandle _input,
                           const std::vector<int>& axes,
                           bool keepdims);
  TensorHandle reduce_sum(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle relu(const TensorHandle _input,
                    bool _inPlace = true);
  TensorHandle reshape(const TensorHandle _input,
                       const std::vector<int>& _shape);
  TensorHandle resize(const TensorHandle _input,
                      const std::vector<int>& _shape);
  TensorHandle round(const TensorHandle _input);
  TensorHandle shape(const TensorHandle _input,
                     OpType _type);
  TensorHandle slice(const TensorHandle _input,
                     const std::vector<int>& _start,
                     const std::vector<int>& _end,
                     const std::vector<int>& _axes,
                     const std::vector<int>& _steps);
  TensorHandle sigmoid(const TensorHandle _input,
                       bool _inPlace = true);
  //void split(Tensor _input, int axis, int c1, int c2, Tensor* outputs);
  //void split(Tensor _input, int axis, int num, const int* sizes, Tensor* outputs);
  void split(const TensorHandle _input, int _axis,
             const std::vector<int>& _sizes,
             TensorHandle* _outputs);
  void split_equal(const TensorHandle _input, int _axis,
                   int _num, TensorHandle* _outputs);
  TensorHandle sqrt(const TensorHandle _input);
  TensorHandle squeeze(const TensorHandle input, const std::vector<int>& axes);
  TensorHandle transpose(const TensorHandle _input,
                         const std::vector<int>& _perm,
                         bool _shuffle = false);
  TensorHandle tanh(const TensorHandle _input,
                    bool _inPlace = true);
  void topk(const TensorHandle _input,
            int _axis, int _numk,
            bool _largest, bool _sorted,
            Tensor* outputs);
  TensorHandle unsqueeze(const TensorHandle input, const std::vector<int>& axes);
  TensorHandle where(const TensorHandle _cond, const TensorHandle _x, const TensorHandle _y);
  //void split(Tensor _input, int axis, int num, Tensor* outputs);

  // Helper Functions for Cython
  Op find_op_or_fail(size_t guid);
  Graph* optimize(float alpha, int budget, bool print_subst);
  std::vector<Graph *> optimizeMulti(float alpha, int budget, bool print_subst, int numResults);
  Graph* preprocess_weights(void);
  int get_operator_list(Op* opList, size_t maxNumOps);
  int get_input_edges(Edge* opList, size_t guid);
  OpType get_operator_type(size_t guid);
  int get_operator_int_attr(size_t guid, PMParameter attr);
  float get_operator_float_attr(size_t guid, PMParameter attr);
  int get_num_outputs(size_t guid);
  int get_input_dims(size_t guid, int* dims, int idx);
  int get_output_dims(size_t guid, int* dims, int idx);
  int get_split_lens(size_t guid, int* lens);
  size_t num_in_edges(Op op);
  size_t num_out_edges(Op op);
  size_t hash(void);
  void print(void);
  bool check_correctness(void);
  bool has_loop(void);
  float total_cost(void);
  /* float run(); */
  void print_costs(void);
  void print_measurements(void);
#ifdef TRT
  void buildTRTNetwork(INetworkDefinition *network);
private:
  void buildTRTNetworkHelper(INetworkDefinition *network, std::map<SrcEdge, ITensor *, SrcEdgeCompare>& outputs, Edge edge);
#endif
  void export_op(ofstream &file_stream, Op &op);
private:
  TensorHandle input_wrapper(const TensorHandle _input);
  TensorHandle weight_wrapper(const TensorHandle _weight);
public:
  Model *model;
  float totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  struct GraphSubst {
    std::vector<Op> srcOps, dstOps;
  };
  std::vector<GraphSubst> subst_history;
};

#endif // _GRAPH_H
