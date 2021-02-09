#ifndef _FLEXFLOW_CONST_H_
#define _FLEXFLOW_CONST_H_

//That this must be consistent with python/taso/_cython/CCore.pxd
enum ActiMode {
  AC_MODE_NONE,
  AC_MODE_SIGMOID,
  AC_MODE_RELU,
  AC_MODE_TANH,
};

enum AggrMode {
  AGGR_MODE_NONE = 20,
  AGGR_MODE_SUM = 21,
  AGGR_MODE_AVG = 22,
};

enum PoolType {
  POOL_MAX = 30,
  POOL_AVG = 31,
};

enum DataType {
  DT_FLOAT = 111,
  DT_DOUBLE = 222,
  DT_HALF = 333,
  DT_INT8 = 444,
  DT_UINT8 = 555,
  DT_INT32 = 666,
  DT_INT64 = 777,
  DT_BOOL = 888,
};

enum LossType {
  LOSS_CATEGORICAL_CROSSENTROPY = 50,
  LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51,
  LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52,
  LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53,
};

enum MetricsType {
  METRICS_ACCURACY = 1001,
  METRICS_CATEGORICAL_CROSSENTROPY = 1002,
  METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004,
  METRICS_MEAN_SQUARED_ERROR = 1008,
  METRICS_ROOT_MEAN_SQUARED_ERROR = 1016,
  METRICS_MEAN_ABSOLUTE_ERROR = 1032,
};

//This must be consistent with python/taso/_cython/CCore.pxd
enum OpType {
  OP_INPUT,
  OP_WEIGHT,
  OP_ANY,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_BATCHMATMUL,
  OP_POOL2D,
  OP_POOL2D_MAX,
  OP_POOL2D_AVG,
  OP_RELU,
  OP_SIGMOID,
  OP_TANH,
  OP_ELU,
  OP_FLAT,
  OP_SOFTMAX,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_EMBEDDING,
  //OP_ELEMENTWISE,
  OP_RESHAPE,
  OP_REVERSE,
  OP_TRANSPOSE,
  OP_EW_ADD,
  OP_EW_MUL,
  OP_MATMUL,
  OP_MUL,
  OP_ENLARGE,
  OP_MERGE_GCONV,
  OP_CONSTANT_IMM,
  OP_CONSTANT_ICONV,
  OP_CONSTANT_ONE,
  OP_CONSTANT_POOL,
  OP_SQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  OP_UNSQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  OP_EW_SUB, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  OP_EW_DIV, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  OP_EW_EQUAL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  OP_EW_GREATER, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  OP_EW_LESS, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  OP_EW_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  OP_EW_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  OP_REDUCE_ARGMAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  OP_REDUCE_ARGMIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  OP_REDUCE_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  OP_REDUCE_MEAN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  OP_REDUCE_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  OP_REDUCE_PROD, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  OP_REDUCE_SUM, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  OP_PAD, //https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  OP_SHAPE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  OP_SIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  OP_TOPK, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  OP_WHERE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  OP_CEIL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  OP_CAST, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  OP_EXP, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  OP_ROUND, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  OP_LOG, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  OP_LOGICAL_NOT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  OP_SQRT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  OP_LEAKYRELU,
  OP_SLICE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  OP_RESIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  OP_PRELU, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
  OP_MULTIHEAD_ATTENTION,
  OP_FUSE_CONV_BATCHNORM,
  OP_FUSE_CONV_BATCHNORM_ALPHA_VAR,
  OP_FUSE_CONV_BATCHNORM_BIAS,
  OP_BROADCAST_ADD,
};
using OperatorType = OpType;

//That this must be consistent with python/taso/_cython/CCore.pxd
enum PaddingMode {
  PD_MODE_SAME,
  PD_MODE_VALID,
};

enum {
  GUID_INVALID = 0,
  GUID_INPUT = 10,
  GUID_WEIGHT = 11,
  GUID_PRESERVED = 19,
};


//This must be consistent with python/taso/_cython/CCore.pxd
enum PMParameter {
  PM_OP_TYPE,   	// AnyOp
  PM_NUM_INPUTS,	// AnyOp
  PM_NUM_OUTPUTS,	// AnyOp
  PM_GROUP,             // Conv2D
  PM_KERNEL_H,		// Conv2D, Pool2D
  PM_KERNEL_W,		// Conv2D, Pool2D
  PM_STRIDE_H,		// Conv2D, Pool2D
  PM_STRIDE_W,		// Conv2D, Pool2D
  PM_PAD,		// Conv2D, Pool2D
  PM_ACTI,		// Conv2D, Pool2D
  PM_NUMDIM,		// Concat, Transpose
  PM_AXIS,		// Concat, Split
  PM_PERM,		// Transpose
  PM_OUTSHUFFLE,	// Transpose
  PM_MERGE_GCONV_COUNT, // MergeGConv
  PM_AXES,		// Squeeze, Unsqueeze, Reduce*
  PM_KEEP_DIMS,         // Reduce*
  PM_EPSILON,   // BatchNorm
};

enum TNParameter {
  IN_0 = 100,
  IN_1 = 101,
  IN_2 = 102,
  IN_3 = 103,
  IN_4 = 104,
  IN_5 = 105,
  OU_0 = 200,
  OU_1 = 201,
  OU_2 = 202,
  OU_3 = 203,
  OU_4 = 204,
  OU_5 = 205,
};

enum DIMParameter {
  DIM_0 = 300,
  DIM_1 = 301,
  DIM_2 = 302,
  DIM_3 = 303,
  DIM_ND = 310,
};


#endif // _FLEXFLOW_CONST_H_
