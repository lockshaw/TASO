#include "taso/graph_ops.h"
#include "flexflow/ffconst.h"
#include "taso/ops.h"
#include <cassert>

using namespace taso;

bool SplitInfo::operator==(const SplitInfo& rhs) const {
  if (num != rhs.num) return false;
  for (int i = 0; i < num; i++)
    if (pos[i] != rhs.pos[i])
      return false;
  return true;
}

void SplitInfo::merge(int offset, const SplitInfo& next) {
  if (num + 1 + next.num >= MAX_NUM_SPLITS) {
    printf("num = %d, next.num = %d\n", num, next.num);
  }
  assert(num + 1 + next.num < MAX_NUM_SPLITS);
  for (int i = 0; i < next.num; i++)
    pos[num++] = offset + next.pos[i];
  pos[num++] = offset;
}

bool SplitInfo::operator!=(const SplitInfo& rhs) const
{
  if (num != rhs.num) return true;
  for (int i = 0; i < num; i++)
    if (pos[i] != rhs.pos[i]) return true;
  return false;
}

SplitInfo& SplitInfo::operator=(const SplitInfo& st)
{
  num = st.num;
  for (int i = 0; i < num; i++)
    pos[i] = st.pos[i];
  return *this;
}

void SplitInfo::divide(SplitInfo& left, SplitInfo& right, int &mid) {
  assert(num > 0);
  left.num = 0;
  right.num = 0;
  mid = pos[num - 1];
  int idx = 0;
  while (idx < num && pos[idx] < mid)
    left.pos[left.num++] = pos[idx++];
  while (idx < num - 1)
    right.pos[right.num++] = pos[idx++] - mid;
}

void SplitInfo::combine(const SplitInfo& next) {
  if (num != next.num)
    num = 0;
  for (int i = 0; i < num; i++)
    if (pos[i] != next.pos[i]) {
      num = 0;
      return;
    }
}

void SplitInfo::serialize(int* keys, int& idx) const {
  keys[idx++] = num;
  for (int i = 0; i < num; i++)
    keys[idx++] = pos[i];
}


const Op Op::INVALID_OP = Op();

Op::Op(void)
{
  guid = GUID_INVALID;
  ptr = NULL;
}

Op::Op(size_t _guid, OpBase* _ptr)
  : guid(_guid), ptr(_ptr)
{ }

bool Op::operator==(const Op& b) const {
  if (guid != b.guid) return false;
  if (ptr != b.ptr) return false;
  return true;
}

bool Op::operator!=(const Op& b) const {
  if (guid != b.guid) return true;
  if (ptr != b.ptr) return true;
  return false;
}

bool Op::operator<(const Op& b) const {
  if (guid != b.guid) return guid < b.guid;
  if (ptr != b.ptr) return ptr < b.ptr;
  return true;
}

Op& Op::operator=(const Op& op)
{
  guid = op.guid;
  ptr = op.ptr;
  return *this;
}

std::string Op::op_to_string(const OpBase* ptr)
{
  switch (ptr->type) {
    case OP_INPUT:
      return "Input";
    case OP_WEIGHT:
      return "Weight";
    case OP_ANY:
      return "Any";
    case OP_CONV2D:
      return "Conv";
    case OP_DROPOUT:
      return "Dropout";
    case OP_LINEAR:
      return "Linear";
    case OP_POOL2D_MAX:
      return "MaxPool";
    case OP_POOL2D_AVG:
      return "AveragePool";
    case OP_RELU:
      return "Relu";
    case OP_SIGMOID:
      return "Sigmoid";
    case OP_TANH:
      return "TanH";
    case OP_BATCHNORM:
      return "Batchnorm";
    case OP_CONCAT:
      return "Concat";
    case OP_SPLIT:
      return "Split";
    case OP_RESHAPE:
      return "Reshape";
    case OP_TRANSPOSE:
      return "Transpose";
    case OP_EW_ADD:
      return "Add";
    case OP_EW_MUL:
      return "Mul";
    case OP_MATMUL:
      return "MatMul";
    case OP_MUL:
      return "Mul";
    case OP_ENLARGE:
      return "Enlarge";
    case OP_SQUEEZE:
      return "Squeeze";
    case OP_UNSQUEEZE:
      return "Unsqueeze";
    case OP_EW_SUB:
      return "Sub";
    case OP_EW_DIV:
      return "Div";
    case OP_EW_EQUAL:
      return "Equal";
    case OP_EW_GREATER:
      return "Greater";
    case OP_EW_LESS:
      return "Less";
    case OP_EW_MAX:
      return "Max";
    case OP_EW_MIN:
      return "Min";
    case OP_REDUCE_ARGMAX:
      return "ArgMax";
    case OP_REDUCE_ARGMIN:
      return "ArgMin";
    case OP_REDUCE_MAX:
      return "ReduceMax";
    case OP_REDUCE_MEAN:
      return "ReduceMean";
    case OP_REDUCE_MIN:
      return "ReduceMin";
    case OP_REDUCE_PROD:
      return "ReduceProd";
    case OP_REDUCE_SUM:
      return "ReduceSum";
    case OP_PAD:
      return "Pad";
    case OP_SHAPE:
      return "Shape";
    case OP_SIZE:
      return "Size";
    case OP_TOPK:
      return "TopK";
    case OP_WHERE:
      return "Where";
    case OP_CEIL:
      return "Ceil";
    case OP_CAST:
      return "Cast";
    case OP_EXP:
      return "Exp";
    case OP_ROUND:
      return "Round";
    case OP_LOG:
      return "Log";
    case OP_LOGICAL_NOT:
      return "Not";
    case OP_SQRT:
      return "Sqrt";
    case OP_LEAKYRELU:
      return "LeakyRelu";
    case OP_SLICE:
      return "Slice";
    case OP_RESIZE:
      return "Resize";
    default:
      return "Unknown_" + std::to_string(ptr->type);
  }
}

std::string Op::to_string(void)
{
  if (ptr != NULL) {
    return op_to_string(ptr) + "_" + std::to_string(guid);
  }
  else {
    return "UnmappedOp_" + std::to_string(guid);
  }
}

bool OpCompare::operator()(const Op& a, const Op& b) const {
  if (a.guid != b.guid) return a.guid < b.guid;
  return a.ptr < b.ptr;
};

Edge::Edge(void)
: srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP), srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

bool EdgeCompare::operator()(const Edge& a, const Edge& b) const {
  if (!(a.srcOp == b.srcOp)) return a.srcOp < b.srcOp;
  if (!(a.dstOp == b.dstOp)) return a.dstOp < b.dstOp;
  if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
  if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
  return false;
};

SrcEdge::SrcEdge(int _idx, Op _op)
: idx(_idx), op(_op)
{}

bool SrcEdgeCompare::operator()(const SrcEdge& a, const SrcEdge& b) const {
  if (!(a.op == b.op)) return a.op < b.op;
  if (a.idx != b.idx) return a.idx < b.idx;
  return false;
};
