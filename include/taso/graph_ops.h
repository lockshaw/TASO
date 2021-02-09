#ifndef _GRAPH_OPS_H
#define _GRAPH_OPS_H

#include <string>
#include <cstddef>

#define MAX_NUM_SPLITS 32

  class OpBase;

  struct Op {
    Op(void);
    Op(size_t _guid, OpBase *_ptr);


    bool operator==(const Op& b) const;
    bool operator!=(const Op& b) const;
    bool operator<(const Op& b) const;
    Op& operator=(const Op& op);
    std::string op_to_string(const OpBase* ptr);
    std::string to_string(void);
    static const Op INVALID_OP;
    size_t guid;
    OpBase* ptr;
  };

  struct OpCompare {
    bool operator()(const Op& a, const Op& b) const;
  };

  struct Edge {
    Edge(void);
    Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx);
    Op srcOp, dstOp;
    int srcIdx, dstIdx;
  };

  struct EdgeCompare {
    bool operator()(const Edge& a, const Edge& b) const;
  };

  struct SrcEdge {
    SrcEdge(int _idx, Op _op);
    int idx;
    Op op;
  };

  struct SrcEdgeCompare {
    bool operator()(const SrcEdge& a, const SrcEdge& b) const;
  };

#endif // _GRAPH_OPS_H
