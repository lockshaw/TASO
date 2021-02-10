#include "flexflow/model.h"
#include "taso/ops.h"
#include "flexflow/convert.h"
#include <ostream>

namespace taso {
  flexflow::FFModel Graph::to_ff() const {
    flexflow::FFConfig ffconfig;
    flexflow::FFModel ff(ffconfig);

    return ffconfig;
  }
}

Converter::Converter(flexflow::FFConfig ffconfig, taso::Graph const &graph)
  : model(ffconfig), graph(graph)
{
  this->convert(NULL);
}

Converter::Converter(flexflow::FFConfig ffconfig, taso::Graph const &graph, std::unique_ptr<std::ostream> oss)
  : model(ffconfig), graph(graph)
{
  this->convert(std::move(oss));
}

flexflow::FFModel Converter::convert(std::unique_ptr<std::ostream> oss) {
  std::list<taso::Op> to_convert;
  for (auto const &kv : graph.inEdges) {
    if (kv.second.empty()) {
      to_convert.push_back(kv.first);
    }
  }
  taso::Op current;
  bool ready;
  while (!to_convert.empty()) {
    current = to_convert.front();
    to_convert.pop_front();
    auto find_result = graph.inEdges.find(current);
    assert (find_result != graph.inEdges.end());
    ready = true;
    for (taso::Edge const &edge : find_result->second) {
      if (this->opMap.find(edge.srcOp) == this->opMap.end()) {
        ready = false;
        break;
      }
    }
    if (!ready) {
      continue;
    } else {
      this->convertOp(std::move(current));
    }
  }
  if (oss != NULL) {
    model.to_dot(std::move(oss));
  }
}

void Converter::convertOp(taso::Op const &op) {
  auto result = opMap.find(op);
  if (result == opMap.end()) {
    this->rawConvertOp(op);
    this->opMap[op] = model.layers.back().get();
  }
}

void Converter::rawConvertOp(taso::Op const &op) {
  taso::Tensor *inputs = op.ptr->inputs;
  taso::Tensor *outputs = op.ptr->outputs;
  switch (op.ptr->type) {
    case taso::OP_CONV2D:
    {
      taso::Conv2D *p = (taso::Conv2D *)op.ptr;
      int padH, padW, groups;
      p->get_padding(&padH, &padW);
      p->get_int_parameter(taso::PM_GROUP, &groups);
      model.conv2d(
        this->convertTensor(inputs[0]),
        outputs[0].dim[1],
        inputs[1].dim[2], inputs[1].dim[3],
        p->strideH, p->strideW,
        padH, padW,
        groups,
        this->convertActiMode(p->activation),
        true
      );
      return;
    }
    case taso::OP_EW_ADD:
    {
      taso::Element *p = (taso::Element *)op.ptr;
      model.add(
        this->convertTensor(inputs[0]),
        this->convertTensor(inputs[1])
      );
      return;
    }
    case taso::OP_POOL2D_MAX:
    case taso::OP_POOL2D_AVG:
    {
      taso::Pool2D *p = (taso::Pool2D *)op.ptr;
      int padH, padW, actiMode;
      p->get_padding(&padH, &padW);
      model.pool2d(
        this->convertTensor(inputs[0]),
        p->kernelH, p->kernelW,
        p->strideH, p->strideW,
        padH, padW,
        (op.ptr->type == taso::OP_POOL2D_MAX) ? flexflow::POOL_MAX : flexflow::POOL_AVG,
        this->convertActiMode(p->activation)
      );
      return;
    }
    case taso::OP_CONCAT:
    {
      taso::Concat *p = (taso::Concat *)op.ptr;
      std::vector<flexflow::Tensor> tensors;
      for (int i = 0; i < p->n; i++) {
        tensors.push_back(this->convertTensor(inputs[i]));
      }
      model.concat(
          p->n,
          tensors.data(),
          p->axis
      );
      return;
    }
    case taso::OP_RELU:
    {
      model.relu(
        this->convertTensor(inputs[0])
      );
      return;
    }
    case taso::OP_SPLIT:
    {
      taso::Split *p = (taso::Split *)op.ptr;
      flexflow::Tensor outputs[MAX_NUM_INPUTS];
      model.split(
        this->convertTensor(inputs[0]),
        outputs,
        p->sizes,
        p->axis
      );
      return;
    }
    /* case taso::OP_INPUT: */
    /* { */

    /* } */
    default:
      assert(false && "Unknown op type in conversion");
  }
}

flexflow::Tensor Converter::convertTensor(taso::Tensor const &in) const {
  flexflow::Tensor out;
  out.numDim = in.numDim;
  for (int i = 0; i < in.numDim; i++) {
    out.adim[i] = in.dim[i];
  }
  out.data_type = flexflow::DT_FLOAT;
  out.owner_idx = in.idx;
  auto owner_iter = this->opMap.find(in.op);
  assert (owner_iter != this->opMap.end());
  out.owner_op = owner_iter->second;
  return out;
}

flexflow::ActiMode Converter::convertActiMode(taso::ActiMode const &in) const {
  switch (in) {
    case taso::AC_MODE_NONE:
      return flexflow::AC_MODE_NONE;
    case taso::AC_MODE_SIGMOID:
      return flexflow::AC_MODE_SIGMOID;
    case taso::AC_MODE_RELU:
      return flexflow::AC_MODE_RELU;
    case taso::AC_MODE_TANH:
      return flexflow::AC_MODE_TANH;
    default:
      assert(false && "Unknown ActiMode in conversion");
  }
}
