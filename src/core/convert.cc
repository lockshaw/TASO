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
}

void Converter::convert() {
  auto in = std::unique_ptr<std::ostream>(nullptr);
  auto out = std::unique_ptr<std::ostream>(nullptr);
  this->convert(std::move(in), std::move(out));
}

void Converter::convert(std::unique_ptr<std::ostream> ossIn, std::unique_ptr<std::ostream> ossOut) {
  if (ossIn != NULL) {
    graph.to_filtered_dot(std::move(ossIn));
  }
  std::list<taso::Op> to_convert;
  for (auto const &kv : graph.outEdges) {
    /* if (kv.first.ptr != NULL && kv.first.ptr->type == taso::OP_INPUT) { */
      for (taso::Edge const &edge : kv.second) {
        if (edge.dstOp.ptr->type != taso::OP_INPUT && edge.dstOp.ptr->type != taso::OP_WEIGHT) {
          assert(edge.dstOp.ptr->type != taso::OP_ENLARGE);
          to_convert.push_back(edge.dstOp);
        }
      }
    /* } */
  }
  taso::Op current;
  bool ready;
  while (!to_convert.empty()) {
    printf("The queue has %d items\n", to_convert.size());
    current = to_convert.front();
    to_convert.pop_front();
    auto find_result = graph.inEdges.find(current);
    assert (find_result != graph.inEdges.end());
    ready = true;
    printf("Are we ready?\n");
    printf("We have %d in edges to examine\n", find_result->second.size());
    for (taso::Edge const &edge : find_result->second) {
      printf("Examining edge with srcOp id %d and srcOp ptr %p and srcOp type %d\n", edge.srcOp.guid, edge.srcOp.ptr, edge.srcOp.ptr ? edge.srcOp.ptr->type : -1);
      if (edge.srcOp.guid != taso::GUID_WEIGHT && edge.srcOp.guid != taso::GUID_INPUT && edge.srcOp.ptr->type != taso::OP_INPUT && edge.srcOp.ptr->type != taso::OP_WEIGHT && this->opMap.find(edge.srcOp) == this->opMap.end()) {
        ready = false;
        printf("No we are not!\n");
        break;
      }
    }
    if (ready) {
      printf("Yes we are!\n");
    }
    if (!ready) {
      continue;
    } else {
      printf("Converting node with id %d and ptr %p\n", current.guid, current.ptr);
      bool didConvert = this->convertOp(current);
      assert (this->opMap.find(current) != this->opMap.end());
      if (didConvert) {
        printf("Adding the destinations of the %d outEdges\n", graph.outEdges[current].size());
        for (taso::Edge const &outEdge : graph.outEdges[current]) {
          to_convert.push_back(outEdge.dstOp);
        }
      }
    }
  }
  if (ossOut != NULL) {
    model.to_dot(std::move(ossOut));
  }
}

bool Converter::convertOp(taso::Op const &op) {
  auto result = this->opMap.find(op);
  if (result == this->opMap.end()) {
    this->rawConvertOp(op);
    this->opMap[op] = model.layers.back().get();
    printf("Inserting converted op with id %d and ptr %p\n", op.guid, op.ptr);
    printf("The op map now has %d items\n", this->opMap.size());
    return true;
  }
  return false;
}

void Converter::rawConvertOp(taso::Op const &op) {
  flexflow::Tensor inputs[MAX_NUM_INPUTS];
  taso::Tensor *t_inputs = op.ptr->inputs;
  taso::Tensor *t_outputs = op.ptr->outputs;
  for (taso::Edge const &inE : this->graph.inEdges[op]) {
    assert (inE.dstOp == op);
    if (inE.srcOp.guid != taso::GUID_WEIGHT && inE.srcOp.guid != taso::GUID_INPUT && inE.srcOp.ptr->type != taso::OP_INPUT && inE.srcOp.ptr->type != taso::OP_WEIGHT) {
      inputs[inE.dstIdx] = this->opMap.at(inE.srcOp)->outputs[inE.srcIdx];
    } else {
      inputs[inE.dstIdx] = this->convertTensor(t_inputs[inE.dstIdx]);
    }
  }
  switch (op.ptr->type) {
    case taso::OP_CONV2D:
    {
      taso::Conv2D *p = (taso::Conv2D *)op.ptr;
      int padH, padW, groups;
      p->get_padding(&padH, &padW);
      p->get_int_parameter(taso::PM_GROUP, &groups);
      printf("Creating convolution layer with input channels %d and groups %d\n", t_inputs[0].dim[1], groups);
      model.conv2d(
        inputs[0],
        t_outputs[0].dim[1],
        t_inputs[1].dim[2], t_inputs[1].dim[3],
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
      model.add(
        inputs[0],
        inputs[1]
      );
      return;
    }
    case taso::OP_POOL2D_MAX:
    case taso::OP_POOL2D_AVG:
    {
      taso::Pool2D *p = (taso::Pool2D *)op.ptr;
      int padH, padW;
      p->get_padding(&padH, &padW);
      model.pool2d(
        inputs[0],
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
        tensors.push_back(inputs[i]);
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
        inputs[0]
      );
      return;
    }
    case taso::OP_SPLIT:
    {
      taso::Split *p = (taso::Split *)op.ptr;
      flexflow::Tensor outputs[MAX_NUM_INPUTS];
      model.split(
        inputs[0],
        outputs,
        p->sizes,
        p->axis
      );
      return;
    }
    default:
      assert(false && "Unknown op type in conversion");
  }
}

flexflow::Tensor Converter::convertTensor(taso::Tensor const &in) const {
  flexflow::Tensor out;
  out.numDim = in.numDim;
  for (int i = 0; i < in.numDim; i++) {
    out.adim[in.numDim - i - 1] = in.dim[i];
  }
  out.data_type = flexflow::DT_FLOAT;
  /* if (in.op.ptr->type != taso::OP_INPUT) { */
  /*   auto owner_iter = this->opMap.find(in.op); */
  /*   assert (owner_iter != this->opMap.end()); */
  /*   out.owner_op = owner_iter->second; */
  /*   out.owner_idx = in.idx; */
  /* } else { */
  out.owner_op = NULL;
  out.owner_idx = 0;
  /* } */
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

flexflow::FFModel &Converter::get_converted() {
  return this->model;
}
