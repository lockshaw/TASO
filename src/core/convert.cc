#include "flexflow/model.h"
#include "taso/ops.h"
#include "flexflow/convert.h"
#include "flexflow/hash_utils.h"
#include <ostream>

namespace taso {
  flexflow::FFModel Graph::to_ff() const {
    flexflow::FFConfig ffconfig;
    flexflow::FFModel ff(ffconfig);

    return ffconfig;
  }
}

void GraphVisitor::run(taso::Graph const &graph) {
  this->graph = &graph;
  std::queue<taso::Op> to_visit;

  for (auto const &node : graph.get_all_nodes()) {
    if (this->isStartingPoint(node)) {
      to_visit.push(node);
    }
  }

  taso::Op current;
  while (!to_visit.empty()) {
    current = to_visit.front();
    to_visit.pop();
    this->visit(current, to_visit);
  }
}

/* struct DeterministicOpComparator { */
/*   bool operator()(taso::Op const &a, taso::Op const &b) const { */
/*     return a.get_hash */
/*   } */
/* } */

/* void DeterministicGraphVisitor::run(taso::Graph const &graph) { */
/*   this->graph = &graph; */
/*   std::priority_queue<taso::Op */
/* } */

void TopoVisitor::visit(taso::Op const &op, std::queue<taso::Op> &toVisit) {
  if (this->visited.find(op) != this->visited.end()) {
    return;
  }
  {
    auto iter = this->graph->inEdges.find(op);
    if (iter != this->graph->inEdges.end()) {
      for (taso::Edge const &e : iter->second) {
        if (this->visited.find(e.srcOp) == this->visited.end()) {
          return;
        }
      }
    }
  }
  this->visitNode(op);
  this->visited.insert(op);
  {
    auto iter = this->graph->outEdges.find(op);
    if (iter != this->graph->outEdges.end()) {
      for (taso::Edge const &e : iter->second) {
        toVisit.push(e.dstOp);
      }
    }
  }
}

bool TopoVisitor::isStartingPoint(taso::Op const &op) {
  return true;
}

HashVisitor::HashVisitor()
  : hash(0)
{ }

void HashVisitor::visitNode(taso::Op const &op) {
  using flexflow::hash_combine;

  struct OpParamHash oph;
  hash_combine(this->hash, oph(op));
}

size_t HashVisitor::get_hash() const {
  return this->hash;
}

namespace std {
  size_t hash<taso::Graph>::operator()(taso::Graph const &g) const {
    HashVisitor hv;
    hv.run(g);
    return hv.get_hash();
  }
}

ConverterVisitor::ConverterVisitor(flexflow::FFConfig const &ffconfig)
  : model(ffconfig)
{ }

bool is_input(taso::Op const &op) {
  return op.guid == taso::GUID_INPUT || (op.ptr != nullptr && op.ptr->type == taso::OP_INPUT);
}

bool is_weight(taso::Op const &op) {
  return op.guid == taso::GUID_WEIGHT || (op.ptr != nullptr && op.ptr->type == taso::OP_WEIGHT);
}

bool ConverterVisitor::isStartingPoint(taso::Op const &op) {
  return is_input(op) || is_weight(op);
}

void ConverterVisitor::visitNode(taso::Op const &op) {
  if (is_weight(op)) {
    this->isDerivedFromWeight.insert(op);
    return;
  } else if (is_input(op)) {
    return;
  }

  flexflow::Tensor inputs[MAX_NUM_INPUTS];
  taso::Tensor *t_inputs = op.ptr->inputs;
  taso::Tensor *t_outputs = op.ptr->outputs;
  auto iter = this->graph->inEdges.find(op);
  bool derivedFromWeight = true;
  if (iter != this->graph->inEdges.end()) {
    for (taso::Edge const &inE : iter->second) {
      assert (inE.dstOp == op);

      if (this->isDerivedFromWeight.find(inE.srcOp) == this->isDerivedFromWeight.end()) {
        derivedFromWeight = false;
      }

      if (!is_input(inE.srcOp) && this->isDerivedFromWeight.find(inE.srcOp) == this->isDerivedFromWeight.end()) {
        inputs[inE.dstIdx] = this->opMap.at(inE.srcOp)->outputs[inE.srcIdx];
      } else {
        inputs[inE.dstIdx] = this->convertTensor(t_inputs[inE.dstIdx]);
      }
    }
  }

  if (derivedFromWeight) {
    this->isDerivedFromWeight.insert(op);
    return;
  }

  switch (op.ptr->type) {
    case taso::OP_CONV2D:
    {
      taso::Conv2D *p = (taso::Conv2D *)op.ptr;
      int padH, padW, groups;
      p->get_padding(&padH, &padW);
      p->get_int_parameter(taso::PM_GROUP, &groups);
      /* printf("Creating convolution layer with input channels %d and groups %d\n", t_inputs[0].dim[1], groups); */
      model.conv2d(
        inputs[0],
        t_outputs[0].dim[1],
        t_inputs[1].dim[2], t_inputs[1].dim[3],
        p->strideH, p->strideW,
        padH, padW,
        groups,
        convertActiMode(p->activation),
        true
      );
      break;
    }
    case taso::OP_EW_ADD:
    {
      model.add(
        inputs[0],
        inputs[1]
      );
      break;
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
        convertActiMode(p->activation)
      );
      break;
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
      break;
    }
    case taso::OP_RELU:
    {
      model.relu(
        inputs[0]
      );
      break;
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
      break;
    }
    default:
      throw std::runtime_error("Unknown op type " + taso::op_type_name(op.ptr->type) + " in conversion");
  }
  this->opMap[op] = this->model.layers.back().get();
}

size_t tensor_dim_hash(taso::Tensor const &t) {
  using flexflow::hash_combine;

  size_t hash = 394876;
  hash_combine(hash, t.numDim);
  for (int i = 0; i < t.numDim; i++) {
    hash_combine(hash, t.dim[i]);
  }
  return hash;
}

size_t OpParamHash::operator()(taso::Op const &op) const {
  using flexflow::hash_combine;

  if (is_input(op) | is_weight(op)) {
    return 0;
  }
  auto inputs = op.ptr->inputs;
  auto outputs = op.ptr->outputs;
  switch (op.ptr->type) {
    case taso::OP_CONV2D:
      {
        taso::Conv2D *p = (taso::Conv2D *)op.ptr;
        int padH, padW, groups;
        p->get_padding(&padH, &padW);
        p->get_int_parameter(taso::PM_GROUP, &groups);
        auto params = std::make_tuple(
            taso::OP_CONV2D,
            tensor_dim_hash(inputs[0]),
            outputs[0].dim[1],
            inputs[1].dim[2], inputs[1].dim[3],
            p->strideH, p->strideW,
            padH, padW,
            groups,
            convertActiMode(p->activation)
        );
        return std::hash<decltype(params)>()(params);
      }
    case taso::OP_EW_ADD:
      {
        auto params = std::make_tuple(
            taso::OP_EW_ADD,
            tensor_dim_hash(inputs[0]),
            tensor_dim_hash(inputs[1])
        );
        return std::hash<decltype(params)>()(params);
      }
    case taso::OP_POOL2D_MAX:
    case taso::OP_POOL2D_AVG:
      {
        taso::Pool2D *p = (taso::Pool2D *)op.ptr;
        int padH, padW;
        p->get_padding(&padH, &padW);
        auto params = std::make_tuple(
          op.ptr->type,
          tensor_dim_hash(inputs[0]),
          p->kernelH, p->kernelW,
          p->strideH, p->strideW,
          padH, padW,
          convertActiMode(p->activation)
        );
        return std::hash<decltype(params)>()(params);
      }
    case taso::OP_CONCAT:
      {
        taso::Concat *p = (taso::Concat *)op.ptr;
        size_t inputsHash = 193875;
        hash_combine(inputsHash, p->n);
        for (int i = 0; i < p->n; i++) {
          hash_combine(inputsHash, tensor_dim_hash(inputs[i]));
        }
        auto params = std::make_tuple(
            taso::OP_CONCAT,
            inputsHash,
            p->axis
        );
        return std::hash<decltype(params)>()(params);
      }
    case taso::OP_RELU:
      {
        auto params = std::make_tuple(
            taso::OP_RELU,
            tensor_dim_hash(inputs[0])
        );
        return std::hash<decltype(params)>()(params);
      }
    case taso::OP_SPLIT:
      {
        taso::Split *p = (taso::Split *)op.ptr;
        size_t outputsHash = 8937465;
        for (int i = 0; i < p->numOutputs; i++) {
          hash_combine(outputsHash, p->sizes[i]);
        }
        auto params = std::make_tuple(
            taso::OP_SPLIT,
            tensor_dim_hash(inputs[0]),
            outputsHash,
            p->axis
        );
        return std::hash<decltype(params)>()(params);
      }
  }
};

flexflow::Tensor ConverterVisitor::convertTensor(taso::Tensor const &in) const {
  flexflow::Tensor out;
  out.numDim = in.numDim;
  for (int i = 0; i < in.numDim; i++) {
    out.adim[in.numDim - i - 1] = in.dim[i];
  }
  out.data_type = flexflow::DT_FLOAT;
  out.owner_op = NULL;
  out.owner_idx = 0;
  return out;
}

flexflow::ActiMode convertActiMode(taso::ActiMode const &in) {
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

flexflow::FFModel &ConverterVisitor::get_converted() {
  return this->model;
}

Converter::Converter(flexflow::FFConfig ffconfig, taso::Graph const &graph)
  : v(ffconfig), graph(graph)
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

  this->v.run(this->graph);
  /* std::list<taso::Op> to_convert; */
  /* for (auto const &kv : graph.outEdges) { */
  /*   /1* if (kv.first.ptr != NULL && kv.first.ptr->type == taso::OP_INPUT) { *1/ */
  /*     for (taso::Edge const &edge : kv.second) { */
  /*       if (edge.dstOp.ptr->type != taso::OP_INPUT && edge.dstOp.ptr->type != taso::OP_WEIGHT) { */
  /*         to_convert.push_back(edge.dstOp); */
  /*       } */
  /*     } */
  /*   /1* } *1/ */
  /* } */
  /* taso::Op current; */
  /* bool ready; */
  /* while (!to_convert.empty()) { */
  /*   /1* printf("The queue has %d items\n", to_convert.size()); *1/ */
  /*   current = to_convert.front(); */
  /*   to_convert.pop_front(); */
  /*   auto find_result = graph.inEdges.find(current); */
  /*   assert (find_result != graph.inEdges.end()); */
  /*   ready = true; */
  /*   /1* printf("Are we ready?\n"); *1/ */
  /*   /1* printf("We have %d in edges to examine\n", find_result->second.size()); *1/ */
  /*   for (taso::Edge const &edge : find_result->second) { */
  /*     /1* printf("Examining edge with srcOp id %d and srcOp ptr %p and srcOp type %d\n", edge.srcOp.guid, edge.srcOp.ptr, edge.srcOp.ptr ? edge.srcOp.ptr->type : -1); *1/ */
  /*     if (edge.srcOp.guid != taso::GUID_WEIGHT */
  /*         && edge.srcOp.guid != taso::GUID_INPUT */
  /*         && edge.srcOp.ptr->type != taso::OP_INPUT */
  /*         && edge.srcOp.ptr->type != taso::OP_WEIGHT */
  /*         && this->opMap.find(edge.srcOp) == this->opMap.end()) { */
  /*       ready = false; */
  /*       /1* printf("No we are not!\n"); *1/ */
  /*       break; */
  /*     } */
  /*   } */
  /*   if (ready) { */
  /*     /1* printf("Yes we are!\n"); *1/ */
  /*   } */
  /*   if (!ready) { */
  /*     continue; */
  /*   } else { */
  /*     /1* printf("Converting node with id %d and ptr %p\n", current.guid, current.ptr); *1/ */
  /*     bool didConvert = this->convertOp(current); */
  /*     assert (this->opMap.find(current) != this->opMap.end()); */
  /*     if (didConvert) { */
  /*       /1* printf("Adding the destinations of the %d outEdges\n", graph.outEdges[current].size()); *1/ */
  /*       for (taso::Edge const &outEdge : graph.outEdges[current]) { */
  /*         to_convert.push_back(outEdge.dstOp); */
  /*       } */
  /*     } */
  /*   } */
  /* } */

  if (ossOut != NULL) {
    this->get_converted().to_dot(std::move(ossOut));
  }
}

/* void Converter::rawConvertOp(taso::Op const &op) { */
/*   flexflow::Tensor inputs[MAX_NUM_INPUTS]; */
/*   taso::Tensor *t_inputs = op.ptr->inputs; */
/*   taso::Tensor *t_outputs = op.ptr->outputs; */
/*   for (taso::Edge const &inE : this->graph.inEdges[op]) { */
/*     assert (inE.dstOp == op); */
/*     if (inE.srcOp.guid != taso::GUID_WEIGHT && inE.srcOp.guid != taso::GUID_INPUT && inE.srcOp.ptr->type != taso::OP_INPUT && inE.srcOp.ptr->type != taso::OP_WEIGHT) { */
/*       inputs[inE.dstIdx] = this->opMap.at(inE.srcOp)->outputs[inE.srcIdx]; */
/*     } else { */
/*       inputs[inE.dstIdx] = this->convertTensor(t_inputs[inE.dstIdx]); */
/*     } */
/*   } */
/*   switch (op.ptr->type) { */
/*     case taso::OP_CONV2D: */
/*     { */
/*       taso::Conv2D *p = (taso::Conv2D *)op.ptr; */
/*       int padH, padW, groups; */
/*       p->get_padding(&padH, &padW); */
/*       p->get_int_parameter(taso::PM_GROUP, &groups); */
/*       /1* printf("Creating convolution layer with input channels %d and groups %d\n", t_inputs[0].dim[1], groups); *1/ */
/*       model.conv2d( */
/*         inputs[0], */
/*         t_outputs[0].dim[1], */
/*         t_inputs[1].dim[2], t_inputs[1].dim[3], */
/*         p->strideH, p->strideW, */
/*         padH, padW, */
/*         groups, */
/*         convertActiMode(p->activation), */
/*         true */
/*       ); */
/*       return; */
/*     } */
/*     case taso::OP_EW_ADD: */
/*     { */
/*       model.add( */
/*         inputs[0], */
/*         inputs[1] */
/*       ); */
/*       return; */
/*     } */
/*     case taso::OP_POOL2D_MAX: */
/*     case taso::OP_POOL2D_AVG: */
/*     { */
/*       taso::Pool2D *p = (taso::Pool2D *)op.ptr; */
/*       int padH, padW; */
/*       p->get_padding(&padH, &padW); */
/*       model.pool2d( */
/*         inputs[0], */
/*         p->kernelH, p->kernelW, */
/*         p->strideH, p->strideW, */
/*         padH, padW, */
/*         (op.ptr->type == taso::OP_POOL2D_MAX) ? flexflow::POOL_MAX : flexflow::POOL_AVG, */
/*         convertActiMode(p->activation) */
/*       ); */
/*       return; */
/*     } */
/*     case taso::OP_CONCAT: */
/*     { */
/*       taso::Concat *p = (taso::Concat *)op.ptr; */
/*       std::vector<flexflow::Tensor> tensors; */
/*       for (int i = 0; i < p->n; i++) { */
/*         tensors.push_back(inputs[i]); */
/*       } */
/*       model.concat( */
/*           p->n, */
/*           tensors.data(), */
/*           p->axis */
/*       ); */
/*       return; */
/*     } */
/*     case taso::OP_RELU: */
/*     { */
/*       model.relu( */
/*         inputs[0] */
/*       ); */
/*       return; */
/*     } */
/*     case taso::OP_SPLIT: */
/*     { */
/*       taso::Split *p = (taso::Split *)op.ptr; */
/*       flexflow::Tensor outputs[MAX_NUM_INPUTS]; */
/*       model.split( */
/*         inputs[0], */
/*         outputs, */
/*         p->sizes, */
/*         p->axis */
/*       ); */
/*       return; */
/*     } */
/*     default: */
/*       throw std::runtime_error("Unknown op type " + taso::op_type_name(op.ptr->type) + " in conversion"); */
/*   } */
/* } */

flexflow::FFModel &Converter::get_converted() {
  return this->v.get_converted();
}


