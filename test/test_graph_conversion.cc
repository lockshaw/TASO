#include "catch/catch.h"
#include "taso/ops.h"
#include "flexflow/convert.h"

using namespace taso;

TensorHandle new_input(Graph *g, std::vector<int> const &dims) {
  return g->new_input(dims.size(), dims.data());
}

TensorHandle new_weight(Graph *g, std::vector<int> const &dims) {
  return g->new_weight(dims.size(), dims.data(), NULL);
}

TEST_CASE("Basic graph conversion", "[graph conv]") {
  Graph *graph = new Graph();
  auto inp1 = new_input(graph, {100, 3, 224, 224});
  auto inp2 = new_input(graph, {100, 3, 224, 224});
  auto inp2p = graph->relu(inp2);
  auto out = graph->element(OP_EW_ADD, inp1, inp2p);
  flexflow::FFConfig config;
  Converter c(config, *graph);
  c.convert();
  flexflow::FFModel &converted = c.get_converted();
  REQUIRE (converted.layers.size() == 2);
  std::set<flexflow::OperatorType> opTypes;
  for (auto const &layer : converted.layers) {
    opTypes.insert(layer->op_type);
  }
  std::set<flexflow::OperatorType> correctAnswer { flexflow::OP_EW_ADD, flexflow::OP_RELU };
  REQUIRE ( opTypes == correctAnswer );
}

TEST_CASE("Weight graph conversion", "[graph conv]") {
  Graph *graph = new Graph();
  auto inp1 = new_input(graph, {100, 3, 224, 224});
  auto wei1 = new_weight(graph, {100, 3, 224, 224});
  auto wei1p = graph->relu(wei1);
  auto out = graph->element(OP_EW_ADD, inp1, wei1p);
  flexflow::FFConfig config;
  Converter c(config, *graph);
  c.convert();
  flexflow::FFModel &converted = c.get_converted();
  REQUIRE (converted.layers.size() == 1);
  REQUIRE (converted.layers[0]->op_type == flexflow::OP_EW_ADD);
}
