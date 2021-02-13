#include "catch/catch.h"
#include "taso/ops.h"
#include "flexflow/convert.h"

using namespace taso;

TEST_CASE("Basic graph conversion", "[graph conv]") {
  Graph *graph = new Graph();
  std::vector<int> dims = {100, 3, 224, 224};
  auto inp1 = graph->new_input(dims.size(), dims.data());
  auto inp2 = graph->new_input(dims.size(), dims.data());
  auto out = graph->element(OP_EW_ADD, inp1, inp2);
  flexflow::FFConfig config;
  Converter c(config, *graph);
  c.convert();
  flexflow::FFModel &converted = c.get_converted();
  REQUIRE (converted.layers.size() == 1);
  REQUIRE (converted.layers[0]->op_type == flexflow::OP_EW_ADD);
}
