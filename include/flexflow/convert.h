#ifndef _FLEXFLOW_CONVERT_H
#define _FLEXFLOW_CONVERT_H

#include "flexflow/model.h"
#include "taso/ops.h"

class Converter {
public:
  Converter(flexflow::FFConfig ffconfig, taso::Graph const &graph);
  void convert(std::unique_ptr<std::ostream> oss);

  flexflow::FFModel const &get_converted() const;
private:
  void convertOp(taso::Op const &op);
  void rawConvertOp(taso::Op const &op);

  flexflow::Tensor convertTensor(taso::Tensor const &) const;
  flexflow::ActiMode convertActiMode(taso::ActiMode const &) const;
private:
  std::map<taso::Op, flexflow::Op *> opMap;
  //std::map<flexflow::Tensor*, taso::Op> tensors;
  flexflow::FFModel model;
  taso::Graph graph;
};

#endif // _FLEXFLOW_CONVERT_H
