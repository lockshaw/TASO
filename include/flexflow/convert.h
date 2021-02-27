#ifndef _FLEXFLOW_CONVERT_H
#define _FLEXFLOW_CONVERT_H

#include "flexflow/model.h"
#include "taso/ops.h"
#include <queue>

class GraphVisitor {
public:
  virtual void visit(taso::Op const &op, std::queue<taso::Op> &toVisit) = 0;
  virtual bool isStartingPoint(taso::Op const &op) = 0;

  void run(taso::Graph const &graph);
protected:
  taso::Graph const *graph;
};

/* class DeterministicGraphVisitor { */
/* public: */
/*   virtual void visit(taso::Op const &op, std::function<void(taso::Op const &)> v) = 0; */
/*   virtual bool isStartingPoint(taso::Op const &op) = 0; */

/*   void run(taso::Graph const &graph); */
/* protected: */
/*   taso::Graph const *graph; */
/* }; */

class TopoVisitor : public GraphVisitor {
public:
  void visit(taso::Op const &op, std::queue<taso::Op> &toVisit) override;

  virtual void visitNode(taso::Op const &op) = 0;
  bool isStartingPoint(taso::Op const &op) override;
private:
  std::set<taso::Op> visited;
};

flexflow::ActiMode convertActiMode(taso::ActiMode const &);

class ConverterVisitor : public TopoVisitor {
public:
  ConverterVisitor(flexflow::FFConfig const &config);

  void visitNode(taso::Op const &op) override;
  bool isStartingPoint(taso::Op const &op) override;
  flexflow::FFModel &get_converted();
  void rawConvertOp(taso::Op const &op);

  flexflow::Tensor convertTensor(taso::Tensor const &) const;
private:
  std::map<taso::Op, flexflow::Op *> opMap;
  std::set<taso::Op> isDerivedFromWeight;
  flexflow::FFModel model;
};

class Converter {
public:
  Converter(flexflow::FFConfig ffconfig, taso::Graph const &graph);
  void convert(std::unique_ptr<std::ostream> in, std::unique_ptr<std::ostream> out);
  void convert();

  flexflow::FFModel &get_converted();
private:
  bool convertOp(taso::Op const &op);
private:
  taso::Graph graph;
  ConverterVisitor v;
};

#endif // _FLEXFLOW_CONVERT_H
