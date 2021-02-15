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

class HashVisitor : public TopoVisitor {
public:
  HashVisitor();

  void visitNode(taso::Op const &op) override;
  size_t get_hash() const;
private:
  size_t hash;
};

namespace std {
  template <>
  struct hash<taso::Graph> {
    size_t operator()(taso::Graph const &) const;
  };
}

/* class DeterministicTopoVisitor : public DeterministicGraphVisitor { */

/* }; */

flexflow::ActiMode convertActiMode(taso::ActiMode const &);

bool is_input(taso::Op const &op);
bool is_weight(taso::Op const &op);

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
  /* std::map<taso::Op, flexflow::Op *> opMap; */
  //std::map<flexflow::Tensor*, taso::Op> tensors;
  /* flexflow::FFModel model; */
  taso::Graph graph;
  ConverterVisitor v;
};

struct OpParamHash {
  size_t operator()(taso::Op const &) const;
};

#endif // _FLEXFLOW_CONVERT_H
