#ifndef _FLEXFLOW_DOT_FILE_H
#define _FLEXFLOW_DOT_FILE_H

#include <string>
#include <map>
#include <memory>
#include <sstream>
#include <fstream>

template <typename T>
class DotFile {
private:
  size_t node_id;
  std::map<T,size_t> node_ids;
  std::unique_ptr<std::ostream> out;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
public:
  DotFile() : node_id(0) {}
  DotFile(std::string const &filename) : DotFile(std::unique_ptr<std::ostream>(new std::ofstream(filename))) {}
  DotFile(std::unique_ptr<std::ostream> s)
    : node_id(0), out(std::move(s))
  {
    *out << "digraph taskgraph {";
  }

  void set_filename(std::string filename) {
    this->out = std::unique_ptr<std::ostream>(new std::ofstream(filename));
    *out << "digraph taskgraph {";
  }
  void reserve_node(T const &t) {
    if (this->node_ids.find(t) == this->node_ids.end()) {
      this->node_ids[t] = this->node_id++;
    }
  }
  void add_node(T const &t, std::map<std::string, std::string> const &params) {
    this->reserve_node(t);
    *out << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
    for (auto it = params.begin(); it != params.end(); ++it)  {
      *out << it->first << "=" << it->second;
      if (std::next(it) != params.end()) {
        *out << ",";
      }
    }
    *out << "];" << std::endl;
  }
  void add_edge(T const &src, T const &dst) {
    this->reserve_node(src);
    this->reserve_node(dst);
    auto src_name = this->get_node_name(this->node_ids.at(src));
    auto dst_name = this->get_node_name(this->node_ids.at(dst));
    *out << "  " << src_name << " -> " << dst_name << ";" << std::endl;
  }
  void close() {
    *out << "}";
    out->flush();
  }
};

#endif // _FLEXFLOW_DOT_FILE_H
