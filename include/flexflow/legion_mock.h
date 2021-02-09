#ifndef _FLEXFLOW_LEGION_MOCK_H_
#define _FLEXFLOW_LEGION_MOCK_H_

#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <set>

#define MAX_TENSOR_DIM 4
#define LEGION_MAX_DIM 4

namespace Legion {
  using MappingTagID = size_t;
  using coord_t = long long;

  class Domain {
  public:
    enum { MAX_RECT_DIM = LEGION_MAX_DIM };

    Domain();

    bool operator==(Domain const &other) const;

    size_t get_volume() const;
    Domain intersection(Domain const &other) const;
  public:
    int dim;
    coord_t rect_data[2*MAX_RECT_DIM];
  };
}

#endif // _FLEXFLOW_LEGION_MOCK_H
