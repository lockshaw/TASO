#include "flexflow/legion_mock.h"
#include <algorithm>

using namespace std;
using namespace Legion;

Domain::Domain()
  : dim(0)
{ }

bool Domain::operator==(Domain const &other) const {
  if (this->dim != other.dim) {
    return false;
  }
  for (int i = 0; i < this->dim; i++) {
    if (this->rect_data[i] != other.rect_data[i]) {
      return false;
    }
  }
  return true;
}

int Domain::get_dim() const {
  return this->dim;
}

coord_t const *Domain::get_lo() const {
  return this->rect_data;
}

coord_t const *Domain::get_hi() const {
  return this->rect_data + MAX_RECT_DIM;
}

size_t Domain::get_volume() const {
  size_t volume = 1;

  int lo, hi;
  for (int i = 0; i < dim; i++) {
    lo = this->rect_data[i];
    hi = this->rect_data[MAX_RECT_DIM + i];
    if (hi <= lo) {
      return 0;
    }
    volume *= (hi - lo + 1);
  }

  return volume;
}

Domain Domain::intersection(Domain const &other) const {
  Domain inter;
  for (int i = 0; i < dim; i++) {
    inter.rect_data[i] = std::max(this->rect_data[i], other.rect_data[i]);
    inter.rect_data[MAX_RECT_DIM+i] = std::min(this->rect_data[MAX_RECT_DIM+i], other.rect_data[MAX_RECT_DIM+i]);
  }

  return inter;
}

