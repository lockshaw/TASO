#ifndef _BOUNDEDPQ_H_
#define _BOUNDEDPQ_H_

#include <queue>

template <typename T, typename Compare>
class ReverseCompare {
private:
  Compare comp;
public:
  bool operator()(T const &x, T const &y) const {
    return comp(y, x);
  }
};

template <
  typename T,
  typename Container = std::vector<T>,
  typename Compare = std::less<typename Container::value_type>
>
class BoundedPriorityQueue {
private:
  typename Container::size_type max_size;
  Compare comp;

  std::priority_queue<T, Container, ReverseCompare<T,Compare>> queue;
public:
  BoundedPriorityQueue(typename Container::size_type size) : max_size(size) { };

  bool push(T t, T *evicted = nullptr) {
    if (queue.size() < max_size) {
      this->queue.push(t);
      return false;
    } else if (comp(queue.top(), t)) {
      if (evicted != nullptr) {
        *evicted = this->queue.top();
      }
      this->queue.pop();
      this->queue.push(t);
      return true;
    }
    if (evicted != nullptr) {
      *evicted = t;
    }
    return true;
  }

  template <class... Args>
  void emplace(Args&&... args)  {
    this->queue.emplace(std::forward<Args>(args)...);
  }

  bool empty() const {
    return this->queue.empty();
  }

  typename Container::size_type size() const {
    return this->queue.size();
  }

  T pop() {
    T value = this->queue.top();
    this->queue.pop();
    return value;
  }

  T top() const {
    return this->queue.top();
  }
};

#endif // _BOUNDEDPQ_H
