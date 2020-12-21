#define CATCH_CONFIG_MAIN
#include "catch/catch.h"
#include "taso/boundedpq.h"

TEST_CASE("Bounded priority queue works", "[boundedpq]") {
  BoundedPriorityQueue<int> q( 3 );
  REQUIRE( q.size() == 0 );
  REQUIRE( q.empty() );
  q.push(2);
  q.push(4);
  REQUIRE( q.size() == 2 );
  q.push(3);
  q.push(1);
  REQUIRE( q.size() == 3 );
  q.push(5);
  REQUIRE( q.size() == 3 );
  REQUIRE( q.pop() == 3 );
  REQUIRE( q.pop() == 4 );
  REQUIRE( q.pop() == 5 );
}
