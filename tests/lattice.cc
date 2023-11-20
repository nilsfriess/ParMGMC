#include <gtest/gtest.h>

#include "parmgmc/lattice/lattice.hh"

TEST(LatticeCoarsening, Lattice2d5x5) {
  namespace pg = parmgmc;

  pg::Lattice lattice(2, 5);

  auto coarse_lattice = lattice.coarsen();

  EXPECT_EQ(coarse_lattice.get_vertices_per_dim(), 3);
  EXPECT_EQ(coarse_lattice.get_n_total_vertices(), 9);
}

TEST(LatticeConstruction, Lexicographic) {
  parmgmc::Lattice lattice(2,
                            3,
                            parmgmc::ParallelLayout::None,
                            parmgmc::LatticeOrdering::Lexicographic);

  std::vector<int> adj_idx_expected = {0, 2, 5, 7, 10, 14, 17, 19, 22, 24};
  std::vector<int> adj_vert_expected = {1, 3, 0, 2, 4, 1, 5, 4, 6, 0, 3, 5,
                                        7, 1, 4, 8, 2, 7, 3, 6, 8, 4, 7, 5};

  EXPECT_EQ(lattice.adj_idx, adj_idx_expected);
  EXPECT_EQ(lattice.adj_vert, adj_vert_expected);
}

TEST(LatticeConstruction, RedBlack) {
  parmgmc::Lattice lattice(2,
                            3,
                            parmgmc::ParallelLayout::None,
                            parmgmc::LatticeOrdering::RedBlack);

  std::vector<int> adj_idx_expected = {0, 2, 4, 8, 10, 12, 15, 18, 21, 24};
  std::vector<int> adj_vert_expected = {5, 6, 5, 7, 6, 7, 8, 5, 8, 6, 8, 7,
                                        0, 1, 2, 2, 3, 0, 2, 4, 1, 3, 4, 2};

  EXPECT_EQ(lattice.adj_idx, adj_idx_expected);
  EXPECT_EQ(lattice.adj_vert, adj_vert_expected);

  std::vector<int> own_vertices_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(lattice.own_vertices, own_vertices_expected);
}

TEST(LatticeConstruction, NumVertices) {
  // Lattice constructed with 2^n + 1 vertices per dim should not change this
  // number in its constructor
  parmgmc::Lattice l1(2, 5);
  parmgmc::Lattice l2(2, 33);

  EXPECT_EQ(l1.get_vertices_per_dim(), 5);
  EXPECT_EQ(l2.get_vertices_per_dim(), 33);

  // Lattice constructed with vertices per dim != 2^n + 1 should change this to
  // the next higher number satisfying this
  parmgmc::Lattice l3(2, 4);
  parmgmc::Lattice l4(2, 100);

  EXPECT_EQ(l3.get_vertices_per_dim(), 5);
  EXPECT_EQ(l4.get_vertices_per_dim(), 129);
}
