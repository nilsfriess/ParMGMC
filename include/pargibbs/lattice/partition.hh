#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stack>
#include <vector>

#include "pargibbs/common/log.hh"
#include "types.hh"

namespace pargibbs::detail {
template <std::size_t dim> using nd_id = std::array<std::size_t, dim>;

template <std::size_t dim> struct partition {
  nd_id<dim> start;
  nd_id<dim> size;
  std::size_t weight; // only used within `worb`, can be ignored outside
};

// Performs weighted orthogonal recursive bisection to partition a rectangular
// grid in dimension `dim` with `dimensions[d]` nodes along dimension `d` into
// `n_partitions` partitions. The resulting partition approximately minimises
// the total length of the boundaries between the partitions (which
// appproximately minimises MPI communication).
//
// Returns the list of partitions where each partition holds its `start`
// coordinate and its `size`.
// TODO: Currently only supports 2D grids.
template <ParallelLayout layout, std::size_t dim,
          std::enable_if_t<layout == ParallelLayout::WORB, bool> = true>
inline std::vector<partition<dim>>
make_partition(const std::array<std::size_t, dim> &dimensions,
               std::size_t n_partitions) {
  static_assert(dim == 2, "Only dim == 2 supported currently");

  std::vector<partition<dim>> final_partitions;
  final_partitions.reserve(n_partitions);
  std::stack<partition<dim>> unfinished_partitions;

  partition<dim> initial_partition;
  initial_partition.start = nd_id<dim>{0};
  initial_partition.size = dimensions;
  initial_partition.weight = n_partitions;

  if (n_partitions == 1) {
    final_partitions.push_back(std::move(initial_partition));
    return final_partitions;
  }

  unfinished_partitions.push(std::move(initial_partition));

  // Recursively traverse list of partitions that are not small enough yet
  while (not unfinished_partitions.empty()) {
    auto cur_partition = unfinished_partitions.top();
    unfinished_partitions.pop();

    std::size_t total_points = 1;
    for (auto d : cur_partition.size)
      total_points *= d;

    // TODO: This assumes 2D
    auto cut_dim = std::distance(
        cur_partition.size.begin(),
        std::max_element(cur_partition.size.begin(), cur_partition.size.end()));
    auto other_dim = 1 - cut_dim; // cut_dim = 0 => other_dim = 1 and vice versa

    auto weight_left =
        static_cast<std::size_t>(std::floor(cur_partition.weight / 2.));
    auto n_left = total_points * weight_left / cur_partition.weight;

    auto weight_right =
        static_cast<std::size_t>(std::ceil(cur_partition.weight / 2.));
    // auto n_right = total_points * weight_right / cur_partition.weight;

    partition<dim> left;
    left.size[cut_dim] = n_left / cur_partition.size[other_dim];
    left.size[other_dim] = cur_partition.size[other_dim];
    left.start = cur_partition.start;
    left.weight = weight_left;

    partition<dim> right;
    right.size[cut_dim] = cur_partition.size[cut_dim] - left.size[cut_dim];
    right.size[other_dim] = cur_partition.size[other_dim];
    right.start = cur_partition.start;
    right.start[cut_dim] += left.size[cut_dim];
    right.weight = weight_right;

    if (left.weight == 1)
      final_partitions.push_back(std::move(left));
    else
      unfinished_partitions.push(std::move(left));

    if (right.weight == 1)
      final_partitions.push_back(std::move(right));
    else
      unfinished_partitions.push(std::move(right));
  }

  return final_partitions;
}

// Partitions a rectangular grid in dimension `dim` with `dimensions[d]` nodes
// along dimension `d` into `n_partitions` partitions by slicing the domain into
// rows of approximately equal size along dimension `dim` (if the domain cannot
// be distributed equally, one partition is assigned a larger subdomain; no load
// balancing is performed).
//
// Returns the list of partitions where each partition holds its `start`
// coordinate and its `size`.
template <ParallelLayout layout, std::size_t dim,
          std::enable_if_t<layout == ParallelLayout::BlockRow, bool> = true>
inline std::vector<partition<dim>>
make_partition(const std::array<std::size_t, dim> &dimensions,
               std::size_t n_partitions) {
  std::vector<partition<dim>> partitions;
  partitions.reserve(n_partitions);

  const std::size_t len = dimensions[dim - 1] / n_partitions;
  if (len == 0) {
    PARGIBBS_DEBUG << "Error: Cannot partition along dimension of length "
                   << dimensions[dim - 1] << " into " << n_partitions
                   << " partitions.\n";
    throw std::runtime_error(
        "Error during block-row partitioning. Too many partitions requested");
  }

  for (std::size_t i = 0; i < n_partitions; ++i) {
    partition<dim> part;

    for (std::size_t d = 0; d < dim - 1; ++d)
      part.start[d] = 0;
    part.start[dim - 1] = i * len;

    for (std::size_t d = 0; d < dim - 1; ++d)
      part.size[d] = dimensions[d];
    part.size[dim - 1] = len;

    // Last partition might be bigger
    if (i == n_partitions - 1)
      part.size[dim - 1] = dimensions[dim - 1] - part.start[dim - 1];

    partitions.push_back(std::move(part));
  }

  return partitions;
}

}; // namespace pargibbs::detail
