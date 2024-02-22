#pragma once

#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/types.hh"
#include "petscao.h"
#include "petscdmda.h"
#include "petscdmtypes.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsf.h>
#include <petscviewer.h>
#include <random>

#include <mpi.h>
#include <petscerror.h>
#include <petsclog.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <set>
#include <utility>

namespace parmgmc {
template <class Engine>
inline PetscErrorCode fill_vec_rand(Vec vec, PetscInt size, Engine &engine) {
  static std::normal_distribution<PetscReal> dist;

  PetscFunctionBeginUser;

  PetscLogEvent rng_event;
  PetscCall(PetscHelper::get_rng_event(&rng_event));
  PetscCall(PetscLogEventBegin(rng_event, NULL, NULL, NULL, NULL));

  PetscScalar *r_arr;
  PetscCall(VecGetArrayWrite(vec, &r_arr));
  std::generate_n(r_arr, size, [&]() { return dist(engine); });
  PetscCall(VecRestoreArrayWrite(vec, &r_arr));

  // Estimated using perf
  PetscCall(PetscLogFlops(size * 27));

  PetscCall(PetscLogEventEnd(rng_event, NULL, NULL, NULL, NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> inline PetscErrorCode fill_vec_rand(Vec vec, Engine &engine) {
  PetscFunctionBeginUser;

  PetscInt size;
  PetscCall(VecGetLocalSize(vec, &size));

  PetscCall(fill_vec_rand(vec, size, engine));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode ISColoring_for_Mat(Mat m, ISColoring *coloring) {
  PetscFunctionBeginUser;

  MatColoring mc;
  PetscCall(MatColoringCreate(m, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGJP));
  PetscCall(MatColoringApply(mc, coloring));
  PetscCall(MatColoringDestroy(&mc));

  PetscCall(ISColoringSetType(*coloring, IS_COLORING_LOCAL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode ISColoring_for_Mat(Mat mat, DM dm, ISColoring *coloring) {
  PetscFunctionBeginUser;

  const PetscInt ncolors = 2;

  PetscInt start, end;
  PetscCall(MatGetOwnershipRange(mat, &start, &end));

  // Global indices owned by current MPI rank
  std::vector<PetscInt> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);

  // Convert to natural indices
  AO ao;
  PetscCall(DMDAGetAO(dm, &ao));
  PetscCall(AOPetscToApplication(ao, indices.size(), indices.data()));

  std::vector<ISColoringValue> colors(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] % 2 == 0)
      colors[i] = 0; // red
    else
      colors[i] = 1; // black
  }

  PetscCall(ISColoringCreate(
      MPI_COMM_WORLD, ncolors, colors.size(), colors.data(), PETSC_COPY_VALUES, coloring));
  PetscCall(ISColoringSetType(*coloring, IS_COLORING_LOCAL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VecScatter_for_Mat(Mat m, VecScatter *scatter, Vec sct_vec = nullptr) {
  PetscFunctionBeginUser;

  Mat B;
  const PetscInt *colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(m, NULL, &B, &colmap));

  const PetscInt *Bi, *Bj;
  PetscCall(MatSeqAIJGetCSRAndMemType(B, &Bi, &Bj, NULL, NULL));

  PetscInt local_rows, global_rows;
  PetscCall(MatGetSize(m, &global_rows, NULL));
  PetscCall(MatGetLocalSize(m, &local_rows, NULL));

  /* This array will have non-zero values at indices corresponding to ghost
     vertices. These are identified by looping over the rows of the
     off-diagonal portion B of the given matrix in MPIAIJ format and marking
     columns that contain non-zero entries.

     For example a 5x5 DMDA grid on four processors would lead to an indices
     array (on rank 0) that could look like this:
     indices = {0 0 0 0 0 0 0 0 0 2 0 5 0 8 0 6 7 8 0 0 0 0 0 0 0}.

     The values of the non-zero entries are the rows where B contains non-zero
     columns, using global indexing. These are used below to determine which
     ghost values have to be communicated during red Gibbs sweeps and which
     during black sweeps.
   */
  std::vector<PetscInt> indices(global_rows, 0);
  std::size_t nz_cols = 0;
  for (PetscInt row = 0; row < local_rows; ++row) {
    for (PetscInt k = Bi[row]; k < Bi[row + 1]; ++k) {
      /* We have to use colmap here since B is compactified, i.e., its
       * non-zero columns are {0, ..., nz_cols} (see MatSetUpMultiply_MPIAIJ).
       */
      if (!indices[colmap[Bj[k]]])
        nz_cols++;
      indices[colmap[Bj[k]]] = 1;
    }
  }

  // Form array of needed columns.
  std::vector<PetscInt> ghost_arr(nz_cols);
  PetscInt cnt = 0;
  for (PetscInt i = 0; i < global_rows; ++i)
    if (indices[i])
      ghost_arr[cnt++] = i;

  IS from;
  PetscCall(ISCreateGeneral(
      MPI_COMM_WORLD, ghost_arr.size(), ghost_arr.data(), PETSC_COPY_VALUES, &from));

  bool return_sct_vec = true;
  if (sct_vec == nullptr)
    return_sct_vec = false;

  Vec gvec, lvec;

  PetscCall(MatCreateVecs(B, &lvec, NULL));
  // Create global vec without allocating actualy memory
  PetscCall(VecCreateMPIWithArray(MPI_COMM_WORLD, 1, local_rows, global_rows, nullptr, &gvec));

  PetscCall(VecScatterCreate(gvec, from, lvec, nullptr, scatter));

  if (return_sct_vec)
    sct_vec = lvec;
  else
    PetscCall(VecDestroy(&lvec));

  PetscCall(ISDestroy(&from));
  PetscCall(VecDestroy(&gvec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode make_botmidtop_partition(Mat mat, BotMidTopPartition &partition) {
  PetscFunctionBeginUser;

  // Get the column layout of the matrix, i.e., the mapping from column ->
  // owning MPI rank
  PetscLayout layout;
  PetscCall(MatGetLayouts(mat, nullptr, &layout));

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  Mat A, B;
  const PetscInt *colmap;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &A, &B, &colmap));

  const PetscInt *Bi, *Bj;
  PetscCall(MatSeqAIJGetCSRAndMemType(B, &Bi, &Bj, nullptr, nullptr));

  PetscInt local_rows, global_rows;
  PetscCall(MatGetSize(mat, &global_rows, nullptr));
  PetscCall(MatGetLocalSize(mat, &local_rows, nullptr));

  ////////////////////////////////////////////////////////////////////////
  //// STEP 1. Create top, bot, mid and interior sets                 ////
  ////////////////////////////////////////////////////////////////////////
  partition.clear();
  std::vector<RemoteNode> curr_neighbors;
  for (PetscInt row = 0; row < local_rows; ++row) {
    curr_neighbors.clear();

    for (PetscInt colptr = Bi[row]; colptr < Bi[row + 1]; ++colptr) {
      auto col = colmap[Bj[colptr]];

      PetscMPIInt owner;
      PetscCall(PetscLayoutFindOwner(layout, col, &owner));

      curr_neighbors.emplace_back(col, owner);
    }

    if (curr_neighbors.size() == 0) {
      // No neighbors => interior node
      partition.interior1.push_back(row); // Splitting into int1 and int2 is done below
    } else {
      // At least one neighbor => boundary node
      if (std::all_of(curr_neighbors.begin(), curr_neighbors.end(), [&](const auto &nb) {
            return nb.owner > rank;
          })) {
        // If all neighbors live on higher ranks, then this is a bot node
        for (const auto &nb : curr_neighbors) {
          BoundaryNode node;
          node.index = row;
          node.neighbor = nb;
          partition.bot.push_back(node);
        }
      } else if (std::all_of(curr_neighbors.begin(), curr_neighbors.end(), [&](const auto &nb) {
                   return nb.owner < rank;
                 })) {
        // If all neighbors live on lower ranks, then this is a top node
        for (const auto &nb : curr_neighbors) {
          BoundaryNode node;
          node.index = row;
          node.neighbor = nb;
          partition.top.push_back(node);
        }
      } else {
        // Some neighbors are one higher ranks, some on lower => mid node
        MidNode node;
        node.index = row;
        node.neighbors = std::move(curr_neighbors);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //// STEP 2. Partition interior nodes into int1 & int 2 parts       ////
  ////////////////////////////////////////////////////////////////////////
  const PetscInt *Ai;
  PetscBool done;
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &Ai, nullptr, &done));
  assert(done);

  // Estimate cost of set of vertices by summing number of neighbors of each
  // index
  PetscInt total_int_cost = 0;
  for (auto i : partition.interior1) // Before partitioning, all interior nodes are in interior1
    total_int_cost += Ai[i + 1] - Ai[i];

  PetscInt bot_cost = 0;
  for (auto i : partition.bot)
    bot_cost += Ai[i.index + 1] - Ai[i.index];

  PetscInt top_cost = 0;
  for (auto i : partition.top)
    top_cost += Ai[i.index + 1] - Ai[i.index];

  // We want to approximately achieve cost(int1) + cost(top) = cost(int2) +
  // cost(bot). We have cost(int1) = (cost(int) + cost(bot) - cost(top)) / 2, where cost(int) is
  // the total cost of all interior nodes.
  const auto tgt_int1_cost = (total_int_cost + bot_cost - top_cost) / 2.;

  int curr_int1_cost = 0;
  std::size_t i;
  for (i = 0; i < partition.interior1.size(); ++i) {
    const auto int_idx = partition.interior1[i];
    curr_int1_cost += Ai[int_idx + 1] - Ai[int_idx];

    if (curr_int1_cost > tgt_int1_cost)
      break;
  }

  // Split partition.interior1 at index 1 and move rest to partition.interior2
  partition.interior2.insert(
      partition.interior2.end(), partition.interior1.begin() + i, partition.interior1.end());
  partition.interior1.resize(i);

  PetscCall(MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, nullptr, &Ai, nullptr, &done));

  ////////////////////////////////////////////////////////////////////////
  //// STEP 3. Create VecScatters for top and bot communication       ////
  ////////////////////////////////////////////////////////////////////////

  // // Create high_to_low
  // std::vector<PetscInt> ghost_indices;
  // for (auto row : partition.bot) {
  //   for (PetscInt colptr = Bi[row]; colptr < Bi[row + 1]; ++colptr) {
  //     auto col = colmap[Bj[colptr]];
  //     ghost_indices.push_back(col);
  //   }
  // }

  // for (auto row : mid) {
  //   for (PetscInt colptr = Bi[row]; colptr < Bi[row + 1]; ++colptr) {
  //     auto col = colmap[Bj[colptr]];

  //     PetscMPIInt owner;
  //     PetscCall(PetscLayoutFindOwner(layout, col, &owner));

  //     if (owner > rank)
  //       ghost_indices.push_back(col);
  //   }
  // }

  // // The scatters scatter into a vector with indices 0,1,...,n (where n is the
  // // number of ghost indices). That is, we don't store the "target indices" of
  // // the the ghost values.
  // IS from;
  // PetscCall(ISCreateGeneral(
  //     MPI_COMM_WORLD, ghost_indices.size(), ghost_indices.data(), PETSC_COPY_VALUES, &from));

  // Vec from_vec, to_vec;
  // PetscCall(MatCreateVecs(mat, &from_vec, nullptr));
  // PetscCall(VecCreateSeq(MPI_COMM_SELF, ghost_indices.size(), &to_vec));

  // PetscCall(VecScatterCreate(from_vec, from, to_vec, nullptr, &partition.high_to_low));

  // PetscCall(VecDestroy(&to_vec));
  // PetscCall(ISDestroy(&from));

  // // Create low_to_high
  // ghost_indices.clear();
  // for (auto row : partition.top) {
  //   for (PetscInt colptr = Bi[row]; colptr < Bi[row + 1]; ++colptr) {
  //     auto col = colmap[Bj[colptr]];
  //     ghost_indices.push_back(col);
  //   }
  // }

  // for (auto row : mid) {
  //   for (PetscInt colptr = Bi[row]; colptr < Bi[row + 1]; ++colptr) {
  //     auto col = colmap[Bj[colptr]];

  //     PetscMPIInt owner;
  //     PetscCall(PetscLayoutFindOwner(layout, col, &owner));

  //     if (owner < rank)
  //       ghost_indices.push_back(col);
  //   }
  // }

  // PetscCall(ISCreateGeneral(
  //     MPI_COMM_WORLD, ghost_indices.size(), ghost_indices.data(), PETSC_COPY_VALUES, &from));

  // PetscCall(VecCreateSeq(MPI_COMM_SELF, ghost_indices.size(), &to_vec));

  // PetscCall(VecScatterCreate(from_vec, from, to_vec, nullptr, &partition.low_to_high));

  // PetscCall(VecDestroy(&from_vec));
  // PetscCall(VecDestroy(&to_vec));
  // PetscCall(ISDestroy(&from));

  ////////////////////////////////////////////////////////////////////////
  //// STEP 4. Populate partition.mid nodes                           ////
  ////////////////////////////////////////////////////////////////////////

  // Send our top nodes to the MPI rank that needs them

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
