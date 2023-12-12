#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/cholesky.hh"
#include "parmgmc/samplers/sampler_preconditioner.hh"
#include "parmgmc/samplers/sor.hh"
#include "parmgmc/samplers/sor_preconditioner.hh"

#include <iostream>
#include <memory>
#include <random>

#include <mpi.h>

#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class MultigridSampler {
public:
  template <class MatAssembler>
  MultigridSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
                   std::size_t n_levels, MatAssembler &&mat_assembler)
      : ops(n_levels) {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    PetscFunctionBeginUser;

    /* As in the SORSampler, we create a full Krylov solver but set it to only
     * run the (Multigrid) preconditioner. */
    call(KSPCreate(MPI_COMM_WORLD, &ksp));
    call(KSPSetType(ksp, KSPPREONLY));
    call(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));
    // call(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

    PC prec;
    call(KSPGetPC(ksp, &prec));
    call(PCSetType(prec, PCMG));

    call(PCMGSetLevels(prec, n_levels, NULL));

    // Don't coarsen operators using Galerkin product, but rediscretize (see
    // below)
    call(PCMGSetGalerkin(prec, PC_MG_GALERKIN_NONE));

    // Create hierachy of meshes and operators
    for (std::size_t level = 0; level < n_levels - 1; ++level)
      ops[level] = std::make_shared<GridOperator>();

    ops[n_levels - 1] = grid_operator;

    for (std::size_t level = n_levels - 1; level > 0; --level) {
      call(DMCoarsen(ops[level]->dm, MPI_COMM_NULL, &(ops[level - 1]->dm)));

      call(DMCreateMatrix(ops[level - 1]->dm, &(ops[level - 1]->mat)));
      call(mat_assembler(ops[level - 1]->mat, ops[level - 1]->dm));
    }

    // Setup multigrid sampler
    // using PCSampler = SamplerPreconditioner<SORSampler<Engine>>;

    for (std::size_t level = 0; level < n_levels; ++level) {
      KSP ksp_level;
      PC pc_level;

      /* We configure the smoother on each level to be a preconditioned
         Richardson smoother with a (stochastic) Gauss-Seidel preconditioner. */
      call(PCMGGetSmoother(prec, level, &ksp_level));
      call(KSPSetType(ksp_level, KSPRICHARDSON));
      call(KSPSetOperators(ksp_level, ops[level]->mat, ops[level]->mat));
      call(KSPSetInitialGuessNonzero(ksp_level, PETSC_TRUE));

      // Set preconditioner to be stochastic SOR
      call(KSPGetPC(ksp_level, &pc_level));
      call(PCSetType(pc_level, PCSHELL));

      auto *context =
          new SORRichardsonContext<Engine>(engine, ops[level]->mat, 1.);

      call(PCShellSetContext(pc_level, context));
      call(
          PCShellSetApplyRichardson(pc_level, sor_pc_richardson_apply<Engine>));
      call(PCShellSetDestroy(pc_level, sor_pc_richardson_destroy<Engine>));

      if (level > 0) {
        Mat grid_transfer;
        DM dm_fine = ops[level]->dm;
        DM dm_coarse = ops[level - 1]->dm;

        call(DMCreateInterpolation(dm_coarse, dm_fine, &grid_transfer, NULL));
        call(PCMGSetInterpolation(prec, level, grid_transfer));

        // We can set the interpolation matrix as restriction matrix, PETSc will
        // figure out that it should use the transpose.
        call(PCMGSetRestriction(prec, level, grid_transfer));
        call(MatDestroy(&grid_transfer));
      }
    }

    // KSP ksp_coarse;
    // call(PCMGGetCoarseSolve(prec, &ksp_coarse));
    // call(KSPSetType(ksp_coarse, KSPPREONLY));
    // call(KSPSetOperators(ksp_coarse, ops[0]->mat, ops[0]->mat));

    // PC pc_coarse;
    // using CoarseSampler = SamplerPreconditioner<CholeskySampler<Engine>>;

    // call(KSPGetPC(ksp_coarse, &pc_coarse));
    // call(PCSetType(pc_coarse, PCSHELL));
    // call(CoarseSampler::attach(pc_coarse, ops[0], engine));

    call(PCSetFromOptions(prec));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(KSPSolve(ksp, rhs, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MultigridSampler() { KSPDestroy(&ksp); }

private:
  std::vector<std::shared_ptr<GridOperator>> ops;

  KSP ksp;
};
} // namespace parmgmc
