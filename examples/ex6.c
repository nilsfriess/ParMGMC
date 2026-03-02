/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Computes the sample covariance between several chains and computes the relative
 *  error w.r.t. the exact covariance matrix.
 *
 *  NOTE: Dependening on the values of `-chains` and `-ksp_max_it`, this program might
 *        require a substantial amount of memory (e.g., for 50000 chains and 200 samples
          per chain, it requires about ~12 GB).
 */

#include "parmgmc/parmgmc.h"
#include "parmgmc/stats.h"

#include <petscpc.h>
#include <petscsystypes.h>
#include <time.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscvec.h>
#include <mpi.h>

// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t  %opts -ksp_type richardson -pc_type gamgmc -gamgmc_mg_levels_pc_gibbs_forward -chains 1000 -ksp_max_it 200 -kappa 1e-4 -gamgmc_pc_gamg_coarse_eq_limit 10 -skip_petscrc

typedef struct {
  Vec            *samples;
  PetscInt        idx;
  PetscInt        chains;
  PetscLogDouble *times;
} *SampleCtx;

static PetscErrorCode SampleCtxDestroy(void *ctx)
{
  SampleCtx sctx = ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(sctx->times));
  PetscCall(PetscFree(sctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleCallback(PetscInt it, Vec y, void *ctx)
{
  SampleCtx sctx = ctx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(y, sctx->samples[it * sctx->chains + sctx->idx]));
  if (it == 0) {
    sctx->times[0] = MPI_Wtime();
  } else {
    sctx->times[it] += 1. / sctx->chains * (MPI_Wtime() - sctx->times[0]) * 1000;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleMatrix(Mat *A)
{
  PetscInt      n     = 10;
  PetscScalar   kappa = 1, values[5];
  MatStencil    rowstencil, colstencil[5];
  DM            da;
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL));
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateMatrix(da, A));
  for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
      PetscInt k = 0;

      rowstencil.i = i;
      rowstencil.j = j;
      if (i != 0) {
        values[k]       = -1;
        colstencil[k].i = i - 1;
        colstencil[k].j = j;
        ++k;
      }
      if (i != info.mx - 1) {
        values[k]       = -1;
        colstencil[k].i = i + 1;
        colstencil[k].j = j;
        ++k;
      }
      if (j != 0) {
        values[k]       = -1;
        colstencil[k].i = i;
        colstencil[k].j = j - 1;
        ++k;
      }
      if (j != info.my - 1) {
        values[k]       = -1;
        colstencil[k].i = i;
        colstencil[k].j = j + 1;
        ++k;
      }
      colstencil[k].i = i;
      colstencil[k].j = j;
      values[k]       = k + kappa;
      ++k;

      PetscCall(MatSetValuesStencil(*A, 1, &rowstencil, k, colstencil, values, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(*A, NULL, "-prec_mat_view"));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscInt    chains = 1, seed = 0xCAFE, samples_per_chain = 1, n;
  Mat         A;
  KSP         ksp;
  Vec        *samples, b, x;
  SampleCtx   ctx;
  PC          pc;
  PetscRandom pr;
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  if (size != 1) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: This example requires sequential exectuion, skipping"));
    return 0;
  }
  PetscCall(AssembleMatrix(&A));
  PetscCall(MatCreateVecs(A, &b, NULL));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Degrees of freedom: %" PetscInt_FMT "\n", n));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-chains", &chains, NULL));

  PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &pr));
  PetscCall(PetscRandomSetType(pr, PARMGMC_ZIGGURAT));
  PetscCall(PetscRandomSetSeed(pr, seed));
  PetscCall(PetscRandomSeed(pr));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetTolerances(ksp, NULL, NULL, NULL, &samples_per_chain));
  samples_per_chain++;

  PetscCall(PetscMalloc1(samples_per_chain * chains, &samples));
  PetscCall(PetscNew(&ctx));
  PetscCall(PetscCalloc1(samples_per_chain, &ctx->times));
  ctx->samples = samples;
  ctx->chains  = chains;
  for (PetscInt i = 0; i < samples_per_chain * chains; ++i) PetscCall(VecDuplicate(b, &samples[i]));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCViewFromOptions(pc, NULL, "-pc_view"));
  /* PetscCall(PCSetPetscRandom(pc, pr)); */
  /* PetscCall(PetscRandomSetSeed(pr, seed)); */
  /* PetscCall(PetscRandomSeed(pr)); */
  PetscCall(VecDuplicate(b, &x));
  PetscCall(KSPSolve(ksp, b, x)); // Burn-in

  PetscCall(PCSetSampleCallback(pc, SampleCallback, ctx, SampleCtxDestroy));
  for (PetscInt i = 0; i < chains; ++i) {
    ctx->idx = i;
    PetscCall(VecZeroEntries(x));
    PetscCall(KSPSolve(ksp, b, x));
  }

  {
    PetscReal *errs;
    PetscCall(PetscMalloc1(samples_per_chain, &errs));
    PetscCall(EstimateCovarianceMatErrors(A, chains, samples_per_chain, samples, errs));

    if (PETSC_TRUE) {
      FILE  *fptr;
      char   filename[256];
      PCType pctype;

      PetscCall(PCGetType(pc, &pctype));
      PetscCall(PetscSNPrintf(filename, 256, "cov_est_error_%s.txt", pctype));
      fptr          = fopen(filename, "w");
      ctx->times[0] = 0;
      for (PetscInt i = 0; i < samples_per_chain; i++) PetscCall(PetscFPrintf(MPI_COMM_WORLD, fptr, "%.6f %.6f\n", errs[i], ctx->times[i]));
      fclose(fptr);
    }

    /* for (PetscInt i = 0; i < samples_per_chain; ++i) { PetscCall(PetscPrintf(MPI_COMM_WORLD, "%.8f\n", errs[i])); } */
    PetscCall(PetscFree(errs));
  }

  for (PetscInt i = 0; i < samples_per_chain * chains; ++i) PetscCall(VecDestroy(&samples[i]));
  PetscCall(PetscFree(samples));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscRandomDestroy(&pr));
  PetscCall(PetscFinalize());
}
