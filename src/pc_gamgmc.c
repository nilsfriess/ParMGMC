/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_chols.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewertypes.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

/** @file pc_gamgmc.c
    @brief A geometric algebraic Multigrid Monte Carlo sampler wrapped as a PETSc PC.

    # Options databse keys
    - `-pc_gamgmc_mg_type` - The type of the underlying multigrid PC. Can be mg or gamg.
      Default is gamg (i.e. algebraic Multigrid Monte Carlo).

    # Notes
    
    This is essentially a wrapper around PETSc's `PCMG` or `PCGAMG` multigrid
    preconditioner that handles the case where the system matrix is of type
    `MATLRC` which represents a low-rank update of a matrix
    \f$A + B \Sigma^{-1} B^T\f$. If the matrix is a simple `MATAIJ` matrix,
    then `PCMG`/`PCGAMG` could also be used directly.

    The underyling multigrid `PC` can be configured using the options database by
    prepending the prefix `gamgmc_`. For example, a three-level MGMC sampler
    to generate 100 samples with Gibbs samplers used as random smoothers on each
    level using four iterations on the coarsest level and two iterations on the
    remaining levels can be configured with the following options:

        -ksp_type richardson -pc_type gamgmc
        -pc_gamgmc_mg_type gamg
        -gamgmc_mg_levels_ksp_type richardson
        -gamgmc_mg_levels_pc_type gibbs
        -gamgmc_mg_coarse_ksp_type richardson
        -gamgmc_mg_coarse_pc_type gibbs
        -gamgmc_mg_levels_ksp_max_it 2
        -gamgmc_mg_coarse_ksp_max_it 4
        -gamgmc_pc_mg_levels 3
        -ksp_max_it 100

    Note that you have to provide additional information about the coarser grid
    matrices and grid transfer operators if you want to use the geometric version
    of this sampler by attaching a `DM` to the outer `KSP` via `KSPSetDM(ksp, dm)`.

    The underlying PCGAMG preconditioner can also be extracted using the function
    `PCGAMGMCGetInternalPC(PC, PC*)`.
*/

typedef struct _PC_GAMGMC {
  char      mgtype[64];
  PC        mg;
  Mat      *As; // The actual matrices used (in case of A+LR this differs from the matrices used to setup the multigrid hierarchy).
  PetscBool setup_called;

  KSP ksp;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_GAMGMC;

static PetscErrorCode PCDestroy_GAMGMC(PC pc)
{
  PC_GAMGMC pg = pc->data;
  PetscInt  levels;

  PetscFunctionBeginUser;
  if (pg->As) {
    PetscCall(PCMGGetLevels(pg->mg, &levels));
    for (PetscInt l = 0; l < levels - 1; ++l) PetscCall(MatDestroy(&(pg->As[l])));
    PetscCall(PetscFree(pg->As));
  }
  PetscCall(PCDestroy(&pg->mg));
  PetscCall(PetscFree(pg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGAMGMCGetInternalPC(PC pc, PC *mg)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  if (mg) *mg = pg->mg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGAMGMCSetLevels(PC pc, PetscInt levels)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCMGSetLevels(pg->mg, levels, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGMC_SetUpHierarchy(PC pc)
{
  PetscInt  levels;
  PC_GAMGMC pg = pc->data;
  MatType   type;
  PetscBool islrc;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &islrc));

  PetscCall(PCMGGetLevels(pg->mg, &levels));
  if (islrc) {
    PetscCall(PetscMalloc1(levels, &pg->As));

    pg->As[levels - 1] = pc->pmat;
    for (PetscInt l = levels - 1; l > 0; --l) {
      Mat Ac, Bf, Bc, Ip;
      Vec Sf;
      KSP kspc;
      PC  pcc;

      PetscCall(MatLRCGetMats(pg->As[l], NULL, &Bf, &Sf, NULL));
      PetscCall(PCMGGetSmoother(pg->mg, l - 1, &kspc));
      PetscCall(KSPGetPC(kspc, &pcc));
      PetscCall(PCGetOperators(pcc, NULL, &Ac));
      PetscCall(PCMGGetInterpolation(pg->mg, l, &Ip));

      PetscCall(MatTransposeMatMult(Ip, Bf, MAT_INITIAL_MATRIX, 1, &Bc));
      PetscCall(MatCreateLRC(Ac, Bc, Sf, NULL, &(pg->As[l - 1])));
      PetscCall(MatDestroy(&Bc));
    }

    for (PetscInt l = levels - 1; l >= 0; --l) {
      KSP ksps;
      PC  pcs;

      PetscCall(PCMGGetSmoother(pg->mg, l, &ksps));
      PetscCall(KSPGetPC(ksps, &pcs));
      PetscCall(KSPReset(ksps));
      PetscCall(KSPSetOperators(ksps, pg->As[l], pg->As[l]));
      PetscCall(KSPSetUp(ksps));
    }
  }

  {
    KSP       ksps;
    PC        pcs;
    PCType    ptype;
    PetscBool ischol;
    Mat       A;

    PetscCall(PCMGGetSmoother(pg->mg, 0, &ksps));
    PetscCall(KSPGetPC(ksps, &pcs));

    // TODO: We just assume here that the coarse grid sampler works only on a single MPI rank, but this can be changed at runtime in GAMG.
    // We need to find a way to query this here.
    PetscCall(PCGetType(pcs, &ptype));
    PetscCall(PetscStrcmp(ptype, PCCHOLSAMPLER, &ischol));
    if (ischol) {
      PetscCall(KSPGetOperators(ksps, &A, NULL));
      PetscCall(PetscObjectReference((PetscObject)A));
      PetscCall(PCReset(pcs));
      // PetscCall(PCCholSamplerSetIsCoarseGAMG(pcs, PETSC_TRUE));
      PetscCall(KSPSetOperators(ksps, A, A));
      PetscCall(PCSetUp(pcs));
      PetscCall(PetscObjectDereference((PetscObject)A));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_GAMGMC(PC pc, Vec x, Vec y)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  if (!pg->setup_called) {
    PetscCall(PCGAMGMC_SetUpHierarchy(pc));
    pg->setup_called = PETSC_TRUE;
  }
  PetscCall(PCApply(pg->mg, x, y));

  if (pg->scb) {
    /* For some weird and annoying reason, using the PCApplyRichardson function below (which would be faster (because it avoids
     the computation of the residual r = b - Ax) does not work. I don't understand why but just grabbing the solution vector
     from the KSP directly seems to work. */
    PetscInt it;
    Vec      z;

    PetscCall(KSPGetIterationNumber(pg->ksp, &it));
    PetscCall(KSPGetSolution(pg->ksp, &z));
    PetscCall(pg->scb(it, z, pg->cbctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* static PetscErrorCode PCApplyRichardson_GAMGMC(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason) */
/* { */
/*   (void)rtol; */
/*   (void)abstol; */
/*   (void)dtol; */
/*   (void)guesszero; */
/*   (void)w; */

/*   PC_GAMGMC pg = pc->data; */
/*   PetscInt  it = 0; */

/*   PetscFunctionBeginUser; */
/*   if (!pg->setup_called) { */
/*     PetscCall(PCGAMGMC_SetUpHierarchy(pc)); */
/*     pg->setup_called = PETSC_TRUE; */
/*   } */

/*   if (pg->scb) PetscCall(pg->scb(it, y, pg->cbctx)); */
/*   for (it = 1; it < its; ++it) { */
/*     PetscCall(PCApply(pg->mg, b, y)); */
/*     if (pg->scb) PetscCall(pg->scb(it, y, pg->cbctx)); */
/*   } */

/*   *outits = its; */
/*   *reason = PCRICHARDSON_CONVERGED_ITS; */
/*   PetscFunctionReturn(PETSC_SUCCESS); */
/* } */

static PetscErrorCode PCView_GAMGMC(PC pc, PetscViewer v)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCView(pg->mg, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_GAMGMC(PC pc)
{
  PC_GAMGMC   pg = pc->data;
  MatType     type;
  Mat         P;
  PetscBool   islrc;
  const char *prefix;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(pg->mg, pg->mgtype));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PCSetOptionsPrefix(pg->mg, prefix));
  PetscCall(PCAppendOptionsPrefix(pg->mg, "gamgmc_"));
  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &islrc));
  if (islrc) PetscCall(MatLRCGetMats(pc->pmat, &P, NULL, NULL, NULL));
  else P = pc->pmat;

  PetscCall(PCSetOperators(pg->mg, P, P));
  if (strcmp(pg->mgtype, PCMG) == 0) {
    PetscCall(PCSetDM(pg->mg, pc->dm));
    // PetscCall(PCMGSetGalerkin(pg->mg, PC_MG_GALERKIN_BOTH));
  }

  // Ugly way to set the default "smoother" (=sampler) to be Gibbs.
  // NOTE: PetscOptionsSetValue does NOT honour the PetscOptionsPrefixPush stack;
  // the option name must be fully qualified (prefix + suffix) manually.
  {
    PetscBool flag;
    char      opt[512];

    PetscCall(PCGetOptionsPrefix(pg->mg, &prefix));

    // Always force Richardson iteration so the PC is called as a pure sampler.
    PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_levels_ksp_type", prefix ? prefix : ""));
    PetscCall(PetscOptionsSetValue(NULL, opt, KSPRICHARDSON));
    PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_coarse_ksp_type", prefix ? prefix : ""));
    PetscCall(PetscOptionsSetValue(NULL, opt, KSPRICHARDSON));

    PetscCall(PetscOptionsHasName(NULL, prefix, "-mg_levels_ksp_max_it", &flag));
    if (!flag) {
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_levels_ksp_max_it", prefix ? prefix : ""));
      PetscCall(PetscOptionsSetValue(NULL, opt, "2"));
    }

    PetscCall(PetscOptionsHasName(NULL, prefix, "-mg_coarse_ksp_max_it", &flag));
    if (!flag) {
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_coarse_ksp_max_it", prefix ? prefix : ""));
      PetscCall(PetscOptionsSetValue(NULL, opt, "1"));
    }

    PetscCall(PetscOptionsHasName(NULL, prefix, "-mg_levels_pc_type", &flag));
    if (!flag) {
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_levels_pc_type", prefix ? prefix : ""));
      PetscCall(PetscOptionsSetValue(NULL, opt, PCGIBBS));
    }

    PetscCall(PetscOptionsHasName(NULL, prefix, "-mg_coarse_pc_type", &flag));
    if (!flag) {
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_coarse_pc_type", prefix ? prefix : ""));
      PetscCall(PetscOptionsSetValue(NULL, opt, PCCHOLSAMPLER));
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%smg_coarse_pc_cholsampler_coarse_gamg", prefix ? prefix : ""));
      PetscCall(PetscOptionsSetValue(NULL, opt, ""));
    }
  }

  PetscCall(PCSetFromOptions(pg->mg));
  PetscCall(PCSetUp(pg->mg));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_GAMGMC(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "PCGAMGMC options");
  PetscCall(PetscOptionsString("-pc_gamgmc_mg_type", "The type of the inner multigrid method", NULL, pg->mgtype, pg->mgtype, sizeof(pg->mgtype), NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_GAMGMC(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  if (pg->scb && pg->del_scb) PetscCall(pg->del_scb(pg->cbctx));
  PetscCheck(cb, MPI_COMM_WORLD, PETSC_ERR_SUP, "Must pass callback function");
  pg->scb = cb;
  if (ctx) pg->cbctx = ctx;
  if (deleter) pg->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_GAMGMC(PC pc, KSP ksp, Vec b, Vec x)
{
  (void)b;
  (void)x;

  PC_GAMGMC pg = pc->data;

  PetscFunctionBeginUser;
  pg->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGMCGetLevels(PC pc, PetscInt *levels)
{
  PC_GAMGMC pg = pc->data;

  PetscFunctionBegin;
  PetscCall(PCMGGetLevels(pg->mg, levels));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_GAMGMC(PC pc)
{
  PC_GAMGMC pg;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&pg));
  PetscCall(PCCreate(MPI_COMM_WORLD, &pg->mg));
  strcpy(pg->mgtype, PCGAMG);
  pg->As      = NULL;
  pg->cbctx   = NULL;
  pg->scb     = NULL;
  pg->del_scb = NULL;

  pc->data       = pg;
  pc->ops->setup = PCSetUp_GAMGMC;
  /* pc->ops->applyrichardson = PCApplyRichardson_GAMGMC; */
  pc->ops->apply          = PCApply_GAMGMC;
  pc->ops->view           = PCView_GAMGMC;
  pc->ops->destroy        = PCDestroy_GAMGMC;
  pc->ops->setfromoptions = PCSetFromOptions_GAMGMC;
  pc->ops->presolve       = PCPreSolve_GAMGMC;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_GAMGMC));

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMGGetLevels_C", PCGAMGMCGetLevels));
  PetscFunctionReturn(PETSC_SUCCESS);
}
