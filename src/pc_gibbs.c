/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/
/** @file pc_gibbs.c
    @brief A Gibbs sampler wrapped as a PETSc PC
    
    # Options database keys
    - `-pc_gibbs_omega` - the SOR parameter (default is omega = 1)

    # Notes
    This implements a Gibbs sampler wrapped as a PETSc PC. In parallel this uses
    a multicolour Gauss-Seidel implementation to obtain a true parallel Gibbs
    sampler.

    Implemented for PETSc's MATAIJ and MATLRC formats. The latter is used for
    matrices of the form \f$A + B \Sigma^{-1} B^T\f$ which come up in Bayesian
    linear inverse problems with Gaussian priors.

    This is supposed to be used in conjunction with `KSPRICHARDSON`, either
    as a stand-alone sampler or as a random smoother in Multigrid Monte Carlo.
    As a stand-alone sampler, its usage is as follows:

        KSPSetType(ksp, KSPRICHARDSON);
        KSPGetPC(KSP, &pc);
        PCSetType(pc, "gibbs");
        KSPSetUp(ksp);
        ...
        KSPSolve(ksp, b, x); // This performs the sampling    
    
    This PC supports setting a callback which is called for each sample by calling

        PCSetSampleCallback(pc, SampleCallback, &ctx, NULL);
    
    where `ctx` is a user defined context (can also be NULL) that is passed to the
    callback along with the sample.
 */

#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>

typedef struct {
  Mat         A, Asor;
  PetscRandom prand;
  Vec         sqrtdiag;
  PetscReal   omega;
  PetscBool   omega_changed;
  MCSOR       mc;
  MatSORType  type;
  Vec         z;

  PetscBool first_call;

  Mat B;
  Vec w;
  Vec sqrtS;

  PetscErrorCode (*prepare_rhs)(PC, Vec, Vec);

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} PC_Gibbs;

static PetscErrorCode PCDestroy_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&pg->prand));
  PetscCall(VecDestroy(&pg->sqrtdiag));
  PetscCall(MCSORDestroy(&pg->mc));
  PetscCall(VecDestroy(&pg->w));
  PetscCall(VecDestroy(&pg->sqrtS));
  PetscCall(VecDestroy(&pg->z));
  if (pg->del_scb) {
    PetscCall(pg->del_scb(pg->cbctx));
    pg->del_scb = NULL;
  }
  PetscCall(PetscFree(pg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&pg->prand));
  PetscCall(VecDestroy(&pg->sqrtdiag));
  PetscCall(MCSORDestroy(&pg->mc));
  PetscCall(VecDestroy(&pg->w));
  PetscCall(VecDestroy(&pg->sqrtS));
  PetscCall(VecDestroy(&pg->z));
  if (pg->del_scb) {
    PetscCall(pg->del_scb(pg->cbctx));
    pg->del_scb = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareRHS_Default(PC pc, Vec rhsin, Vec rhsout)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecSetRandom(rhsout, pg->prand));
  PetscCall(VecPointwiseMult(rhsout, rhsout, pg->sqrtdiag));
  if (rhsin) PetscCall(VecAXPY(rhsout, 1., rhsin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareRHS_LRC(PC pc, Vec rhsin, Vec rhsout)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PrepareRHS_Default(pc, rhsin, rhsout));
  PetscCall(VecSetRandom(pg->w, pg->prand));
  PetscCall(VecPointwiseMult(pg->w, pg->w, pg->sqrtS));
  PetscCall(MatMultAdd(pg->B, pg->w, rhsout, rhsout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGibbsUpdateSqrtDiag(PC pc)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MCSORSetOmega(pg->mc, pg->omega));
  PetscCall(MatGetDiagonal(pg->Asor, pg->sqrtdiag));
  PetscCall(VecSqrtAbs(pg->sqrtdiag));
  PetscCall(VecScale(pg->sqrtdiag, PetscSqrtReal((2 - pg->omega) / pg->omega)));
  pg->omega_changed = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_Gibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  if (pg->omega_changed) PetscCall(PCGibbsUpdateSqrtDiag(pc));

  for (PetscInt it = 0; it < its; ++it) {
    if (pg->scb) PetscCall(pg->scb(it, y, pg->cbctx));

    if (pg->type == SOR_FORWARD_SWEEP || pg->type == SOR_BACKWARD_SWEEP) {
      PetscCall(pg->prepare_rhs(pc, b, w));
      PetscCall(MCSORApply(pg->mc, w, y));
      /* PetscCall(MatSOR(pg->A, w, 1, SOR_FORWARD_SWEEP, 0, 1, 1, y)); */
    } else {
      PetscCall(MCSORSetSweepType(pg->mc, SOR_FORWARD_SWEEP));
      PetscCall(pg->prepare_rhs(pc, b, w));
      PetscCall(MCSORApply(pg->mc, w, y));
      /* PetscCall(MatSOR(pg->A, w, 1, SOR_FORWARD_SWEEP, 0, 1, 1, y)); /\*  *\/ */

      PetscCall(MCSORSetSweepType(pg->mc, SOR_BACKWARD_SWEEP));
      PetscCall(pg->prepare_rhs(pc, b, w));
      PetscCall(MCSORApply(pg->mc, w, y));
      /* PetscCall(MatSOR(pg->A, w, 1, SOR_BACKWARD_SWEEP, 0, 1, 1, y)); */
    }
  }
  if (pg->scb) PetscCall(pg->scb(its, y, pg->cbctx));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Gibbs(PC pc, PetscOptionItems_ARG PetscOptionsObject)
{
  PC_Gibbs *pg = pc->data;
  PetscBool flag;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Gibbs options");
  PetscCall(PetscOptionsRangeReal("-pc_gibbs_omega", "Gibbs SOR parameter", NULL, pg->omega, &pg->omega, &flag, 0.0, 2.0));
  if (flag) pg->omega_changed = PETSC_TRUE;

  flag = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-pc_gibbs_forward", "Gibbs forward sweep", NULL, pg->type == SOR_FORWARD_SWEEP, &flag, NULL));
  if (flag) pg->type = SOR_FORWARD_SWEEP;
  flag = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-pc_gibbs_backward", "Gibbs backward sweep", NULL, pg->type == SOR_BACKWARD_SWEEP, &flag, NULL));
  if (flag) pg->type = SOR_BACKWARD_SWEEP;
  flag = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-pc_gibbs_symmetric", "Gibbs symmetric sweep", NULL, pg->type == SOR_SYMMETRIC_SWEEP, &flag, NULL));
  if (flag) pg->type = SOR_SYMMETRIC_SWEEP;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Gibbs(PC pc)
{
  PC_Gibbs *pg = pc->data;
  MatType   type;
  Mat       P = pc->pmat;

  PetscFunctionBeginUser;
  if (pc->setupcalled) {
    PetscCall(PetscRandomDestroy(&pg->prand));
    PetscCall(VecDestroy(&pg->sqrtS));
    PetscCall(VecDestroy(&pg->sqrtdiag));
    PetscCall(VecDestroy(&pg->w));
    PetscCall(VecDestroy(&pg->z));
    PetscCall(MCSORDestroy(&pg->mc));
  }
  PetscCall(MCSORCreate(P, &pg->mc));
  PetscCall(MCSORSetSweepType(pg->mc, pg->type));
  PetscCall(MCSORSetUp(pg->mc));
  PetscCall(MatGetType(P, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    pg->A    = P;
    pg->Asor = pg->A;
  } else if (strcmp(type, MATMPIAIJ) == 0) {
    pg->A    = P;
    pg->Asor = pg->A;
  } else if (strcmp(type, MATLRC) == 0) {
    Vec S;

    pg->A = P;
    PetscCall(MatLRCGetMats(pg->A, &pg->Asor, &pg->B, &S, NULL));
    PetscCall(VecDuplicate(S, &pg->sqrtS));
    PetscCall(VecCopy(S, pg->sqrtS));
    PetscCall(VecSqrtAbs(pg->sqrtS));
    PetscCall(VecDuplicate(S, &pg->w));
    pg->prepare_rhs = PrepareRHS_LRC;
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
  }
  PetscCall(MatCreateVecs(pg->A, &pg->sqrtdiag, &pg->z));
  pg->omega_changed = PETSC_TRUE;
  PetscCall(ParMGMCGetPetscRandom(&pg->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Gibbs(PC pc, PetscViewer viewer)
{
  PC_Gibbs *pg = pc->data;
  PetscInt  ncolors;

  PetscFunctionBeginUser;
  PetscCall(MCSORGetNumColors(pg->mc, &ncolors));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of colours: %" PetscInt_FMT "\n", ncolors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Sets the Gibbs-SOR parameter. Default is omega = 1.
 */
PetscErrorCode PCGibbsSetOmega(PC pc, PetscReal omega)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  pg->omega         = omega;
  pg->omega_changed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGibbsSetSweepType(PC pc, MatSORType type)
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MCSORSetSweepType(pg->mc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_Gibbs(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_Gibbs *pg = pc->data;

  PetscFunctionBeginUser;
  if (pg->del_scb) {
    PetscCall(pg->del_scb(pg->cbctx));
    pg->del_scb = NULL;
  }
  pg->scb     = cb;
  pg->cbctx   = ctx;
  pg->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Gibbs(PC pc)
{
  PC_Gibbs *gibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&gibbs));
  gibbs->omega       = 1;
  gibbs->prepare_rhs = PrepareRHS_Default;
  gibbs->type        = SOR_FORWARD_SWEEP;
  gibbs->cbctx       = NULL;
  gibbs->scb         = NULL;
  gibbs->del_scb     = NULL;

  pc->data                 = gibbs;
  pc->ops->setup           = PCSetUp_Gibbs;
  pc->ops->destroy         = PCDestroy_Gibbs;
  pc->ops->applyrichardson = PCApplyRichardson_Gibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_Gibbs;
  pc->ops->reset           = PCReset_Gibbs;
  pc->ops->view            = PCView_Gibbs;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Gibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
