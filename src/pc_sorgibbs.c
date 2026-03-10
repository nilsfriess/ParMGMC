/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_sorgibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stddef.h>

typedef struct {
  Vec         sqrtdiag;
  PetscRandom prand;
  MatSORType  type;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_SORGibbs;

static PetscErrorCode PCApplyRichardson_SORGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt it = 0; it < its; ++it) {
    if (sorgibbs->scb) PetscCall(sorgibbs->scb(it, y, sorgibbs->cbctx));

    PetscCall(VecSetRandom(w, sorgibbs->prand));
    PetscCall(VecPointwiseMult(w, w, sorgibbs->sqrtdiag));
    PetscCall(VecAXPY(w, 1., b));
    PetscCall(MatSOR(pc->pmat, w, 1., sorgibbs->type, 0., 1., 1., y));
  }
  if (sorgibbs->scb) PetscCall(sorgibbs->scb(its, y, sorgibbs->cbctx));

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_SORGibbs(PC pc)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&sorgibbs->prand));
  PetscCall(VecDestroy(&sorgibbs->sqrtdiag));
  if (sorgibbs->del_scb) {
    PetscCall(sorgibbs->del_scb(sorgibbs->cbctx));
    sorgibbs->del_scb = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_SORGibbs(PC pc)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&sorgibbs->prand));
  PetscCall(VecDestroy(&sorgibbs->sqrtdiag));
  if (sorgibbs->del_scb) {
    PetscCall(sorgibbs->del_scb(sorgibbs->cbctx));
    sorgibbs->del_scb = NULL;
  }
  PetscCall(PetscFree(sorgibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_SORGibbs(PC pc)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(pc->pmat, &sorgibbs->sqrtdiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, sorgibbs->sqrtdiag));
  PetscCall(VecSqrtAbs(sorgibbs->sqrtdiag));
  if (!sorgibbs->prand) PetscCall(ParMGMCGetPetscRandom(&sorgibbs->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_SORGibbs(PC pc, PetscOptionItems_ARG PetscOptionsObject)
{
  PC_SORGibbs sorgibbs = pc->data;
  PetscBool   flag     = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SOR Gibbs options");
  PetscCall(PetscOptionsBool("-pc_sorgibbs_forward", "SOR Gibbs forward sweep", NULL, sorgibbs->type == SOR_FORWARD_SWEEP, &flag, NULL));
  if (flag) sorgibbs->type = SOR_FORWARD_SWEEP;
  flag = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-pc_sorgibbs_local_forward", "SOR Gibbs local forward sweep (Hogwild sampler)", NULL, sorgibbs->type == SOR_LOCAL_FORWARD_SWEEP, &flag, NULL));
  if (flag) sorgibbs->type = SOR_LOCAL_FORWARD_SWEEP;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_SORGibbs(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  if (sorgibbs->del_scb) {
    PetscCall(sorgibbs->del_scb(sorgibbs->cbctx));
    sorgibbs->del_scb = NULL;
  }
  sorgibbs->scb     = cb;
  sorgibbs->cbctx   = ctx;
  sorgibbs->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_SORGibbs(PC pc)
{
  PC_SORGibbs sorgibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sorgibbs));
  pc->data       = sorgibbs;
  sorgibbs->type = SOR_FORWARD_SWEEP;

  pc->ops->applyrichardson = PCApplyRichardson_SORGibbs;
  pc->ops->destroy         = PCDestroy_SORGibbs;
  pc->ops->reset           = PCReset_SORGibbs;
  pc->ops->setup           = PCSetUp_SORGibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_SORGibbs;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_SORGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
