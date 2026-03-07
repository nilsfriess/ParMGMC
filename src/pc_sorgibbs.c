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

  PC_SORGibbs hw = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt it = 0; it < its; ++it) {
    if (hw->scb) PetscCall(hw->scb(it, y, hw->cbctx));

    PetscCall(VecSetRandom(w, hw->prand));
    PetscCall(VecPointwiseMult(w, w, hw->sqrtdiag));
    PetscCall(VecAXPY(w, 1., b));
    PetscCall(MatSOR(pc->pmat, w, 1., SOR_FORWARD_SWEEP, 0., 1., 1., y));
  }
  if (hw->scb) PetscCall(hw->scb(its, y, hw->cbctx));

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_SORGibbs(PC pc)
{
  PC_SORGibbs hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&hw->prand));
  PetscCall(VecDestroy(&hw->sqrtdiag));
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_SORGibbs(PC pc)
{
  PC_SORGibbs hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&hw->prand));
  PetscCall(VecDestroy(&hw->sqrtdiag));
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  PetscCall(PetscFree(hw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_SORGibbs(PC pc)
{
  PC_SORGibbs hw = pc->data;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(pc->pmat, &hw->sqrtdiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, hw->sqrtdiag));
  PetscCall(VecSqrtAbs(hw->sqrtdiag));
  if (!hw->prand) PetscCall(ParMGMCGetPetscRandom(&hw->prand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_SORGibbs(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_SORGibbs hw = pc->data;

  PetscFunctionBeginUser;
  if (hw->del_scb) {
    PetscCall(hw->del_scb(hw->cbctx));
    hw->del_scb = NULL;
  }
  hw->scb     = cb;
  hw->cbctx   = ctx;
  hw->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_SORGibbs(PC pc)
{
  PC_SORGibbs hw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&hw));
  pc->data = hw;

  pc->ops->applyrichardson = PCApplyRichardson_SORGibbs;
  pc->ops->destroy         = PCDestroy_SORGibbs;
  pc->ops->reset           = PCReset_SORGibbs;
  pc->ops->setup           = PCSetUp_SORGibbs;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_SORGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
