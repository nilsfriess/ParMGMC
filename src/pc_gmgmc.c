/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_gmgmc.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscstring.h>
#include <petscsys.h>

typedef struct _PC_GMGMC {
  PC   mg;
  Mat *As; // The actual matrices used (in case of A+LR this differs from the matrices used to setup the multigrid hierarchy).
} *PC_GMGMC;

static PetscErrorCode PCDestroy_GMGMC(PC pc)
{
  SampleCallbackCtx cbctx = pc->user;
  PC_GMGMC          pg    = pc->data;
  PetscInt          levels;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(cbctx));
  if (pg->As) {
    PetscCall(PCMGGetLevels(pg->mg, &levels));
    for (PetscInt l = 0; l < levels - 1; ++l) {
      Mat B;
      PetscCall(MatLRCGetMats(pg->As[l], NULL, &B, NULL, NULL));
      PetscCall(MatDestroy(&B));
      PetscCall(MatDestroy(&(pg->As[l])));
    }
    PetscCall(PetscFree(pg->As));
  }
  PetscCall(PCDestroy(&pg->mg));
  PetscCall(PetscFree(pg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCGMGMCSetLevels(PC pc, PetscInt levels)
{
  PC_GMGMC pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCMGSetLevels(pg->mg, levels, NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_GMGMC(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;
  (void)w;

  PC_GMGMC          pg    = pc->data;
  SampleCallbackCtx cbctx = pc->user;

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < its; ++i) {
    PetscCall(PCApplyRichardson(pg->mg, b, y, w, 0, 0, 0, 1, PETSC_TRUE, outits, reason));
    if (cbctx->cb) PetscCall(cbctx->cb(i, y, cbctx->ctx));
  }
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_GMGMC(PC pc, PetscViewer v)
{
  PC_GMGMC pg = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCView(pg->mg, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_GMGMC(PC pc)
{
  PC_GMGMC  pg = pc->data;
  MatType   type;
  Mat       P;
  PetscInt  levels;
  PetscBool islrc;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &islrc));
  if (islrc) PetscCall(MatLRCGetMats(pc->pmat, &P, NULL, NULL, NULL));
  else P = pc->pmat;

  PetscCall(PCSetOperators(pg->mg, P, P));
  PetscCall(PCSetDM(pg->mg, pc->dm));
  PetscCall(PCMGSetGalerkin(pg->mg, PC_MG_GALERKIN_BOTH));
  PetscCall(PCSetFromOptions(pg->mg));
  PetscCall(PCSetUp(pg->mg));

  if (islrc) {
    PetscCall(PCMGGetLevels(pg->mg, &levels));
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
    }

    for (PetscInt l = levels - 1; l >= 0; --l) {
      KSP ksps;
      PC  pcs;

      PetscCall(PCMGGetSmoother(pg->mg, l, &ksps));
      PetscCall(KSPGetPC(ksps, &pcs));
      PetscCall(PCSetOperators(pcs, pg->As[l], pg->As[l]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_GMGMC(PC pc)
{
  PC_GMGMC          pg;
  SampleCallbackCtx cbctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&pg));
  PetscCall(PCCreate(MPI_COMM_WORLD, &pg->mg));
  PetscCall(PCSetOptionsPrefix(pg->mg, "gmgmc_"));
  PetscCall(PCSetType(pg->mg, PCMG));
  pg->As = NULL;

  PetscCall(PetscNew(&cbctx));
  pc->user = cbctx;

  pc->data                 = pg;
  pc->ops->setup           = PCSetUp_GMGMC;
  pc->ops->applyrichardson = PCApplyRichardson_GMGMC;
  pc->ops->view            = PCView_GMGMC;
  pc->ops->destroy         = PCDestroy_GMGMC;
  PetscFunctionReturn(PETSC_SUCCESS);
}
