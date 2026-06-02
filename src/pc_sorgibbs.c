/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_sorgibbs.h"
#include "parmgmc/pc/pc_parsor.h"
#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stddef.h>
#include <string.h>

typedef struct {
  Vec         sqrtdiag;
  Vec         work;
  PetscRandom prand;
  MatSORType  type;

  /* For parallel SOR on MPIAIJ matrices */
  PC        parsor_pc;
  PetscBool use_parsor;
  PetscInt  sample_index;

  /* MATLRC support: when pc->pmat is A_post = A + B Sigma^{-1} B^T we run
     the SOR sweep on the base AIJ `Asor = A` and apply a Woodbury
     post-correction y -= Bb * (B^T y) after every sweep.  Asor / B are
     borrowed from the LRC matrix; Bb, sqrtS, wk, zn are owned. */
  PetscBool is_lrc;
  Mat       Asor;
  Mat       B, Bb;
  Vec       sqrtS;
  Vec       wk;
  Vec       zn;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_SORGibbs;

/* Apply one deterministic SOR sweep using the same iteration operator the
   sampler will use.  Called by MCSORBuildLRCCorrection column-by-column on
   B (with y zeroed on entry, so this returns M_A^{-1} times each column).  */
static PetscErrorCode SORGibbsDetSOR(void *ctx, Vec b, Vec y)
{
  PC_SORGibbs sorgibbs = (PC_SORGibbs)ctx;

  PetscFunctionBeginUser;
  if (sorgibbs->use_parsor) {
    PetscCall(PCPARSORApplySOR(sorgibbs->parsor_pc, b, 1, PETSC_TRUE, y));
  } else {
    PetscCall(MatSOR(sorgibbs->Asor, b, 1., sorgibbs->type, 0., 1., 1., y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSORGibbsNotifySample(PC pc, Vec y)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  if (sorgibbs->scb) PetscCall(sorgibbs->scb(sorgibbs->sample_index++, y, sorgibbs->cbctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSORGibbsSample(PC pc, Vec b, Vec y, Vec w)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecSetRandomStandardNormal(w, sorgibbs->prand));
  PetscCall(VecPointwiseMult(w, w, sorgibbs->sqrtdiag));
  PetscCall(VecAXPY(w, 1., b));
  /* MATLRC: add the noise B * sqrt(Sigma^{-1}) * eta to the RHS so that
     the chain samples from N(*, A_post^{-1}) instead of N(*, A^{-1}). */
  if (sorgibbs->is_lrc) {
    PetscCall(VecSetRandomStandardNormal(sorgibbs->wk, sorgibbs->prand));
    PetscCall(VecPointwiseMult(sorgibbs->wk, sorgibbs->wk, sorgibbs->sqrtS));
    PetscCall(MatMultAdd(sorgibbs->B, sorgibbs->wk, w, w));
  }
  if (sorgibbs->use_parsor) {
    PetscCall(PCPARSORApplySOR(sorgibbs->parsor_pc, w, 1, PETSC_FALSE, y));
  } else {
    PetscCall(MatSOR(sorgibbs->Asor, w, 1., sorgibbs->type, 0., 1., 1., y));
  }
  /* MATLRC: Sherman-Morrison-Woodbury post-correction y -= Bb * (B^T y). */
  if (sorgibbs->is_lrc) {
    PetscCall(MatMultTranspose(sorgibbs->B, y, sorgibbs->wk));
    PetscCall(MatMult(sorgibbs->Bb, sorgibbs->wk, sorgibbs->zn));
    PetscCall(VecAXPY(y, -1., sorgibbs->zn));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_SORGibbs(PC pc, Vec b, Vec y)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscCall(PCSORGibbsSample(pc, b, y, sorgibbs->work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_SORGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  sorgibbs->sample_index = 0;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(PCSORGibbsSample(pc, b, y, w));
    PetscCall(PCSORGibbsNotifySample(pc, y));
  }

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
  PetscCall(VecDestroy(&sorgibbs->work));
  PetscCall(PCDestroy(&sorgibbs->parsor_pc));
  PetscCall(MatDestroy(&sorgibbs->Bb));
  PetscCall(VecDestroy(&sorgibbs->sqrtS));
  PetscCall(VecDestroy(&sorgibbs->wk));
  PetscCall(VecDestroy(&sorgibbs->zn));
  sorgibbs->use_parsor = PETSC_FALSE;
  sorgibbs->is_lrc     = PETSC_FALSE;
  sorgibbs->B          = NULL;
  sorgibbs->Asor       = NULL;
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
  PetscCall(VecDestroy(&sorgibbs->work));
  PetscCall(PCDestroy(&sorgibbs->parsor_pc));
  PetscCall(MatDestroy(&sorgibbs->Bb));
  PetscCall(VecDestroy(&sorgibbs->sqrtS));
  PetscCall(VecDestroy(&sorgibbs->wk));
  PetscCall(VecDestroy(&sorgibbs->zn));
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
  MatType     mtype;
  PetscBool   is_mpiaij, is_lrc;
  Vec         S = NULL;

  PetscFunctionBeginUser;
  /* Tear down per-setup state so PCSetUp can be re-run on a new operator. */
  PetscCall(VecDestroy(&sorgibbs->sqrtdiag));
  PetscCall(VecDestroy(&sorgibbs->work));
  PetscCall(MatDestroy(&sorgibbs->Bb));
  PetscCall(VecDestroy(&sorgibbs->sqrtS));
  PetscCall(VecDestroy(&sorgibbs->wk));
  PetscCall(VecDestroy(&sorgibbs->zn));
  sorgibbs->B    = NULL;
  sorgibbs->Asor = NULL;

  /* Identify the operator type.  For MATLRC we run the sweep on the base
     AIJ matrix Asor and add a rank-k correction; otherwise Asor is just
     pc->pmat. */
  PetscCall(MatGetType(pc->pmat, &mtype));
  PetscCall(PetscStrcmp(mtype, MATLRC, &is_lrc));
  if (is_lrc) {
    sorgibbs->is_lrc = PETSC_TRUE;
    PetscCall(MatLRCGetMats(pc->pmat, &sorgibbs->Asor, &sorgibbs->B, &S, NULL));
    /* MATLRC stores S as a sequential (replicated) vec.  We need parallel
       size-k workspaces matching B's column layout so MatMultTranspose
       (used in the post-correction below) sees consistent local dims. */
    PetscCall(MatCreateVecs(sorgibbs->B, &sorgibbs->wk, NULL));
    PetscCall(VecDuplicate(sorgibbs->wk, &sorgibbs->sqrtS));
    {
      const PetscScalar *Sarr;
      PetscScalar       *sqrtSarr;
      PetscInt           istart, iend;

      PetscCall(VecGetOwnershipRange(sorgibbs->sqrtS, &istart, &iend));
      PetscCall(VecGetArrayRead(S, &Sarr));
      PetscCall(VecGetArray(sorgibbs->sqrtS, &sqrtSarr));
      for (PetscInt i = istart; i < iend; ++i) sqrtSarr[i - istart] = Sarr[i];
      PetscCall(VecRestoreArrayRead(S, &Sarr));
      PetscCall(VecRestoreArray(sorgibbs->sqrtS, &sqrtSarr));
    }
    PetscCall(VecSqrtAbs(sorgibbs->sqrtS));
  } else {
    sorgibbs->is_lrc = PETSC_FALSE;
    sorgibbs->Asor   = pc->pmat;
  }

  /* sqrtdiag / work come from the base matrix Asor.  Using pc->pmat here
     fails for MATLRC (no MatGetDiagonal / MatCreateVecs side-effects we
     want), and the sampling math wants D of the base A anyway. */
  PetscCall(MatCreateVecs(sorgibbs->Asor, &sorgibbs->sqrtdiag, NULL));
  PetscCall(VecDuplicate(sorgibbs->sqrtdiag, &sorgibbs->work));
  PetscCall(MatGetDiagonal(sorgibbs->Asor, sorgibbs->sqrtdiag));
  PetscCall(VecSqrtAbs(sorgibbs->sqrtdiag));
  if (!sorgibbs->prand) PetscCall(ParMGMCGetPetscRandom(&sorgibbs->prand));

  /* PCPARSOR path: true parallel Gauss-Seidel for MPIAIJ + forward sweep. */
  PetscCall(MatGetType(sorgibbs->Asor, &mtype));
  PetscCall(PetscStrcmp(mtype, MATMPIAIJ, &is_mpiaij));
  if (is_mpiaij && sorgibbs->type == SOR_FORWARD_SWEEP) {
    sorgibbs->use_parsor = PETSC_TRUE;
    if (!sorgibbs->parsor_pc) {
      PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &sorgibbs->parsor_pc));
      PetscCall(PCSetType(sorgibbs->parsor_pc, PCPARSOR));
    }
    PetscCall(PCSetOperators(sorgibbs->parsor_pc, sorgibbs->Asor, sorgibbs->Asor));
    PetscCall(PCSetUp(sorgibbs->parsor_pc));
  } else {
    sorgibbs->use_parsor = PETSC_FALSE;
  }

  /* Build the Woodbury correction matrix Bb = M_A^{-1} B (S^{-1} + B^T M_A^{-1} B)^{-1}.
     SORGibbsDetSOR supplies M_A^{-1} via MatSOR / PCPARSOR — whichever the
     actual sampling sweep will use, so the iteration matrix matches. */
  if (sorgibbs->is_lrc) {
    PetscCall(MCSORBuildLRCCorrection(SORGibbsDetSOR, sorgibbs, sorgibbs->Asor, sorgibbs->B, S, &sorgibbs->Bb));
    PetscCall(MatCreateVecs(sorgibbs->Bb, NULL, &sorgibbs->zn));
  }
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

static PetscErrorCode PCView_SORGibbs(PC pc, PetscViewer viewer)
{
  PC_SORGibbs sorgibbs = pc->data;

  PetscFunctionBeginUser;
  PetscAssert(sorgibbs->type == SOR_FORWARD_SWEEP || sorgibbs->type == SOR_LOCAL_FORWARD_SWEEP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Forgot to add sweep in PCView_SORGibbs");
  if (sorgibbs->type == SOR_FORWARD_SWEEP) PetscCall(PetscViewerASCIIPrintf(viewer, "Sweep type: Forward\n"));
  if (sorgibbs->type == SOR_LOCAL_FORWARD_SWEEP) PetscCall(PetscViewerASCIIPrintf(viewer, "Sweep type: Local forward (a.k.a Hogwild sampler)\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_SORGibbs(PC pc)
{
  PC_SORGibbs sorgibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sorgibbs));
  pc->data       = sorgibbs;
  sorgibbs->type = SOR_FORWARD_SWEEP;

  pc->ops->apply           = PCApply_SORGibbs;
  pc->ops->applyrichardson = PCApplyRichardson_SORGibbs;
  pc->ops->destroy         = PCDestroy_SORGibbs;
  pc->ops->reset           = PCReset_SORGibbs;
  pc->ops->setup           = PCSetUp_SORGibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_SORGibbs;
  pc->ops->view            = PCView_SORGibbs;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_SORGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
