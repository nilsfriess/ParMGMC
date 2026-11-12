/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/** @file pc_patchgibbs.c
    @brief Overlapping block (patch) Gibbs sampler, i.e. the multiplicative overlapping Schwarz
    sampler, for small patches.

    The patches (sets of DOF indices) are given by the user via PCPatchGibbsSetPatches().  One sweep
    visits the patches in order and, for patch p with DOFs I_p, local matrix A_p = A(I_p, I_p) =
    L_p L_p^T and the current state x, sets

        x(I_p) <- x(I_p) + L_p^{-T} (L_p^{-1} r_p + z_p),   r_p = (b - A x)(I_p),  z_p ~ N(0, I),

    i.e. it draws x(I_p) from its conditional distribution given the other entries.  Without the
    noise z_p (-pc_patchgibbs_deterministic) this is the multiplicative overlapping Schwarz
    smoother with exact patch solves, the deterministic counterpart of the sampler.

    This does the same as PCASM (multiplicative, one subdomain per patch) with a PCCHOLSAMPLER on
    each subdomain, but is meant for many small patches: the dense Cholesky factors of all patch
    matrices are computed once, and a sweep works directly on the CSR arrays of A without any
    per-patch PETSc objects, scatters or residual updates.  Sequential (MATSEQAIJ) only.

    # Options database keys
    - `-pc_patchgibbs_sweep <forward,backward,symmetric>` - order in which the patches are visited
      (backward = reversed order, symmetric = forward followed by backward); default forward
    - `-pc_patchgibbs_deterministic` - no noise (deterministic smoother)

    PCApplyRichardson() performs sweeps in place on the given state; PCApply() performs one sweep
    starting from zero, i.e. computes M^{-1} r (plus noise), so the PC can also be used as a smoother
    inside KSPRICHARDSON/PCMG.
*/

#include "parmgmc/pc/pc_patchgibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscblaslapack.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>

typedef enum {
  PATCHGIBBS_FORWARD,
  PATCHGIBBS_BACKWARD,
  PATCHGIBBS_SYMMETRIC
} PatchGibbsSweepType;

static const char *const PatchGibbsSweepTypes[] = {"forward", "backward", "symmetric"};

static PetscLogEvent PATCHGIBBS_SWEEP = 0, PATCHGIBBS_SETUP = 0;

typedef struct {
  /* Patches (set by the user): DOFs of patch p are idx[offsets[p]], ..., idx[offsets[p+1]-1] */
  PetscInt  npatches;
  PetscInt *offsets, *idx;

  /* Set up in PCSetUp */
  PetscInt64      *foff; /* offset of the factor of patch p in L */
  PetscScalar     *L;    /* dense lower Cholesky factors, column major */
  PetscScalar     *work;
  PetscInt         maxsize;
  Mat              A; /* pmat whose row structure we hold (reference) */
  PetscInt         nrows;
  const PetscInt  *ai, *aj;
  PetscBool        rowij;
  Vec              z; /* noise for one sweep, one entry per patch DOF */
  PetscRandom      prand;

  PatchGibbsSweepType sweep;
  PetscBool           deterministic;

  PetscInt sample_index;
  void    *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_PatchGibbs;

static PetscErrorCode PCPatchGibbsResetSetup(PC_PatchGibbs pg)
{
  PetscFunctionBegin;
  if (pg->rowij) {
    PetscBool done;

    PetscCall(MatRestoreRowIJ(pg->A, 0, PETSC_FALSE, PETSC_FALSE, &pg->nrows, &pg->ai, &pg->aj, &done));
    pg->rowij = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&pg->A));
  PetscCall(PetscFree(pg->foff));
  PetscCall(PetscFree(pg->L));
  PetscCall(PetscFree(pg->work));
  PetscCall(VecDestroy(&pg->z));
  pg->maxsize = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_PatchGibbs(PC pc)
{
  PC_PatchGibbs pg = pc->data;

  PetscFunctionBegin;
  PetscCall(PCPatchGibbsResetSetup(pg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_PatchGibbs(PC pc)
{
  PC_PatchGibbs pg = pc->data;

  PetscFunctionBegin;
  PetscCall(PCPatchGibbsResetSetup(pg));
  PetscCall(PetscFree(pg->offsets));
  PetscCall(PetscFree(pg->idx));
  PetscCall(PetscRandomDestroy(&pg->prand));
  if (pg->del_scb) PetscCall(pg->del_scb(pg->cbctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPatchGibbsSetPatches_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPatchGibbsSetPatches_PatchGibbs(PC pc, PetscInt npatches, const PetscInt offsets[], const PetscInt idx[])
{
  PC_PatchGibbs pg = pc->data;

  PetscFunctionBegin;
  PetscCheck(npatches >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of patches must be non-negative");
  PetscCheck(offsets[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "offsets[0] must be 0");
  for (PetscInt p = 0; p < npatches; ++p) PetscCheck(offsets[p + 1] >= offsets[p], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "offsets must be non-decreasing");
  PetscCall(PetscFree(pg->offsets));
  PetscCall(PetscFree(pg->idx));
  PetscCall(PetscMalloc1(npatches + 1, &pg->offsets));
  PetscCall(PetscMalloc1(offsets[npatches], &pg->idx));
  PetscCall(PetscArraycpy(pg->offsets, offsets, npatches + 1));
  PetscCall(PetscArraycpy(pg->idx, idx, offsets[npatches]));
  pg->npatches = npatches;
  pc->setupcalled = PETSC_FALSE; /* the factors have to be recomputed */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCPatchGibbsSetPatches - Sets the patches of a PCPATCHGIBBS sampler.

  Input Parameters:
+ pc       - the PC
. npatches - number of patches
. offsets  - array of length npatches + 1; the DOFs of patch p are idx[offsets[p]], ..., idx[offsets[p+1]-1]
- idx      - DOF indices (row indices of the operator); the arrays are copied

  Notes:
  The patches are visited in the given order (reversed for a backward sweep).
@*/
PetscErrorCode PCPatchGibbsSetPatches(PC pc, PetscInt npatches, const PetscInt offsets[], const PetscInt idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCPatchGibbsSetPatches_C", (PC, PetscInt, const PetscInt[], const PetscInt[]), (pc, npatches, offsets, idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_PatchGibbs(PC pc)
{
  PC_PatchGibbs      pg = pc->data;
  PetscMPIInt        size;
  PetscBool          isseqaij, done;
  const PetscScalar *aa;
  PetscInt          *pos;
  MatInfo            info;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCPATCHGIBBS is sequential only");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat, MATSEQAIJ, &isseqaij));
  PetscCheck(isseqaij, PETSC_COMM_SELF, PETSC_ERR_SUP, "PCPATCHGIBBS requires a MATSEQAIJ matrix");
  PetscCheck(pg->offsets, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PCPatchGibbsSetPatches() before PCSetUp()");
  PetscCall(PetscLogEventBegin(PATCHGIBBS_SETUP, pc, 0, 0, 0));

  PetscCall(PCPatchGibbsResetSetup(pg));
  PetscCall(PetscObjectReference((PetscObject)pc->pmat));
  pg->A = pc->pmat;
  /* No inode compression: ai/aj must be the plain CSR structure matching the value array */
  PetscCall(MatGetRowIJ(pg->A, 0, PETSC_FALSE, PETSC_FALSE, &pg->nrows, &pg->ai, &pg->aj, &done));
  PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not get the CSR structure of the matrix");
  pg->rowij = PETSC_TRUE;
  PetscCall(MatGetInfo(pg->A, MAT_LOCAL, &info));
  PetscCheck((PetscInt)info.nz_used == pg->ai[pg->nrows], PETSC_COMM_SELF, PETSC_ERR_PLIB, "CSR structure (%" PetscInt_FMT " entries) does not match the matrix (%g nonzeros)", pg->ai[pg->nrows], info.nz_used);

  PetscCall(PetscMalloc1(pg->npatches + 1, &pg->foff));
  pg->foff[0] = 0;
  for (PetscInt p = 0; p < pg->npatches; ++p) {
    PetscInt n = pg->offsets[p + 1] - pg->offsets[p];

    pg->maxsize     = PetscMax(pg->maxsize, n);
    pg->foff[p + 1] = pg->foff[p] + (PetscInt64)n * n;
  }
  PetscCall(PetscMalloc1(pg->foff[pg->npatches], &pg->L));
  PetscCall(PetscMalloc1(pg->maxsize, &pg->work));
  PetscCall(PetscMalloc1(pg->nrows, &pos));
  for (PetscInt i = 0; i < pg->nrows; ++i) pos[i] = -1;

  PetscCall(MatSeqAIJGetArrayRead(pg->A, &aa));
  for (PetscInt p = 0; p < pg->npatches; ++p) {
    const PetscInt *dofs  = pg->idx + pg->offsets[p];
    PetscInt        n  = pg->offsets[p + 1] - pg->offsets[p];
    PetscScalar    *Lp = pg->L + pg->foff[p];
    PetscBLASInt    bn, linfo;

    if (n == 0) continue;
    for (PetscInt i = 0; i < n; ++i) {
      PetscCheck(dofs[i] >= 0 && dofs[i] < pg->nrows, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Patch %" PetscInt_FMT " contains index %" PetscInt_FMT " outside [0, %" PetscInt_FMT ")", p, dofs[i], pg->nrows);
      PetscCheck(pos[dofs[i]] < 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Patch %" PetscInt_FMT " contains index %" PetscInt_FMT " twice", p, dofs[i]);
      pos[dofs[i]] = i;
    }
    PetscCall(PetscArrayzero(Lp, n * n));
    for (PetscInt i = 0; i < n; ++i) {
      for (PetscInt k = pg->ai[dofs[i]]; k < pg->ai[dofs[i] + 1]; ++k) {
        PetscInt j = pos[pg->aj[k]];
        if (j >= 0) Lp[j * n + i] = aa[k]; /* A_p(i, j), column major */
      }
    }
    for (PetscInt i = 0; i < n; ++i) pos[dofs[i]] = -1;

    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("L", &bn, Lp, &bn, &linfo));
    PetscCheck(linfo == 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Cholesky factorisation of patch %" PetscInt_FMT " (size %" PetscInt_FMT ") failed (info %d); is the matrix SPD?", p, n, (int)linfo);
  }
  PetscCall(MatSeqAIJRestoreArrayRead(pg->A, &aa));
  PetscCall(PetscFree(pos));

  if (!pg->deterministic) {
    if (!pg->prand) PetscCall(ParMGMCGetPetscRandom(&pg->prand));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, pg->offsets[pg->npatches], &pg->z));
  }
  PetscCall(PetscLogEventEnd(PATCHGIBBS_SETUP, pc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One sweep over all patches (in reverse order if `reverse`), in place on x, right-hand side b.
   noise == NULL: deterministic. */
static PetscErrorCode PCPatchGibbsSweep(PC_PatchGibbs pg, const PetscScalar *restrict aa, const PetscScalar *restrict b, PetscScalar *restrict x, PetscBool reverse, const PetscScalar *restrict noise)
{
  const PetscInt *restrict ai = pg->ai, *restrict aj = pg->aj;
  PetscScalar *restrict    w  = pg->work;

  PetscFunctionBegin;
  for (PetscInt q = 0; q < pg->npatches; ++q) {
    const PetscInt              p  = reverse ? pg->npatches - 1 - q : q;
    const PetscInt              n  = pg->offsets[p + 1] - pg->offsets[p];
    const PetscInt *restrict    dofs  = pg->idx + pg->offsets[p];
    const PetscScalar *restrict Lp = pg->L + pg->foff[p];

    /* w = r_p = (b - A x)(I_p) */
    for (PetscInt i = 0; i < n; ++i) {
      const PetscInt row = dofs[i];
      PetscScalar    s   = b[row];

      for (PetscInt k = ai[row]; k < ai[row + 1]; ++k) s -= aa[k] * x[aj[k]];
      w[i] = s;
    }
    /* w = L^{-1} w */
    for (PetscInt i = 0; i < n; ++i) {
      PetscScalar s = w[i];

      for (PetscInt j = 0; j < i; ++j) s -= Lp[j * n + i] * w[j];
      w[i] = s / Lp[i * n + i];
    }
    if (noise) {
      const PetscScalar *restrict zp = noise + pg->offsets[p];

      for (PetscInt i = 0; i < n; ++i) w[i] += zp[i];
    }
    /* w = L^{-T} w */
    for (PetscInt i = n - 1; i >= 0; --i) {
      PetscScalar s = w[i];

      for (PetscInt j = i + 1; j < n; ++j) s -= Lp[i * n + j] * w[j];
      w[i] = s / Lp[i * n + i];
    }
    for (PetscInt i = 0; i < n; ++i) x[dofs[i]] += w[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One application (one sweep, or forward + backward for symmetric) in place on x */
static PetscErrorCode PCPatchGibbsApplyOnce(PC_PatchGibbs pg, const PetscScalar *aa, const PetscScalar *b, PetscScalar *x)
{
  const PetscScalar *noise = NULL;
  PetscInt           nsweeps = pg->sweep == PATCHGIBBS_SYMMETRIC ? 2 : 1;

  PetscFunctionBegin;
  for (PetscInt s = 0; s < nsweeps; ++s) {
    PetscBool reverse = (pg->sweep == PATCHGIBBS_BACKWARD || (pg->sweep == PATCHGIBBS_SYMMETRIC && s == 1)) ? PETSC_TRUE : PETSC_FALSE;

    if (!pg->deterministic) {
      PetscCall(VecSetRandomStandardNormal(pg->z, pg->prand));
      PetscCall(VecGetArrayRead(pg->z, &noise));
    }
    PetscCall(PetscLogEventBegin(PATCHGIBBS_SWEEP, 0, 0, 0, 0));
    PetscCall(PCPatchGibbsSweep(pg, aa, b, x, reverse, noise));
    PetscCall(PetscLogEventEnd(PATCHGIBBS_SWEEP, 0, 0, 0, 0));
    if (!pg->deterministic) PetscCall(VecRestoreArrayRead(pg->z, &noise));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PatchGibbs(PC pc, Vec r, Vec y)
{
  PC_PatchGibbs      pg = pc->data;
  const PetscScalar *aa, *rr;
  PetscScalar       *yy;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(y));
  PetscCall(MatSeqAIJGetArrayRead(pg->A, &aa));
  PetscCall(VecGetArrayRead(r, &rr));
  PetscCall(VecGetArray(y, &yy));
  PetscCall(PCPatchGibbsApplyOnce(pg, aa, rr, yy));
  PetscCall(VecRestoreArray(y, &yy));
  PetscCall(VecRestoreArrayRead(r, &rr));
  PetscCall(MatSeqAIJRestoreArrayRead(pg->A, &aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_PatchGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_PatchGibbs      pg = pc->data;
  const PetscScalar *aa, *bb;
  PetscScalar       *yy;

  (void)w;
  (void)rtol;
  (void)abstol;
  (void)dtol;

  PetscFunctionBegin;
  if (guesszero) PetscCall(VecZeroEntries(y));
  PetscCall(MatSeqAIJGetArrayRead(pg->A, &aa));
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(VecGetArrayRead(b, &bb));
    PetscCall(VecGetArray(y, &yy));
    PetscCall(PCPatchGibbsApplyOnce(pg, aa, bb, yy));
    PetscCall(VecRestoreArray(y, &yy));
    PetscCall(VecRestoreArrayRead(b, &bb));
    if (pg->scb) PetscCall(pg->scb(pg->sample_index++, y, pg->cbctx));
  }
  PetscCall(MatSeqAIJRestoreArrayRead(pg->A, &aa));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_PatchGibbs(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_PatchGibbs pg = pc->data;
  PetscInt      sweep = pg->sweep;
  PetscBool     det   = pg->deterministic;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Patch Gibbs options");
  PetscCall(PetscOptionsEList("-pc_patchgibbs_sweep", "Order in which the patches are visited", NULL, PatchGibbsSweepTypes, 3, PatchGibbsSweepTypes[pg->sweep], &sweep, NULL));
  PetscCall(PetscOptionsBool("-pc_patchgibbs_deterministic", "Deterministic smoother (no noise)", NULL, det, &det, NULL));
  PetscOptionsHeadEnd();
  pg->sweep = (PatchGibbsSweepType)sweep;
  if (det != pg->deterministic) {
    pg->deterministic = det;
    pc->setupcalled   = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_PatchGibbs(PC pc, PetscViewer viewer)
{
  PC_PatchGibbs pg = pc->data;
  PetscBool     isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  %" PetscInt_FMT " patches, %" PetscInt_FMT " patch DOFs in total, largest patch %" PetscInt_FMT "\n", pg->npatches, pg->offsets ? pg->offsets[pg->npatches] : 0, pg->maxsize));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  sweep: %s%s\n", PatchGibbsSweepTypes[pg->sweep], pg->deterministic ? ", deterministic (no noise)" : ""));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_PatchGibbs(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_PatchGibbs pg = pc->data;

  PetscFunctionBegin;
  if (pg->del_scb) PetscCall(pg->del_scb(pg->cbctx));
  pg->scb     = cb;
  pg->cbctx   = ctx;
  pg->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_PatchGibbs(PC pc)
{
  PC_PatchGibbs pg;

  PetscFunctionBegin;
  if (!PATCHGIBBS_SWEEP) {
    PetscCall(PetscLogEventRegister("PatchGibbsSweep", PARMGMC_CLASSID, &PATCHGIBBS_SWEEP));
    PetscCall(PetscLogEventRegister("PatchGibbsSetUp", PARMGMC_CLASSID, &PATCHGIBBS_SETUP));
  }
  PetscCall(PetscNew(&pg));
  pg->sweep         = PATCHGIBBS_FORWARD;
  pg->deterministic = PETSC_FALSE;
  pc->data          = pg;

  pc->ops->setup           = PCSetUp_PatchGibbs;
  pc->ops->reset           = PCReset_PatchGibbs;
  pc->ops->destroy         = PCDestroy_PatchGibbs;
  pc->ops->apply           = PCApply_PatchGibbs;
  pc->ops->applyrichardson = PCApplyRichardson_PatchGibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_PatchGibbs;
  pc->ops->view            = PCView_PatchGibbs;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPatchGibbsSetPatches_C", PCPatchGibbsSetPatches_PatchGibbs));
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_PatchGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
