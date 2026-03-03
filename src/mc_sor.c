/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petscerror.h>
#include <petscis.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsftypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

/** @file mc_sor.c
    @brief Multicolour Gauss-Seidel/SOR

    # Notes
    This implements a true parallel Gauss-Seidel method (as opposed to PETSc's
    parallel SOR which is actually block Jacobi with Gauss-Seidel in the blocks.

    Implemented for `MATAIJ` and `MATLRC` matrices (with `MATAIJ` as the base
    matrix type).

    ## Developer notes
    Should this be a PC?
*/

typedef struct _MCSOR_Ctx {
  Mat         A, Asor;
  PetscInt   *diagptrs;
  PetscInt    ncolors;
  PetscReal   omega;
  PetscBool   omega_changed;
  VecScatter *scatters;
  Vec        *ghostvecs;
  Vec         idiag;
  ISColoring  isc;
  MatSORType  type;

  Mat B, Bb, Bb_bk;
  Vec z, w, u;

  PetscErrorCode (*sor)(struct _MCSOR_Ctx *, Vec, Vec);
  PetscErrorCode (*postsor)(MCSOR, Vec);
} *MCSOR_Ctx;

PetscErrorCode MCSORDestroy(MCSOR *mc)
{
  PetscFunctionBeginUser;
  if (*mc) {
    MCSOR_Ctx ctx = (*mc)->ctx;

    PetscCall(PetscFree(ctx->diagptrs));
    if (ctx->scatters && ctx->ghostvecs) {
      for (PetscInt i = 0; i < ctx->ncolors; ++i) {
        PetscCall(VecScatterDestroy(&ctx->scatters[i]));
        PetscCall(VecDestroy(&ctx->ghostvecs[i]));
      }
      PetscCall(PetscFree(ctx->ghostvecs));
      PetscCall(PetscFree(ctx->scatters));
    }
    PetscCall(VecDestroy(&ctx->idiag));

    if (ctx->z) PetscCall(VecDestroy(&ctx->z));
    if (ctx->w) PetscCall(VecDestroy(&ctx->w));
    if (ctx->u) PetscCall(VecDestroy(&ctx->u));

    PetscCall(MatDestroy(&ctx->Bb));
    PetscCall(MatDestroy(&ctx->Bb_bk));

    PetscCall(ISColoringDestroy(&ctx->isc));
    PetscCall(PetscFree(ctx));
    PetscCall(PetscFree(*mc));
    *mc = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORGetISColoring(MCSOR mc, ISColoring *isc)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  if (isc) *isc = ctx->isc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORPostSOR_LRC(MCSOR mc, Vec y)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMultTranspose(ctx->B, y, ctx->w));
  PetscAssert(ctx->type == SOR_FORWARD_SWEEP || ctx->type == SOR_BACKWARD_SWEEP, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Symmetric sweep type should be handled outside this function");
  if (ctx->type == SOR_FORWARD_SWEEP) PetscCall(MatMult(ctx->Bb, ctx->w, ctx->z));
  else PetscCall(MatMult(ctx->Bb_bk, ctx->w, ctx->z));
  PetscCall(VecAXPY(y, -1, ctx->z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORUpdateIDiag(MCSOR mc)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCall(MatGetDiagonal(ctx->Asor, ctx->idiag));
  PetscCall(VecReciprocal(ctx->idiag));
  PetscCall(VecScale(ctx->idiag, ctx->omega));
  ctx->omega_changed = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalPointers(Mat A, PetscInt **diagptrs)
{
  PetscInt        rows;
  const PetscInt *i, *j;
  PetscReal      *a;
  Mat             P;
  MatType         type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &type));
  if (strcmp(type, MATSEQAIJ) == 0) P = A;
  else PetscCall(MatMPIAIJGetSeqAIJ(A, &P, NULL, NULL));

  PetscCall(MatGetSize(P, &rows, NULL));
  PetscCall(PetscMalloc1(rows, diagptrs));

  PetscCall(MatSeqAIJGetCSRAndMemType(P, &i, &j, &a, NULL));
  for (PetscInt row = 0; row < rows; ++row) {
    for (PetscInt k = i[row]; k < i[row + 1]; ++k) {
      PetscInt col = j[k];
      if (col == row) (*diagptrs)[row] = k;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateScatters(Mat mat, ISColoring isc, VecScatter **scatters, Vec **ghostvecs)
{
  PetscInt        ncolors, localRows, globalRows, *nTotalOffProc;
  IS             *iss, is;
  Mat             ao; // off-processor part of matrix
  const PetscInt *colmap, *rowptr, *colptr;
  Vec             gvec;

  PetscFunctionBeginUser;
  PetscCall(ISColoringGetIS(isc, PETSC_USE_POINTER, &ncolors, &iss));
  PetscCall(PetscMalloc1(ncolors, scatters));
  PetscCall(PetscMalloc1(ncolors, ghostvecs));
  PetscCall(MatMPIAIJGetSeqAIJ(mat, NULL, &ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(ao, &rowptr, &colptr, NULL, NULL));

  PetscCall(MatGetSize(mat, &globalRows, NULL));
  PetscCall(MatGetLocalSize(mat, &localRows, NULL));
  PetscCall(VecCreateMPIWithArray(MPI_COMM_WORLD, 1, localRows, globalRows, NULL, &gvec));

  // First count the total number of off-processor values for each color
  PetscCall(PetscCalloc1(ncolors, &nTotalOffProc));
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(ISGetLocalSize(iss[color], &nCurCol));
    PetscCall(ISGetIndices(iss[color], &curidxs));
    for (PetscInt i = 0; i < nCurCol; ++i) nTotalOffProc[color] += rowptr[curidxs[i] + 1] - rowptr[curidxs[i]];
    PetscCall(ISRestoreIndices(iss[color], &curidxs));
  }

  // Find the maximum off-processor count across all colors so we can reuse a single buffer.
  PetscInt maxOffProc = 0;
  for (PetscInt color = 0; color < ncolors; ++color)
    if (nTotalOffProc[color] > maxOffProc) maxOffProc = nTotalOffProc[color];

  PetscInt *offProcIdx;
  PetscCall(PetscMalloc1(maxOffProc, &offProcIdx));

  // Now we again loop over all colors and create the required VecScatters
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(ISGetLocalSize(iss[color], &nCurCol));
    PetscCall(ISGetIndices(iss[color], &curidxs));
    PetscInt cnt = 0;
    for (PetscInt i = 0; i < nCurCol; ++i)
      for (PetscInt k = rowptr[curidxs[i]]; k < rowptr[curidxs[i] + 1]; ++k) offProcIdx[cnt++] = colmap[colptr[k]];
    PetscCall(ISRestoreIndices(iss[color], &curidxs));

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nTotalOffProc[color], offProcIdx, PETSC_COPY_VALUES, &is));
    PetscCall(VecCreateSeq(MPI_COMM_SELF, nTotalOffProc[color], &(*ghostvecs)[color]));
    PetscCall(VecScatterCreate(gvec, is, (*ghostvecs)[color], NULL, &((*scatters)[color])));
    PetscCall(ISDestroy(&is));
  }

  PetscCall(PetscFree(offProcIdx));

  PetscCall(ISColoringRestoreIS(isc, PETSC_USE_POINTER, &iss));
  PetscCall(PetscFree(nTotalOffProc));
  PetscCall(VecDestroy(&gvec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORApply(MCSOR mc, Vec b, Vec y)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(MULTICOL_SOR, ctx->A, b, y, NULL));
  if (ctx->omega_changed) PetscCall(MCSORUpdateIDiag(mc));
  if (ctx->type == SOR_SYMMETRIC_SWEEP) {
    ctx->type = SOR_FORWARD_SWEEP;
    PetscCall(ctx->sor(ctx, b, y));
    if (ctx->postsor) PetscCall(ctx->postsor(mc, y));

    ctx->type = SOR_BACKWARD_SWEEP;
    PetscCall(ctx->sor(ctx, b, y));
    if (ctx->postsor) PetscCall(ctx->postsor(mc, y));

    ctx->type = SOR_SYMMETRIC_SWEEP;
  } else {
    PetscCall(ctx->sor(ctx, b, y));
    if (ctx->postsor) PetscCall(ctx->postsor(mc, y));
  }
  PetscCall(PetscLogEventEnd(MULTICOL_SOR, ctx->A, b, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORApply_SEQAIJ(MCSOR_Ctx ctx, Vec b, Vec y)
{
  PetscInt         nind, ncolors;
  const PetscInt  *rowptr, *colptr, *rowind;
  const PetscReal *idiagarr, *barr;
  PetscReal       *matvals, *yarr;
  IS              *iss;

  PetscFunctionBeginUser;
  PetscCall(MatSeqAIJGetCSRAndMemType(ctx->Asor, &rowptr, &colptr, &matvals, NULL));
  PetscCall(ISColoringGetIS(ctx->isc, PETSC_USE_POINTER, &ncolors, &iss));
  PetscCall(VecGetArrayRead(ctx->idiag, &idiagarr));
  PetscCall(VecGetArrayRead(b, &barr));

  PetscCall(VecGetArray(y, &yarr));
  if (ctx->type == SOR_FORWARD_SWEEP) {
    for (PetscInt color = 0; color < ncolors; ++color) {
      PetscCall(ISGetLocalSize(iss[color], &nind));
      PetscCall(ISGetIndices(iss[color], &rowind));
      for (PetscInt i = 0; i < nind; ++i) {
        const PetscInt r   = rowind[i];
        PetscReal      sum = barr[r];

        for (PetscInt k = rowptr[r]; k < ctx->diagptrs[r]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        for (PetscInt k = ctx->diagptrs[r] + 1; k < rowptr[r + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];

        yarr[r] = (1. - ctx->omega) * yarr[r] + idiagarr[r] * sum;
      }

      PetscCall(ISRestoreIndices(iss[color], &rowind));
    }
  }
  if (ctx->type == SOR_BACKWARD_SWEEP) {
    for (PetscInt color = ncolors - 1; color >= 0; --color) {
      PetscCall(ISGetLocalSize(iss[color], &nind));
      PetscCall(ISGetIndices(iss[color], &rowind));
      for (PetscInt i = nind - 1; i >= 0; --i) {
        const PetscInt r   = rowind[i];
        PetscReal      sum = barr[r];

        for (PetscInt k = rowptr[r]; k < ctx->diagptrs[r]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        for (PetscInt k = ctx->diagptrs[r] + 1; k < rowptr[r + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];

        yarr[r] = (1. - ctx->omega) * yarr[r] + idiagarr[r] * sum;
      }

      PetscCall(ISRestoreIndices(iss[color], &rowind));
    }
  }
  PetscCall(VecRestoreArray(y, &yarr));

  PetscCall(VecRestoreArrayRead(b, &barr));
  PetscCall(VecRestoreArrayRead(ctx->idiag, &idiagarr));
  PetscCall(ISColoringRestoreIS(ctx->isc, PETSC_USE_POINTER, &iss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORApply_MPIAIJ(MCSOR_Ctx ctx, Vec b, Vec y)
{
  Mat              ad, ao; // Local and off-processor parts of mat
  PetscInt         nind, gcnt, ncolors;
  const PetscInt  *rowptr, *colptr, *bRowptr, *bColptr, *rowind;
  const PetscReal *idiagarr, *barr, *ghostarr;
  PetscReal       *matvals, *bMatvals, *yarr;
  IS              *iss;

  PetscFunctionBeginUser;
  PetscCall(MatMPIAIJGetSeqAIJ(ctx->Asor, &ad, &ao, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(ad, &rowptr, &colptr, &matvals, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(ao, &bRowptr, &bColptr, &bMatvals, NULL));
  PetscCall(ISColoringGetIS(ctx->isc, PETSC_USE_POINTER, &ncolors, &iss));

  PetscCall(VecGetArrayRead(ctx->idiag, &idiagarr));
  PetscCall(VecGetArrayRead(b, &barr));

  if (ctx->type == SOR_FORWARD_SWEEP) {
    for (PetscInt color = 0; color < ncolors; ++color) {
      PetscCall(VecScatterBegin(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(ctx->ghostvecs[color], &ghostarr));

      PetscCall(ISGetLocalSize(iss[color], &nind));
      PetscCall(ISGetIndices(iss[color], &rowind));
      PetscCall(VecGetArray(y, &yarr));

      gcnt = 0;
      for (PetscInt i = 0; i < nind; ++i) {
        PetscReal sum = 0;

        for (PetscInt k = rowptr[rowind[i]]; k < ctx->diagptrs[rowind[i]]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        for (PetscInt k = ctx->diagptrs[rowind[i]] + 1; k < rowptr[rowind[i] + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        for (PetscInt k = bRowptr[rowind[i]]; k < bRowptr[rowind[i] + 1]; ++k) sum -= bMatvals[k] * ghostarr[gcnt++];

        yarr[rowind[i]] = (1 - ctx->omega) * yarr[rowind[i]] + idiagarr[rowind[i]] * (sum + barr[rowind[i]]);
      }

      PetscCall(VecRestoreArray(y, &yarr));
      PetscCall(VecRestoreArrayRead(ctx->ghostvecs[color], &ghostarr));
      PetscCall(ISRestoreIndices(iss[color], &rowind));
    }
  }

  if (ctx->type == SOR_BACKWARD_SWEEP) {
    for (PetscInt color = ncolors - 1; color >= 0; --color) {
      PetscCall(VecScatterBegin(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ctx->scatters[color], y, ctx->ghostvecs[color], INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(ctx->ghostvecs[color], &ghostarr));

      PetscCall(ISGetLocalSize(iss[color], &nind));
      PetscCall(ISGetIndices(iss[color], &rowind));
      PetscCall(VecGetArray(y, &yarr));

      // ghostarr is laid out in forward row order (rowind[0], rowind[1], ...).
      // For the backward sweep we iterate rows in reverse, so we compute each
      // row's start offset by working backwards from the total ghost count.
      PetscInt ghostSize;
      PetscCall(VecGetLocalSize(ctx->ghostvecs[color], &ghostSize));
      gcnt = ghostSize;
      for (PetscInt i = nind - 1; i >= 0; --i) {
        PetscReal sum = 0;
        gcnt -= bRowptr[rowind[i] + 1] - bRowptr[rowind[i]];

        for (PetscInt k = rowptr[rowind[i]]; k < ctx->diagptrs[rowind[i]]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        for (PetscInt k = ctx->diagptrs[rowind[i]] + 1; k < rowptr[rowind[i] + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];
        PetscInt go = gcnt;
        for (PetscInt k = bRowptr[rowind[i]]; k < bRowptr[rowind[i] + 1]; ++k) sum -= bMatvals[k] * ghostarr[go++];

        yarr[rowind[i]] = (1 - ctx->omega) * yarr[rowind[i]] + idiagarr[rowind[i]] * (sum + barr[rowind[i]]);
      }

      PetscCall(VecRestoreArray(y, &yarr));
      PetscCall(VecRestoreArrayRead(ctx->ghostvecs[color], &ghostarr));
      PetscCall(ISRestoreIndices(iss[color], &rowind));
    }
  }

  PetscCall(VecRestoreArrayRead(b, &barr));
  PetscCall(VecRestoreArrayRead(ctx->idiag, &idiagarr));
  PetscCall(ISColoringRestoreIS(ctx->isc, PETSC_USE_POINTER, &iss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateISColoring_AIJ(Mat A, ISColoring *isc)
{
  MatColoring mc;

  PetscFunctionBeginUser;
  PetscCall(MatColoringCreate(A, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGJP));
  PetscCall(MatColoringApply(mc, isc));
  PetscCall(ISColoringSetType(*isc, IS_COLORING_LOCAL));
  PetscCall(MatColoringDestroy(&mc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateISColoring_Seq(Mat A, ISColoring *isc)
{
  PetscInt         start, end;
  ISColoringValue *vals;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  PetscCall(PetscMalloc1(end - start, &vals));
  for (PetscInt i = 0; i < end - start; ++i) { vals[i] = 0; }

  PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)A), 1, end - start, vals, PETSC_OWN_POINTER, isc));
  PetscCall(ISColoringSetType(*isc, IS_COLORING_LOCAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORSetOmega(MCSOR mc, PetscReal omega)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  ctx->omega         = omega;
  ctx->omega_changed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORSetSweepType(MCSOR mc, MatSORType type)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCheck(type == SOR_FORWARD_SWEEP || type == SOR_BACKWARD_SWEEP || type == SOR_SYMMETRIC_SWEEP, PetscObjectComm((PetscObject)ctx->A), PETSC_ERR_SUP, "Only forward, backward and symmetric sweep supported");
  ctx->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORGetSweepType(MCSOR mc, MatSORType *type)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  *type = ctx->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORSetupSOR(MCSOR mc)
{
  MCSOR_Ctx   ctx = mc->ctx;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(MatGetDiagonalPointers(ctx->Asor, &(ctx->diagptrs)));
  PetscCall(MatCreateVecs(ctx->Asor, &ctx->idiag, NULL));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ctx->Asor), &size));
  if (size == 1) PetscCall(MatCreateISColoring_Seq(ctx->Asor, &ctx->isc));
  else PetscCall(MatCreateISColoring_AIJ(ctx->Asor, &ctx->isc));
  PetscCall(ISColoringGetIS(ctx->isc, PETSC_USE_POINTER, &ctx->ncolors, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORCreateUpdateMat(MCSOR mc, MatSORType direction, Mat A, Mat *C)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscAssert(direction == SOR_FORWARD_SWEEP || direction == SOR_BACKWARD_SWEEP, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Symmetric sweep type should be handled outside this function");
  PetscCall(MatDuplicate(ctx->B, MAT_DO_NOT_COPY_VALUES, C));
  {
    MCSOR    mca;
    PetscInt cols;
    Vec      x;

    PetscCall(MatGetSize(ctx->B, NULL, &cols));
    PetscCall(MCSORCreate(ctx->Asor, &mca));
    PetscCall(MCSORSetSweepType(mca, direction));
    PetscCall(MCSORSetUp(mca));
    PetscCall(MatCreateVecs(A, &x, NULL));
    for (PetscInt i = 0; i < cols; ++i) {
      Vec b, c;

      PetscCall(VecZeroEntries(x));
      PetscCall(MatDenseGetColumnVecRead(ctx->B, i, &b));
      PetscCall(MCSORApply(mca, b, x));
      PetscCall(MatDenseRestoreColumnVecRead(ctx->B, i, &b));

      PetscCall(MatDenseGetColumnVecWrite(*C, i, &c));
      PetscCall(VecCopy(x, c));
      PetscCall(MatDenseRestoreColumnVecWrite(*C, i, &c));
    }
    PetscCall(VecDestroy(&x));
    PetscCall(MCSORDestroy(&mca));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORCreateBbarMat(MCSOR mc, Mat C, Vec S, Mat *Bb)
{
  MCSOR_Ctx  ctx = mc->ctx;
  Mat        tmp, Id, Sb;
  KSP        ksp;
  Vec        Si;
  IS         sctis;
  VecScatter sct;
  PetscInt   sctsize;

  PetscFunctionBeginUser;
  PetscCall(MatTransposeMatMult(ctx->B, C, MAT_INITIAL_MATRIX, 1, &tmp)); // tmp = B^T P^T M_P^-1 B

  PetscCall(VecGetSize(S, &sctsize));
  PetscCall(ISCreateStride(MPI_COMM_WORLD, sctsize, 0, 1, &sctis));
  PetscCall(MatCreateVecs(C, &Si, NULL));
  PetscCall(VecScatterCreate(S, sctis, Si, NULL, &sct));
  PetscCall(VecScatterBegin(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&sct));
  PetscCall(ISDestroy(&sctis));
  PetscCall(VecReciprocal(Si));

  PetscCall(MatDiagonalSet(tmp, Si, ADD_VALUES)); // tmp = S^-1 +  B^T M_P^-1 B
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, tmp, tmp));
  PetscCall(MatDuplicate(tmp, MAT_DO_NOT_COPY_VALUES, &Id));
  PetscCall(MatShift(Id, 1));
  PetscCall(MatDuplicate(tmp, MAT_DO_NOT_COPY_VALUES, &Sb));
  PetscCall(KSPMatSolve(ksp, Id, Sb)); // ctx->Sb = ( S^-1 +  B^T M_P^-1 B )^-1

  PetscCall(MatMatMult(C, Sb, MAT_INITIAL_MATRIX, 1, Bb));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&Si));
  PetscCall(MatDestroy(&Id));
  PetscCall(MatDestroy(&Sb));
  PetscCall(MatDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORSetUp(MCSOR mc)
{
  MCSOR_Ctx ctx = mc->ctx;
  MatType   type;
  Mat       A = ctx->A;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    ctx->Asor = A;
  } else if (strcmp(type, MATMPIAIJ) == 0) {
    ctx->Asor = A;
  } else if (strcmp(type, MATLRC) == 0) {
    PetscCall(MatLRCGetMats(A, &ctx->Asor, NULL, NULL, NULL));
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
  }
  PetscCall(MCSORSetupSOR(mc));

  if (strcmp(type, MATLRC) == 0) {
    Mat C;
    Vec S;

    PetscCall(MatLRCGetMats(A, &ctx->Asor, &ctx->B, &S, NULL));

    // Compute C = M_P^-1 B, where M_P is the lower triangular part of PAP^T.
    // We compute this column-by-column using multicolour Gauss-Seidel.
    // This way, we don't need to explicitly permute A and get M_P.
    PetscCall(MCSORCreateUpdateMat(mc, SOR_FORWARD_SWEEP, A, &C));
    PetscCall(MCSORCreateBbarMat(mc, C, S, &ctx->Bb));
    PetscCall(MatDestroy(&C));

    PetscCall(MCSORCreateUpdateMat(mc, SOR_BACKWARD_SWEEP, A, &C));
    PetscCall(MCSORCreateBbarMat(mc, C, S, &ctx->Bb_bk));
    PetscCall(MatDestroy(&C));

    PetscCall(MatCreateVecs(ctx->Bb, &ctx->w, &ctx->z));

    ctx->postsor = MCSORPostSOR_LRC;
  }

  PetscCall(MatGetType(ctx->Asor, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    ctx->sor = MCSORApply_SEQAIJ;
  } else {
    PetscCall(MatCreateScatters(ctx->Asor, ctx->isc, &ctx->scatters, &ctx->ghostvecs));
    ctx->sor = MCSORApply_MPIAIJ;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORGetNumColors(MCSOR mc, PetscInt *colors)
{
  MCSOR_Ctx ctx = mc->ctx;
  PetscInt  ncolors;

  PetscFunctionBeginUser;
  PetscCall(ISColoringGetIS(ctx->isc, PETSC_USE_POINTER, &ncolors, NULL));
  *colors = ncolors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORCreate(Mat A, MCSOR *m)
{
  MCSOR     mc;
  MCSOR_Ctx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&mc));
  PetscCall(PetscNew(&ctx));

  ctx->scatters      = NULL;
  ctx->ghostvecs     = NULL;
  ctx->omega_changed = PETSC_TRUE;
  ctx->A             = A;
  mc->ctx            = ctx;
  ctx->B             = NULL;
  ctx->z             = NULL;
  ctx->w             = NULL;
  ctx->postsor       = NULL;
  ctx->type          = SOR_FORWARD_SWEEP;
  ctx->omega         = 1;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mc_sor_omega", &ctx->omega, NULL)); // TODO: Put this in a seperate MCSORSetFromOptions

  *m = mc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
