/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"

#include <petscksp.h>
#include <petscoptions.h>
#include <petscvec.h>
#include <stdbool.h>

#include <petscsys.h>
#include <petscis.h>
#include <petscmat.h>

/** @file mc_sor.c
    @brief Multicolour Gauss-Seidel/SOR

    # Notes
    This implements a true parallel Gauss-Seidel method (as opposed to PETSc's
    parallel SOR which is actually block Jacobi with Gauss-Seidel in the blocks.

    Implemented for `MATAIJ` and `MATLRC` matrices (with `MATAIJ` as the base
    matrix type).

    Users should not use this class directly but rather access it through
    PCSOR.

    ## Developer notes
    Should this be a PC?
*/

typedef struct _MCSOR_Ctx {
  Mat         A, Asor;
  PetscInt   *diagptrs, ncolors;
  PetscReal   omega;
  PetscBool   omega_changed;
  VecScatter *scatters;
  Vec        *ghostvecs;
  Vec         idiag;
  ISColoring  isc;

  PetscBool explicit_lr; // whether or not to explicitly build the matrix that is used in each iteration to update the solution (

  Mat L, B, Sb, Bb;
  Vec z, w, v, u;

  PetscErrorCode (*sor)(struct _MCSOR_Ctx *, Vec, Vec);
  PetscErrorCode (*postsor)(MCSOR, Vec);
} *MCSOR_Ctx;

PetscErrorCode MCSORDestroy(MCSOR *mc)
{
  PetscFunctionBeginUser;
  if (*mc) {
    MCSOR_Ctx ctx = (*mc)->ctx;
    PetscInt  ncolors;

    PetscCall(PetscFree(ctx->diagptrs));
    PetscCall(ISColoringGetIS(ctx->isc, PETSC_USE_POINTER, &ncolors, NULL));
    if (ctx->scatters && ctx->ghostvecs) {
      for (PetscInt i = 0; i < ncolors; ++i) {
        PetscCall(VecScatterDestroy(&ctx->scatters[i]));
        PetscCall(VecDestroy(&ctx->ghostvecs[i]));
      }
    }
    PetscCall(VecDestroy(&ctx->idiag));

    if (ctx->L) PetscCall(MatDestroy(&ctx->L));
    if (ctx->Sb) PetscCall(MatDestroy(&ctx->Sb));
    if (ctx->z) PetscCall(VecDestroy(&ctx->z));
    if (ctx->w) PetscCall(VecDestroy(&ctx->w));
    if (ctx->v) PetscCall(VecDestroy(&ctx->v));
    if (ctx->u) PetscCall(VecDestroy(&ctx->u));

    if (ctx->explicit_lr) PetscCall(MatDestroy(&ctx->Bb));

    PetscCall(ISColoringDestroy(&ctx->isc));
    PetscCall(PetscFree(ctx));
    PetscCall(PetscFree(*mc));
    *mc = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MCSORPostSOR_LRC(MCSOR mc, Vec y)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMultTranspose(ctx->B, y, ctx->w));
  if (ctx->explicit_lr) {
    PetscCall(MatMult(ctx->Bb, ctx->w, ctx->z));
  } else {
    PetscCall(MatMult(ctx->Sb, ctx->w, ctx->v));
    PetscCall(MatMult(ctx->B, ctx->v, ctx->u));
    PetscCall(MatSolve(ctx->L, ctx->u, ctx->z));
  }
  PetscCall(VecAXPY(y, -1, ctx->z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorLowerTriangular(Mat A, Mat *LL)
{
  PetscInt           lstart, lend, ncols, diagcol;
  const PetscInt    *cols;
  const PetscScalar *vals;
  IS                 rowperm, colperm;
  Mat                L;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &L));
  PetscCall(MatGetOwnershipRange(A, &lstart, &lend));
  for (PetscInt i = lstart; i < lend; ++i) {
    PetscCall(MatGetRow(A, i, &ncols, &cols, &vals));
    diagcol = 0;
    for (PetscInt j = 0; j < ncols; ++j) {
      diagcol++;
      if (cols[j] == i) break;
    }
    PetscCall(MatSetValues(L, 1, &i, diagcol, cols, vals, INSERT_VALUES));
    PetscCall(MatRestoreRow(A, i, &ncols, &cols, &vals));
  }
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatEliminateZeros(L, PETSC_TRUE));
  PetscCall(MatGetOrdering(L, MATORDERINGEXTERNAL, &rowperm, &colperm));
  PetscCall(MatGetFactor(L, MATSOLVERMUMPS, MAT_FACTOR_LU, LL));
  PetscCall(MatLUFactorSymbolic(*LL, L, rowperm, colperm, NULL));
  PetscCall(MatLUFactorNumeric(*LL, L, NULL));

  PetscCall(MatDestroy(&L));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
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

  // Now we again loop over all colors and create the required VecScatters
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscInt       *offProcIdx;
    PetscInt        nCurCol;
    const PetscInt *curidxs;
    PetscCall(PetscMalloc1(nTotalOffProc[color], &offProcIdx));
    PetscCall(ISGetLocalSize(iss[color], &nCurCol));
    PetscCall(ISGetIndices(iss[color], &curidxs));
    PetscInt cnt = 0;
    for (PetscInt i = 0; i < nCurCol; ++i)
      for (PetscInt k = rowptr[curidxs[i]]; k < rowptr[curidxs[i] + 1]; ++k) offProcIdx[cnt++] = colmap[colptr[k]];
    PetscCall(ISRestoreIndices(iss[color], &curidxs));

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nTotalOffProc[color], offProcIdx, PETSC_USE_POINTER, &is));
    PetscCall(VecCreateSeq(MPI_COMM_SELF, nTotalOffProc[color], &(*ghostvecs)[color]));
    PetscCall(VecScatterCreate(gvec, is, (*ghostvecs)[color], NULL, &((*scatters)[color])));
    PetscCall(ISDestroy(&is));
    PetscCall(PetscFree(offProcIdx));
  }

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
  PetscCall(ctx->sor(ctx, b, y));
  if (ctx->postsor) PetscCall(ctx->postsor(mc, y));
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
  for (PetscInt color = 0; color < ncolors; ++color) {
    PetscCall(ISGetLocalSize(iss[color], &nind));
    PetscCall(ISGetIndices(iss[color], &rowind));

    for (PetscInt i = 0; i < nind; ++i) {
      PetscInt  r   = rowind[i];
      PetscReal sum = barr[r];

      for (PetscInt k = rowptr[r]; k < ctx->diagptrs[r]; ++k) sum -= matvals[k] * yarr[colptr[k]];
      for (PetscInt k = ctx->diagptrs[r] + 1; k < rowptr[r + 1]; ++k) sum -= matvals[k] * yarr[colptr[k]];

      yarr[r] = (1. - ctx->omega) * yarr[r] + idiagarr[r] * sum;
    }

    PetscCall(ISRestoreIndices(iss[color], &rowind));
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
  PetscCall(MatColoringSetType(mc, MATCOLORINGGREEDY));
  PetscCall(MatColoringApply(mc, isc));
  PetscCall(ISColoringSetType(*isc, IS_COLORING_LOCAL));
  PetscCall(MatColoringDestroy(&mc));
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

static PetscErrorCode MCSORSetupSOR(MCSOR mc)
{
  MCSOR_Ctx ctx = mc->ctx;

  PetscFunctionBeginUser;
  PetscCall(MatGetDiagonalPointers(ctx->Asor, &(ctx->diagptrs)));
  PetscCall(MatCreateVecs(ctx->Asor, &ctx->idiag, NULL));
  PetscCall(MatCreateISColoring_AIJ(ctx->Asor, &ctx->isc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MCSORCreate(Mat A, PetscReal omega, PetscBool explicit_lr, MCSOR *m)
{
  MatType   type;
  MCSOR     mc;
  MCSOR_Ctx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&mc));
  PetscCall(PetscNew(&ctx));
  ctx->omega = omega;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mc_sor_omega", &ctx->omega, NULL));

  ctx->scatters      = NULL;
  ctx->ghostvecs     = NULL;
  ctx->omega_changed = PETSC_TRUE;
  ctx->A             = A;
  mc->ctx            = ctx;
  ctx->L             = NULL;
  ctx->B             = NULL;
  ctx->Sb            = NULL;
  ctx->z             = NULL;
  ctx->w             = NULL;
  ctx->v             = NULL;
  ctx->postsor       = NULL;
  ctx->explicit_lr   = explicit_lr;

  PetscCall(MatGetType(A, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    ctx->Asor = A;
  } else if (strcmp(type, MATMPIAIJ) == 0) {
    ctx->Asor = A;
  } else if (strcmp(type, MATLRC) == 0) {
    Mat        tmp, tmp2, Id;
    KSP        ksp;
    Vec        S, Si;
    IS         sctis;
    VecScatter sct;
    PetscInt   sctsize;
    PetscCall(MatLRCGetMats(A, &ctx->Asor, &ctx->B, &S, NULL));
    PetscCall(MatLUFactorLowerTriangular(ctx->Asor, &ctx->L));
    PetscCall(MatCreateVecs(ctx->L, &ctx->z, NULL));
    PetscCall(MatDuplicate(ctx->B, MAT_DO_NOT_COPY_VALUES, &tmp));
    PetscCall(MatMatSolve(ctx->L, ctx->B, tmp));                               // tmp = L^-1 B
    PetscCall(MatTransposeMatMult(ctx->B, tmp, MAT_INITIAL_MATRIX, 1, &tmp2)); // tmp2 = B^T L^-1 B

    PetscCall(VecGetSize(S, &sctsize));
    PetscCall(ISCreateStride(MPI_COMM_WORLD, sctsize, 0, 1, &sctis));
    PetscCall(MatCreateVecs(tmp, &Si, NULL));
    PetscCall(VecScatterCreate(S, sctis, Si, NULL, &sct));
    PetscCall(VecScatterBegin(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&sct));
    PetscCall(ISDestroy(&sctis));
    PetscCall(VecReciprocal(Si));

    PetscCall(MatDiagonalSet(tmp2, Si, ADD_VALUES)); // tmp2 = S^-1 + B^T L^-1 B
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, tmp2, tmp2));
    PetscCall(MatDuplicate(tmp2, MAT_DO_NOT_COPY_VALUES, &Id));
    PetscCall(MatShift(Id, 1));
    PetscCall(MatDuplicate(tmp2, MAT_DO_NOT_COPY_VALUES, &ctx->Sb));
    PetscCall(KSPMatSolve(ksp, Id, ctx->Sb)); // ctx->Sb = (S^-1 + B^T L^-1 B)^-1

    if (explicit_lr) {
      PetscCall(MatMatMult(ctx->B, ctx->Sb, MAT_INITIAL_MATRIX, 1, &ctx->Bb));
      PetscCall(MatMatSolve(ctx->L, ctx->Bb, ctx->Bb));
    }

    PetscCall(MatCreateVecs(ctx->Sb, &ctx->w, NULL));
    PetscCall(MatCreateVecs(ctx->Sb, &ctx->v, NULL));
    PetscCall(MatCreateVecs(ctx->B, NULL, &ctx->u));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&Si));
    PetscCall(MatDestroy(&Id));
    PetscCall(MatDestroy(&tmp));
    PetscCall(MatDestroy(&tmp2));

    ctx->postsor = MCSORPostSOR_LRC;
  } else {
    PetscCheck(false, MPI_COMM_WORLD, PETSC_ERR_SUP, "Matrix type not supported");
  }

  PetscCall(MCSORSetupSOR(mc));
  PetscCall(MatGetType(ctx->Asor, &type));
  if (strcmp(type, MATSEQAIJ) == 0) {
    ctx->sor = MCSORApply_SEQAIJ;
  } else {
    PetscCall(MatCreateScatters(ctx->Asor, ctx->isc, &ctx->scatters, &ctx->ghostvecs));
    ctx->sor = MCSORApply_MPIAIJ;
  }

  *m = mc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
