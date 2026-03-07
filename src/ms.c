/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/ms.h"
#include "parmgmc/parmgmc.h"

#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscdmplex.h>
#include <petscdmtypes.h>
#include <petscds.h>
#include <petscdstypes.h>
#include <petscerror.h>
#include <petscfe.h>
#include <petscfetypes.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <mpi.h>

typedef struct _MSCtx {
  MPI_Comm     comm;
  DM           dm;
  Mat          A, Id;
  KSP          ksp;
  Vec          b, mean, var;
  PetscScalar  kappa;
  PetscBool    save_samples, assemble_only;
  Vec         *samples;
  PetscScalar *qois;
  PetscErrorCode (*qoi)(PetscInt, Vec, PetscScalar *, void *);
  void *qoictx;
} *MSCtx;

/** @file ms.c
    @brief A sampler to simulate Matérn random fields.

    # Notes
    This class encapsulates a sampler that can generate random samples from
    zero mean Whittle-Matérn fields using the Multigrid Monte Carlo method.

    Internally it uses the PCGAMGMC implementation.
 */

PetscErrorCode MSDestroy(MS *ms)
{
  MSCtx ctx = (*ms)->ctx;

  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->b));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(KSPDestroy(&ctx->ksp));
  PetscCall(VecDestroy(&ctx->mean));
  PetscCall(VecDestroy(&ctx->var));
  PetscCall(DMDestroy(&ctx->dm));
  PetscCall(PetscFree(ctx->qois));
  PetscCall(PetscFree(ctx));
  PetscCall(PetscFree(*ms));
  ms = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetDM(MS ms, DM dm)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  ctx->dm = dm;
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = constants[0] * constants[0] * u[0];
}

static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  if (numConstants > 0) g0[0] = constants[0] * constants[0];
}

static void g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

#pragma GCC diagnostic pop

static PetscErrorCode MS_AssembleMat(MS ms)
{
  MSCtx          ctx = ms->ctx;
  SNES           snes;
  PetscInt       dim;
  PetscBool      simplex;
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  DM             cdm;
  const PetscInt id = 0;

  PetscFunctionBeginUser;
  PetscCall(SNESCreate(ctx->comm, &snes));
  PetscCall(SNESSetDM(snes, ctx->dm));
  PetscCall(SNESSetLagPreconditioner(snes, -1));
  PetscCall(SNESSetLagJacobian(snes, -2));

  PetscCall(DMGetDimension(ctx->dm, &dim));
  PetscCall(DMPlexIsSimplex(ctx->dm, &simplex));
  PetscCall(PetscFECreateLagrange(ctx->comm, dim, 1, simplex, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(ctx->dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(ctx->dm));
  PetscCall(DMGetDS(ctx->dm, &ds));
  if (ctx->kappa != 0) PetscCall(PetscDSSetConstants(ds, 1, &ctx->kappa));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0, NULL, NULL, g3));

  PetscCall(DMGetLabel(ctx->dm, "Face Sets", &label));
  if (!label) {
    PetscCall(DMCreateLabel(ctx->dm, "boundary"));
    PetscCall(DMGetLabel(ctx->dm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(ctx->dm, PETSC_DETERMINE, label));
  }
  PetscCall(DMPlexLabelComplete(ctx->dm, label));
  PetscCall(DMAddBoundary(ctx->dm, DM_BC_NATURAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));

  cdm = ctx->dm;
  while (cdm) {
    PetscCall(DMCreateLabel(cdm, "boundary"));
    PetscCall(DMGetLabel(cdm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(cdm, PETSC_DETERMINE, label));
    PetscCall(DMCopyDisc(ctx->dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }

  PetscCall(DMPlexSetSNESLocalFEM(ctx->dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetUp(snes));
  PetscCall(DMCreateMatrix(ctx->dm, &ctx->A));
  PetscCall(MatCreateVecs(ctx->A, &ctx->b, NULL));
  PetscCall(SNESComputeJacobian(snes, ctx->b, ctx->A, ctx->A));

  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MS_SampleCallback(PetscInt it, Vec y, void *msctx)
{
  MSCtx ctx = msctx;

  PetscFunctionBeginUser;
  if (ctx->save_samples) PetscCall(VecCopy(y, ctx->samples[it]));
  if (ctx->qoi) PetscCall(ctx->qoi(it, y, &ctx->qois[it], ctx->qoictx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSample(MS ms, Vec x)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCall(KSPSolve(ctx->ksp, ctx->b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetNumSamples(MS ms, PetscInt nsamples)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCall(KSPSetTolerances(ctx->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, nsamples));
  if (ctx->qois) PetscCall(PetscFree(ctx->qois));
  PetscCall(PetscCalloc1(nsamples, &ctx->qois));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetSamples(MS ms, const Vec **samples)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCheck(ctx->save_samples, ctx->comm, PETSC_ERR_SUP, "Samples can only be obtained between a call to MSBeginSaveSamples and a call to MSEndSaveSamples");
  *samples = ctx->samples;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSBeginSaveSamples(MS ms)
{
  MSCtx    ctx = ms->ctx;
  PetscInt nsamples;

  PetscFunctionBeginUser;
  PetscCall(KSPGetTolerances(ctx->ksp, NULL, NULL, NULL, &nsamples));
  if (ctx->samples) PetscCall(PetscFree(ctx->samples));
  PetscCall(PetscMalloc1(nsamples, &ctx->samples));
  for (PetscInt i = 0; i < nsamples; ++i) { PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->samples[i])); }

  ctx->save_samples = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MS_ComputeMeanAndVar(MS ms)
{
  MSCtx    ctx = ms->ctx;
  Vec      tmp;
  PetscInt nsamples;

  PetscFunctionBeginUser;
  if (!ctx->mean) {
    PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->mean));
    PetscCall(VecDuplicate(ctx->mean, &ctx->var));
  }
  PetscCall(KSPGetTolerances(ctx->ksp, NULL, NULL, NULL, &nsamples));
  PetscCall(VecZeroEntries(ctx->mean));
  for (PetscInt i = 0; i < nsamples; ++i) PetscCall(VecAXPY(ctx->mean, 1. / nsamples, ctx->samples[i]));

  PetscCall(VecZeroEntries(ctx->var));
  PetscCall(VecDuplicate(ctx->var, &tmp));
  for (PetscInt i = 0; i < nsamples; ++i) {
    PetscCall(VecCopy(ctx->samples[i], tmp));
    PetscCall(VecAXPY(tmp, -1, ctx->mean));
    PetscCall(VecPointwiseMult(tmp, tmp, tmp));
    PetscCall(VecAXPY(ctx->var, 1. / (nsamples - 1), tmp));
  }

  PetscCall(PetscObjectSetName((PetscObject)ctx->mean, "mean"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->var, "var"));

  PetscCall(VecDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSEndSaveSamples(MS ms)
{
  MSCtx    ctx = ms->ctx;
  PetscInt nsamples;

  PetscFunctionBeginUser;
  PetscCall(MS_ComputeMeanAndVar(ms));
  ctx->save_samples = PETSC_FALSE;
  PetscCall(KSPGetTolerances(ctx->ksp, NULL, NULL, NULL, &nsamples));
  for (PetscInt i = 0; i < nsamples; ++i) PetscCall(VecDestroy(&ctx->samples[i]));
  PetscCall(PetscFree(ctx->samples));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetMeanAndVar(MS ms, Vec *mean, Vec *var)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  if (mean) *mean = ctx->mean;
  if (var) *var = ctx->var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetKappa(MS ms, PetscScalar kappa)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscCheck(kappa >= 0, ctx->comm, PETSC_ERR_SUP, "Range parameter kappa must be nonnegative");
  ctx->kappa = kappa;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetDM(MS ms, DM *dm)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  *dm = ctx->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMeshDefault(MPI_Comm comm, DM *dm)
{
  DM       distdm;
  PetscInt faces[2];

  PetscFunctionBeginUser;
  faces[0] = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-box_faces", &faces[0], NULL));
  faces[1] = faces[0];
#if PETSC_VERSION_LT(3, 22, 0)
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, dm));
#else
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_FALSE, dm));
#endif
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &distdm));
  if (distdm) {
    PetscCall(DMDestroy(dm));
    *dm = distdm;
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetUp(MS ms)
{
  MSCtx ctx = ms->ctx;
  PC    pc;

  PetscFunctionBeginUser;
  if (!ctx->dm) PetscCall(CreateMeshDefault(ctx->comm, &ctx->dm));
  PetscCall(MS_AssembleMat(ms));

  if (!ctx->assemble_only) {
    PetscCall(KSPCreate(ctx->comm, &ctx->ksp));
    PetscCall(KSPSetOperators(ctx->ksp, ctx->A, ctx->A));
    PetscCall(KSPSetType(ctx->ksp, KSPRICHARDSON));
    PetscCall(KSPSetDM(ctx->ksp, ctx->dm));
    PetscCall(KSPSetDMActive(ctx->ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
    PetscCall(KSPGetPC(ctx->ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMGMC));
    PetscCall(KSPSetOptionsPrefix(ctx->ksp, "ms_"));
    PetscCall(KSPSetFromOptions(ctx->ksp));
    PetscCall(KSPSetUp(ctx->ksp));
    PetscCall(KSPSetNormType(ctx->ksp, KSP_NORM_NONE));
    PetscCall(KSPSetConvergenceTest(ctx->ksp, KSPConvergedSkip, NULL, NULL));
    PetscCall(KSPSetTolerances(ctx->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    PetscCall(PCSetSampleCallback(pc, MS_SampleCallback, ctx, NULL));
    PetscCall(KSPSetInitialGuessNonzero(ctx->ksp, PETSC_TRUE));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetQOI(MS ms, PetscErrorCode (*qoi)(PetscInt, Vec, PetscScalar *, void *), void *qoictx)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  ctx->qoi = qoi;
  if (qoictx) ctx->qoictx = qoictx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetQOIValues(MS ms, const PetscScalar **values)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  *values = ctx->qois;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSGetPrecisionMatrix(MS ms, Mat *A)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  *A = ctx->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetAssemblyOnly(MS ms, PetscBool flag)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  ctx->assemble_only = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSSetFromOptions(MS ms)
{
  MSCtx ctx = ms->ctx;

  PetscFunctionBeginUser;
  PetscOptionsBegin(ctx->comm, NULL, "Options for the Matern sampler", NULL);
  PetscCall(PetscOptionsReal("-matern_kappa", "Set the range parameter of the Matern covariance", NULL, ctx->kappa, &ctx->kappa, NULL));
  PetscCall(PetscOptionsBool("-matern_assemble_only", "If true, does not setup the sampler, only assembles the precision matrix", NULL, ctx->assemble_only, &ctx->assemble_only, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MSCreate(MPI_Comm comm, MS *ms)
{
  MSCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(ms));
  PetscCall(PetscNew(&ctx));
  (*ms)->ctx = ctx;

  ctx->dm            = NULL;
  ctx->comm          = comm;
  ctx->kappa         = 1;
  ctx->assemble_only = PETSC_FALSE;
  ctx->samples       = NULL;
  ctx->qoi           = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
