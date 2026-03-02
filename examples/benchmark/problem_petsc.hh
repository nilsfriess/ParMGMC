#pragma once

#include "parmgmc/obs.h"
#include "problems.hh"

#include "params.hh"
#include "parmgmc/ms.h"

#include <iostream>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdexcept>

struct MeasCtx {
  PetscScalar *centre, radius; // used when qoi is average over sphere
  PetscScalar *start, *end;    // used when qoi is average over rect/cuboid
  PetscScalar  vol;
};

inline PetscErrorCode f_sphere(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)time;
  (void)Nc;
  auto       *octx = (MeasCtx *)ctx;
  PetscScalar diff = 0;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; ++i) diff += PetscSqr(x[i] - octx->centre[i]);
  if (diff < PetscSqr(octx->radius)) *u = 1;
  else *u = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode f_rect(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)time;
  (void)Nc;
  auto     *octx   = (MeasCtx *)ctx;
  PetscBool inside = PETSC_TRUE;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; ++i) {
    if (x[i] < octx->start[i] || x[i] > octx->end[i]) {
      inside = PETSC_FALSE;
      break;
    }
  }
  if (inside) *u = 1;
  else *u = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VolumeOfSphere(DM dm, PetscScalar r, PetscScalar *v)
{
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == 2 || cdim == 3, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only dim=2 and dim=3 supported");
  if (cdim == 2) *v = PETSC_PI * r * r;
  else *v = 4 * PETSC_PI / 3. * r * r * r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VolumeOfRect(DM dm, PetscScalar *start, PetscScalar *end, PetscScalar *v)
{
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == 2 || cdim == 3, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only dim=2 and dim=3 supported");
  *v = 1;
  for (PetscInt i = 0; i < cdim; ++i) *v *= end[i] - start[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

class PetscProblem : public Problem {
public:
  PetscProblem(Parameters params)
  {
    char      filename[512];
    PetscBool flag;
    MS        ms;

    PetscFunctionBeginUser;
    // Assemble precision matrix
    PetscCallVoid(MSCreate(MPI_COMM_WORLD, &ms));
    PetscCallVoid(MSSetAssemblyOnly(ms, PETSC_TRUE));
    PetscCallVoid(MSSetFromOptions(ms));

    PetscCallVoid(PetscOptionsGetString(nullptr, nullptr, "-mesh_file", filename, 512, &flag));
    if (flag) {
      DM mdm;

      PetscCallVoid(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &mdm));
      PetscCallVoid(MSSetDM(ms, mdm));
    }
    PetscCallVoid(MSSetUp(ms));
    PetscCallVoid(MSGetPrecisionMatrix(ms, &A));
    PetscCallVoid(MSGetDM(ms, &dm));
    PetscCallVoid(PetscObjectReference((PetscObject)(A)));
    if (!flag) PetscCallVoid(PetscObjectReference((PetscObject)(dm)));
    PetscCallVoid(MSDestroy(&ms));

    // Create RHS vector
    if (params->with_lr) {
      PetscInt   nobs, cdim, nobs_given;
      PetscReal *obs_coords, *obs_radii, *obs_values, obs_sigma2 = 1e-4;
      PetscBool  flag = PETSC_FALSE;

      PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-nobs", &nobs, &flag));
      PetscCheckAbort(flag, MPI_COMM_WORLD, PETSC_ERR_ARG_NULL, "Must provide observations");
      PetscCallVoid(DMGetCoordinateDim(dm, &cdim));

      nobs_given = nobs * cdim;
      PetscCallVoid(PetscMalloc1(nobs_given, &obs_coords));
      PetscCallVoid(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_coords", obs_coords, &nobs_given, nullptr));
      PetscCheckAbort(nobs_given == nobs * cdim, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation coordinates provided, expected %" PetscInt_FMT " got %" PetscInt_FMT, nobs * cdim, nobs_given);

      PetscCallVoid(PetscMalloc1(nobs, &obs_radii));
      nobs_given = nobs;
      PetscCallVoid(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_radii", obs_radii, &nobs_given, nullptr));
      PetscCheckAbort(nobs_given == 1 || nobs_given == nobs, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation radii provided, expected either 1 or `nobs` got %" PetscInt_FMT, nobs_given);
      if (nobs_given == 1)
        for (PetscInt i = 1; i < nobs; ++i) obs_radii[i] = obs_radii[0]; // If only one radius provided, use that for all observations

      PetscCallVoid(PetscMalloc1(nobs, &obs_values));
      nobs_given = nobs;
      PetscCallVoid(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_values", obs_values, &nobs_given, nullptr));
      PetscCheckAbort(nobs_given == 1 || nobs_given == nobs, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation values provided, expected either 1 or `nobs` got %" PetscInt_FMT, nobs_given);
      if (nobs_given == 1)
        for (PetscInt i = 1; i < nobs; ++i) obs_values[i] = obs_values[0]; // If only one value provided, use that for all observations

      PetscCallVoid(PetscOptionsGetReal(nullptr, nullptr, "-obs_sigma2", &obs_sigma2, nullptr));

      Mat A2, B;
      Vec S;
      PetscCallVoid(MakeObservationMats(dm, nobs, obs_sigma2, obs_coords, obs_radii, obs_values, &B, &S, &rhs));
      PetscCallVoid(MatCreateLRC(A, B, S, nullptr, &A2));
      PetscCallVoid(MatDestroy(&B));
      PetscCallVoid(VecDestroy(&S));
      PetscCallVoid(PetscFree(obs_coords));
      PetscCallVoid(PetscFree(obs_radii));
      PetscCallVoid(PetscFree(obs_values));
      PetscCallVoid(MatDestroy(&A));
      A = A2;
    } else {
      PetscCallVoid(DMCreateGlobalVector(dm, &rhs));
      PetscCallVoid(VecZeroEntries(rhs));
    }

    PetscCallVoid(CreateMeasurementVec());
  }

  PetscErrorCode GetPrecisionMat(Mat *mat) override
  {
    PetscFunctionBeginUser;
    *mat = A;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode GetRHSVec(Vec *v) override
  {
    PetscFunctionBeginUser;
    *v = rhs;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode GetMeasurementVec(Vec *v) override
  {
    PetscFunctionBeginUser;
    *v = meas_vec;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode GetDM(DM *dm) override
  {
    PetscFunctionBeginUser;
    *dm = this->dm;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode VisualiseResults(Vec sample, Vec mean, Vec var) override
  {
    PetscViewer viewer;
    char        filename[512] = "results.vtu";

    PetscFunctionBeginUser;
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
    if (mean) PetscCall(PetscObjectSetName((PetscObject)(mean), "mean"));
    if (var) PetscCall(PetscObjectSetName((PetscObject)(var), "var"));
    if (sample) PetscCall(PetscObjectSetName((PetscObject)(sample), "sample"));
    PetscCall(PetscObjectSetName((PetscObject)(meas_vec), "measurement vec"));

    if (mean) PetscCall(VecView(mean, viewer));
    if (var) PetscCall(VecView(var, viewer));
    PetscCall(VecView(meas_vec, viewer));
    if (sample) PetscCall(VecView(sample, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~PetscProblem()
  {
    PetscFunctionBeginUser;
    PetscCallVoid(DMDestroy(&dm));
    PetscCallVoid(MatDestroy(&A));
    PetscCallVoid(VecDestroy(&rhs));
    PetscCallVoid(VecDestroy(&meas_vec));
    PetscFunctionReturnVoid();
  }

private:
  Mat A;
  Vec meas_vec, rhs;
  DM  dm;

  PetscErrorCode CreateMeasurementVec()
  {
    Mat       M;
    Vec       u;
    MeasCtx   ctx;
    PetscInt  dim, got_dim;
    void     *mctx         = &ctx;
    char      qoi_type[64] = "sphere";
    PetscBool valid_type, flag;
    PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

    PetscFunctionBeginUser;
    PetscCall(DMCreateGlobalVector(dm, &meas_vec));
    PetscCall(DMCreateMassMatrix(dm, dm, &M));
    PetscCall(DMGetGlobalVector(dm, &u));
    PetscCall(DMGetCoordinateDim(dm, &dim));

    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-qoi_type", qoi_type, 64, nullptr));
    PetscCall(PetscStrcmpAny(qoi_type, &valid_type, "sphere", "rect", ""));
    PetscCheck(valid_type, MPI_COMM_WORLD, PETSC_ERR_SUP, "-qoi_type must be sphere or rect");

    PetscCall(PetscStrcmp(qoi_type, "sphere", &flag));
    if (flag) {
      PetscCall(PetscCalloc1(dim, &ctx.centre));
      got_dim = dim;
      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_centre", ctx.centre, &got_dim, nullptr));
      PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed, expected %" PetscInt_FMT, dim);
      ctx.radius = 1;
      PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-qoi_radius", &ctx.radius, nullptr));
      PetscCall(VolumeOfSphere(dm, ctx.radius, &ctx.vol));

      funcs[0] = f_sphere;
    } else {
      PetscCall(PetscCalloc1(dim, &ctx.start));
      PetscCall(PetscCalloc1(dim, &ctx.end));
      for (PetscInt i = 0; i < dim; ++i) ctx.end[i] = 1;
      got_dim = dim;
      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_start", ctx.start, &got_dim, nullptr));
      PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for start, expected %" PetscInt_FMT, dim);
      got_dim = dim;
      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_end", ctx.end, &got_dim, nullptr));
      PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for end, expected %" PetscInt_FMT, dim);
      PetscCall(VolumeOfRect(dm, ctx.start, ctx.end, &ctx.vol));

      funcs[0] = f_rect;
    }
    PetscCall(DMProjectFunction(dm, 0, funcs, &mctx, INSERT_VALUES, u));
    PetscCall(MatMult(M, u, meas_vec));
    PetscCall(DMRestoreGlobalVector(dm, &u));
    PetscCall(MatDestroy(&M));
    if (flag) {
      PetscCall(PetscFree(ctx.centre));
    } else {
      PetscCall(PetscFree(ctx.start));
      PetscCall(PetscFree(ctx.end));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
  {
    PetscFunctionBeginUser;
    PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(DMViewFromOptions(*dm, nullptr, "-dm_view"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};
