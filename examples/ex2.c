/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Samples from a Gaussian random field with Matern covariance using the MS
 *  interface.
 *
 */

/**************************** Test specification ****************************/
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -dm_refine 2 -matern_kappa 10 %opts
/****************************************************************************/

#include "mpi.h"
#include "parmgmc/iact.h"
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscmath.h>
#include <petscpartitioner.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <petscsys.h>

#include <parmgmc/ms.h>
#include <parmgmc/parmgmc.h>

static PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define N_BURNIN  100
#define N_SAMPLES 50000
#define N_SAVE    10

static PetscErrorCode qoi(PetscInt it, Vec sample, PetscScalar *value, void *qctx)
{
  (void)it;
  Vec meas_vec = (Vec)qctx;

  PetscFunctionBeginUser;
  PetscCall(VecDot(sample, meas_vec, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM                 dm;
  MS                 ms;
  Vec                x, var, mean, meas_vec;
  PetscViewer        viewer;
  PetscBool          flag;
  char               filename[512];
  const Vec         *samples;
  const PetscScalar *qois;
  PetscScalar        tau, qoimean = 0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_file", filename, 512, &flag));
  if (flag) {
    PetscCall(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &dm));
    PetscCall(MSSetDM(ms, dm));
  }
  flag = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-save_samples", &flag, NULL));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting setup... "));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetDM(ms, &dm));

  PetscCall(DMCreateGlobalVector(dm, &meas_vec));
  PetscCall(VecSet(meas_vec, 1));
  PetscCall(VecNormalize(meas_vec, NULL));

  PetscCall(MSSetQOI(ms, qoi, meas_vec));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Done\n"));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting burnin... "));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(MSSetNumSamples(ms, N_BURNIN));
  PetscCall(MSSample(ms, x));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Done.\n"));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting sampling... "));
  PetscCall(MSSetNumSamples(ms, N_SAMPLES));
  if (flag) PetscCall(MSBeginSaveSamples(ms));
  PetscCall(MSSample(ms, x));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Done\n"));

  PetscCall(MSGetQOIValues(ms, &qois));
  PetscCall(IACT(N_SAMPLES, qois, &tau, NULL, NULL));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "IACT = %.5f\n", tau));
  for (PetscInt i = 0; i < N_SAMPLES; ++i) qoimean += qois[i] / N_SAMPLES;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Relative mean error = %.5f\n", PetscAbs(qoimean)));

  PetscCheck(PetscIsCloseAtTol(qoimean, 0, 0.01, 0.01), MPI_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "QOI mean has not converged: got %.4f, expected %.4f", qoimean, 0.f);

  if (flag) {
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, "samples.vtu", FILE_MODE_WRITE, &viewer));
    PetscCall(MSGetSamples(ms, &samples));
    for (PetscInt i = 0; i < N_SAVE; ++i) {
      char name[256];
      Vec  sample = samples[N_SAMPLES - N_SAVE + i];

      sprintf(name, "Sample %02d_", i);
      PetscCall(PetscObjectSetName((PetscObject)(sample), name));
      PetscCall(VecView(sample, viewer));
    }

    PetscCall(MSEndSaveSamples(ms));
    PetscCall(MSGetMeanAndVar(ms, &mean, &var));

    PetscCall(VecView(mean, viewer));
    PetscCall(VecView(var, viewer));
    PetscCall(VecView(meas_vec, viewer));

    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&meas_vec));
  PetscCall(VecDestroy(&x));
  PetscCall(MSDestroy(&ms));
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
  return 0;
}
