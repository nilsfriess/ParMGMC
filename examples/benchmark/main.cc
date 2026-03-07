#include <petscdm.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewertypes.h>
#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>

#include "parmgmc/iact.h"
#include "parmgmc/parmgmc.h"

#include "params.hh"
#include "problem_petsc.hh"

#if defined(PARMGMC_HAVE_MFEM)
  #include "problem_mfem.hh"
#endif

struct SampleCtx {
  SampleCtx(Vec b, PetscInt nqois, Vec meas_vec, PetscBool est_mean_and_var) : nqois{nqois}, meas_vec{meas_vec}, est_mean_and_var{est_mean_and_var}
  {
    PetscFunctionBeginUser;
    PetscCallVoid(VecDuplicate(b, &mean));
    PetscCallVoid(VecSet(mean, 0));
    PetscCallVoid(VecDuplicate(mean, &M));
    PetscCallVoid(VecSet(M, 0));
    PetscCallVoid(VecDuplicate(mean, &var));
    PetscCallVoid(VecDuplicate(mean, &delta));
    PetscCallVoid(VecDuplicate(mean, &delta2));
    PetscCallVoid(PetscCalloc1(nqois, &qois));
    PetscFunctionReturnVoid();
  }

  PetscErrorCode GetMean(Vec *mean)
  {
    PetscFunctionBeginUser;
    *mean = this->mean;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode GetVar(Vec *var)
  {
    PetscFunctionBeginUser;
    // Divide M (sum of squared deviations) by nseen to get the population
    // variance. Use a copy so this method is safe to call multiple times.
    PetscCall(VecCopy(M, this->var));
    if (nseen > 0) PetscCall(VecScale(this->var, 1. / nseen));
    *var = this->var;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~SampleCtx()
  {
    PetscFunctionBeginUser;
    PetscCallVoid(VecDestroy(&M));
    PetscCallVoid(VecDestroy(&var));
    PetscCallVoid(VecDestroy(&mean));
    PetscCallVoid(VecDestroy(&delta));
    PetscCallVoid(VecDestroy(&delta2));
    PetscCallVoid(PetscFree(qois));
    PetscFunctionReturnVoid();
  }

  PetscInt     nqois;
  PetscInt     nseen = 0;
  PetscScalar *qois;
  Vec          meas_vec = nullptr;
  Vec          M = nullptr, var = nullptr, mean = nullptr, delta = nullptr, delta2 = nullptr;
  PetscBool    est_mean_and_var;
};

static PetscErrorCode InfoView(Mat A, Parameters params, PetscViewer viewer)
{
  PetscInt    n;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCheck(viewer == PETSC_VIEWER_STDOUT_WORLD || viewer == PETSC_VIEWER_STDOUT_SELF, MPI_COMM_WORLD, PETSC_ERR_SUP, "Viewer not supported");
  PetscCall(PetscViewerASCIIPrintf(viewer, "################################################################################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                              Benchmark parameters\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "################################################################################\n"));
  PetscCall(ParametersView(params, viewer));

  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(MatGetSize(A, &n, nullptr));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Problem size (degrees of freedom): %" PetscInt_FMT "\n\n", n));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Running on %d MPI ranks\n\n", size));

  PetscCall(PetscOptionsView(nullptr, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SamplerCreate(Mat A, DM dm, Parameters params, KSP *ksp)
{
  (void)params;

  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(MPI_COMM_WORLD, ksp));
  PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetConvergenceTest(*ksp, KSPConvergedSkip, nullptr, nullptr));
  PetscCall(KSPSetOperators(*ksp, A, A));
  if (dm) {
    PetscCall(KSPSetDM(*ksp, dm));
#if PETSC_VERSION_GT(3, 24, 5)
    PetscCall(KSPSetDMActive(*ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
#else
    PetscCall(KSPSetDMActive(*ksp, PETSC_FALSE));
#endif
  }
  PetscCall(KSPSetUp(*ksp));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Burnin(KSP ksp, Vec b, Parameters params)
{
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, params->n_burnin));
  PetscCall(VecDuplicate(b, &x));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Sample(KSP ksp, Vec b, Parameters params, Vec x)
{
  PetscFunctionBeginUser;
  PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, params->n_samples));
  PetscCall(KSPSolve(ksp, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SaveSample(PetscInt it, Vec y, void *ctx)
{
  auto       *sctx = (SampleCtx *)ctx;
  PetscScalar qoi, *qois = sctx->qois;
  Vec         meas_vec = sctx->meas_vec;

  PetscFunctionBeginUser;
  if (sctx->est_mean_and_var) {
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    ++sctx->nseen;
    PetscInt i = sctx->nseen;

    PetscCall(VecCopy(y, sctx->delta));
    PetscCall(VecAXPY(sctx->delta, -1, sctx->mean));
    PetscCall(VecAXPY(sctx->mean, 1. / i, sctx->delta));

    PetscCall(VecCopy(y, sctx->delta2));
    PetscCall(VecAXPY(sctx->delta2, -1, sctx->mean));
    PetscCall(VecPointwiseMult(sctx->delta2, sctx->delta2, sctx->delta));
    PetscCall(VecAXPY(sctx->M, 1., sctx->delta2));
  }
  PetscCall(VecDot(y, meas_vec, &qoi));
  qois[it] = qoi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TIME(functioncall, name, time) \
  do { \
    double _starttime, _endtime; \
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting %s... ", name)); \
    MPI_Barrier(MPI_COMM_WORLD); \
    _starttime = MPI_Wtime(); \
    PetscCall(functioncall); \
    MPI_Barrier(MPI_COMM_WORLD); \
    _endtime = MPI_Wtime(); \
    PetscCall(PetscPrintf(MPI_COMM_WORLD, " done. Took %.4fs.\n", _endtime - _starttime)); \
    *time = _endtime - _starttime; \
  } while (0);

int main(int argc, char *argv[])
{
  Parameters  params;
  Mat         A;
  DM          dm = nullptr; // Only set when building Mat with PETSc
  KSP         ksp;
  PC          pc;
  Vec         x, b, meas_vec;
  PetscRandom pr;
  double      time;
  PetscMPIInt rank;
  PetscBool   mfem                 = PETSC_FALSE;
  PetscBool   seed_from_dev_random = PETSC_FALSE;
  Problem    *problem;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(ParMGMCInitialize());

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "#############                Benchmark Test Program                #############\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

  PetscCall(ParametersCreate(&params));
  PetscCall(ParametersRead(params));

  PetscCheck(params->measure_iact || params->measure_sampling_time, MPI_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Pass at least one of -measure_sampling_time or -measure_iact");

  PetscCall(ParMGMCGetPetscRandom(&pr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-seed_from_dev_random", &seed_from_dev_random, nullptr));
  if (seed_from_dev_random) {
    int           dr = open("/dev/random", O_RDONLY);
    unsigned long seed;
    read(dr, &seed, sizeof(seed));
    close(dr);
    PetscCall(PetscRandomSetSeed(pr, seed));
  } else {
    PetscInt seed = 1;

    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-seed", &seed, nullptr));
    PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    PetscCall(PetscRandomSetSeed(pr, seed + rank));
  }
  PetscCall(PetscRandomSeed(pr));
  PetscCall(PetscRandomDestroy(&pr));

  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-mfem", &mfem, nullptr));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Starting assembly of operator... "));
  time = MPI_Wtime();
#ifdef PARMGMC_HAVE_MFEM
  if (mfem) {
    problem = new MFEMProblem(params);
  } else
#endif
    problem = new PetscProblem(params);
  time = MPI_Wtime() - time;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "done. Took %.4fs.\n", time));

  PetscCall(problem->GetPrecisionMat(&A));
  PetscCall(problem->GetRHSVec(&b));
  PetscCall(problem->GetMeasurementVec(&meas_vec));
  if (!mfem) PetscCall(problem->GetDM(&dm));
  PetscCall(VecDuplicate(b, &x));

  TIME(SamplerCreate(A, dm, params, &ksp), "Setup sampler", &time);
  PetscCall(KSPGetPC(ksp, &pc));

  if (params->measure_sampling_time) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                              Measure sampling time\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

    TIME(Burnin(ksp, b, params), "Burn-in", &time);
    TIME(Sample(ksp, b, params, x), "Sampling", &time);

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per sample [ms]: %.6f\n\n", time / params->n_samples * 1000));
  }

  if (params->measure_iact) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "                                  Measure IACT\n"));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "################################################################################\n"));

    auto *ctx = new SampleCtx(b, params->n_samples + 1, meas_vec, params->est_mean_and_var);

    TIME(Burnin(ksp, b, params), "Burn-in", &time);
    PetscCall(PCSetSampleCallback(pc, SaveSample, ctx, nullptr));
    TIME(Sample(ksp, b, params, x), "Sampling", &time);

    PetscBool    print_acf = PETSC_FALSE;
    PetscScalar *acf;
    PetscScalar  tau = 0;
    PetscBool    valid;

    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-print_acf", &print_acf, nullptr));
    PetscCall(IACT(params->n_samples, ctx->qois, &tau, print_acf ? &acf : nullptr, &valid));
    if (!valid) PetscCall(PetscPrintf(MPI_COMM_WORLD, "WARNING: Chain is too short to give reliable IACT estimate (need at least %d)\n", (int)ceil(500 * tau)));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "IACT: %.5f\n", tau));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Time per independent sample [ms]: %.6f\n\n", PetscMax(tau, 1) * time / params->n_samples * 1000));
    if (print_acf) {
      FILE *fptr;
      fptr = fopen("acf.txt", "w");
      for (PetscInt i = 0; i < params->n_samples; i++) PetscCall(PetscFPrintf(MPI_COMM_WORLD, fptr, "%.6f\n", acf[i]));
      fclose(fptr);
    }

    {
      Vec mean = nullptr, var = nullptr;

      PetscCall(ctx->GetMean(&mean));
      PetscCall(ctx->GetVar(&var));
      PetscCall(problem->VisualiseResults(x, mean, var));
    }

    delete ctx;
  }
  delete problem;

  PetscCall(InfoView(A, params, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PCViewFromOptions(pc, nullptr, "-view_sampler"));

  PetscCall(PetscRandomDestroy(&pr));
  PetscCall(ParametersDestroy(&params));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
}
