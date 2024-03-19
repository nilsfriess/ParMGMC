#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <array>
#include <memory>
#include <random>

#include <mpi.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscsystypes.h>

using namespace parmgmc;

class ShiftedLaplaceFD {
public:
  ShiftedLaplaceFD(PetscInt coarseVerticesPerDim, PetscInt refineLevels, PetscReal kappainv = 1.,
                   bool colorMatrixWithDM = true) {
    PetscFunctionBeginUser;

    // Create coarse DM
    DM da;
    PetscCallVoid(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                               DMDA_STENCIL_STAR, coarseVerticesPerDim, coarseVerticesPerDim,
                               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &da));
    PetscCallVoid(DMSetUp(da));
    PetscCallVoid(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0));

    // Create hierarchy
    hierarchy = std::make_shared<DMHierarchy>(da, refineLevels, true);

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy->get_fine(), &mat));

    // TODO: Maybe not needed?
    PetscCallVoid(MatSetOption(mat, MAT_USE_INODES, PETSC_FALSE));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 5> cols;
    std::array<PetscReal, 5> vals;

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(hierarchy->get_fine(), &info));

    dirichletRows.reserve(4 * info.mx);
    double h2inv = 1. / ((info.mx - 1) * (info.mx - 1));
    const auto kappa2 = 1. / (kappainv * kappainv);

    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
      for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
        row.j = j;
        row.i = i;

        if ((i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)) {
          dirichletRows.push_back(j * info.my + i);
        } else {
          std::size_t k = 0;

          if (j != 0) {
            cols[k].j = j - 1;
            cols[k].i = i;
            vals[k] = -h2inv;
            ++k;
          }

          if (i != 0) {
            cols[k].j = j;
            cols[k].i = i - 1;
            vals[k] = -h2inv;
            ++k;
          }

          cols[k].j = j;
          cols[k].i = i;
          vals[k] = 4 * h2inv + kappa2;
          ++k;

          if (j != info.my - 1) {
            cols[k].j = j + 1;
            cols[k].i = i;
            vals[k] = -h2inv;
            ++k;
          }

          if (i != info.mx - 1) {
            cols[k].j = j;
            cols[k].i = i + 1;
            vals[k] = -h2inv;
            ++k;
          }

          PetscCallVoid(
              MatSetValuesStencil(mat, 1, &row, k, cols.data(), vals.data(), INSERT_VALUES));
        }
      }
    }

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    // Dirichlet rows are in natural ordering, convert to global using the DM's
    // ApplicationOrdering
    AO ao;
    PetscCallVoid(DMDAGetAO(hierarchy->get_fine(), &ao));
    PetscCallVoid(AOApplicationToPetsc(ao, dirichletRows.size(), dirichletRows.data()));

    PetscCallVoid(
        MatZeroRowsColumns(mat, dirichletRows.size(), dirichletRows.data(), 1., nullptr, nullptr));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    op = std::make_shared<LinearOperator>(mat, true);
    if (colorMatrixWithDM)
      op->color_matrix(da);
    else
      op->color_matrix();

    PetscFunctionReturnVoid();
  }

  const std::shared_ptr<LinearOperator> &getOperator() const { return op; }
  const std::shared_ptr<DMHierarchy> &getHierarchy() const { return hierarchy; }

  const std::vector<PetscInt> &getDirichletRows() const { return dirichletRows; }

  DM getCoarseDM() const { return hierarchy->get_coarse(); }
  DM getFineDM() const { return hierarchy->get_fine(); }

private:
  std::shared_ptr<LinearOperator> op;
  std::shared_ptr<DMHierarchy> hierarchy;
  // DM da;

  std::vector<PetscInt> dirichletRows;
};

struct TimingResult {
  TimingResult operator+=(const TimingResult &other) {
    setupTime += other.setupTime;
    sampleTime += other.sampleTime;
    return *this;
  }

  TimingResult operator/=(double d) {
    setupTime /= d;
    sampleTime /= d;
    return *this;
  }

  double setupTime = 0;
  double sampleTime = 0;
};

template <typename Engine>
PetscErrorCode testGibbsSampler(const ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                                PetscScalar omega, GibbsSweepType sweepType, bool fixRhs,
                                TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->get_mat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->get_mat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  PetscCall(fill_vec_rand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MulticolorGibbsSampler sampler(problem.getOperator(), &engine, omega, sweepType);

  if (fixRhs)
    sampler.setFixedRhs(rhs);

  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Engine>
PetscErrorCode testMGMCSampler(const ShiftedLaplaceFD &problem, PetscInt nSamples, Engine &engine,
                               const MGMCParameters &params, TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->get_mat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->get_mat(), problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(), 1., sample, rhs));

  PetscCall(fill_vec_rand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MultigridSampler sampler(problem.getOperator(), problem.getHierarchy(), &engine, params);
  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < nSamples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResult(const std::string &name, TimingResult timing) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n+++-------------------------------------------------"
                                        "-----------+++\n\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Name: %s\n", name.c_str()));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Timing [s]:\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Setup time:    %.4f\n", timing.setupTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Sampling time: %.4f\n", timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   -----------------------\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   Total:         %.4f\n",
                        timing.setupTime + timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n+++-------------------------------------------------"
                                        "-----------+++\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscFunctionBeginUser;

  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-size", &size, nullptr));
  PetscInt nSamples = 1000;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-samples", &nSamples, nullptr));
  PetscInt nRuns = 5;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-runs", &nRuns, nullptr));
  PetscInt nRefine = 3;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-refine", &nRefine, nullptr));

  PetscBool runGibbs = PETSC_FALSE, runMGMC = PETSC_FALSE, runCholesky = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-gibbs", &runGibbs, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-mgmc", &runMGMC, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-cholesky", &runCholesky, nullptr));

  PetscMPIInt mpisize;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &mpisize));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "####            Running strong scaling test suite           ######\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "##################################################"
                                        "################\n"));

  if (!(runGibbs || runMGMC || runCholesky)) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "No sampler selected, not running any tests.\n"
                                          "Pass at least one of\n"
                                          "     -gibbs     -mgmc     -cholesky\n"
                                          "to run the test with the respective sampler.\n"));
    return 0;
  }

  ShiftedLaplaceFD problem(size, nRefine);
  DMDALocalInfo fineInfo, coarseInfo;
  PetscCall(DMDAGetLocalInfo(problem.getFineDM(), &fineInfo));
  PetscCall(DMDAGetLocalInfo(problem.getCoarseDM(), &coarseInfo));

  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "Configuration: \n"
                        "\tMPI rank(s):           %d\n"
                        "\tProblem size (coarse): %dx%d = %d\n"
                        "\tProblem size (fine):   %dx%d = %d\n"
                        "\tLevels:                %d\n"
                        "\tSamples:               %d\n"
                        "\tRuns:                  %d\n",
                        mpisize, coarseInfo.mx, coarseInfo.mx, (coarseInfo.mx * coarseInfo.mx),
                        fineInfo.mx, fineInfo.mx, (fineInfo.mx * fineInfo.mx), nRefine, nSamples,
                        nRuns));

  std::mt19937 engine;

  if (runGibbs) {
    TimingResult avg;

    for (int i = 0; i < nRuns; ++i) {
      TimingResult timing;
      PetscCall(
          testGibbsSampler(problem, nSamples, engine, 1., GibbsSweepType::Forward, true, timing));

      avg += timing;
    }

    avg /= nRuns;

    PetscCall(printResult("Gibbs sampler, forward sweep, fixed rhs", avg));
  }

  if (runMGMC) {
    TimingResult avg;

    MGMCParameters params = MGMCParameters::Default();

    for (int i = 0; i < nRuns; ++i) {
      TimingResult timing;
      PetscCall(testMGMCSampler(problem, nSamples, engine, params, timing));

      avg += timing;
    }

    avg /= nRuns;

    PetscCall(printResult("MGMC sampler", avg));
  }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Forward,
  //                              false,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, forward sweep, nonfixed rhs", timing));
  // }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Symmetric,
  //                              true,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, symmetric sweep, nonfixed rhs", timing));
  // }
}
