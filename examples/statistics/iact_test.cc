#include "mat.hh"
#include "qoi.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/sample_chain.hh"

#include <mpi.h>
#include <pcg_random.hpp>

#include <petscdm.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include <memory>
#include <random>

using namespace parmgmc;

template <class Chain>
inline PetscErrorCode iact(const std::string &name, Chain &chain,
                           Vec sample_rhs) {
  PetscFunctionBeginUser;

  Vec initial_sample;
  PetscCall(VecDuplicate(sample_rhs, &initial_sample));

  for (std::size_t n = 0; n < chain.get_n_chains(); ++n) {
    PetscCall(VecSet(initial_sample, (n + 1) * 100));
    chain.set_sample(initial_sample, n);
  }
  PetscCall(VecDestroy(&initial_sample));

  PetscInt n_burnin = 100;
  PetscOptionsGetInt(NULL, NULL, "-n_burnin", &n_burnin, NULL);

  PetscCall(chain.sample(sample_rhs, n_burnin));
  chain.reset();

  PetscInt n_samples = 100;
  PetscOptionsGetInt(NULL, NULL, "-n_samples", &n_samples, NULL);

  PetscCall(chain.sample(sample_rhs, n_samples));

  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "%s IACT: %zu (has %sconverged, R = %f)\n",
                        name.c_str(),
                        chain.integrated_autocorr_time(),
                        chain.converged() ? "" : "not ",
                        chain.gelman_rubin()));

  PetscFunctionReturn(PETSC_SUCCESS);
}

struct Coordinate {
  PetscReal x;
  PetscReal y;
};

int main(int argc, char *argv[]) {
  PetscHelper helper(&argc, &argv);

  PetscFunctionBeginUser;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup DM hierarchy
  std::shared_ptr<DMHierarchy> dm_hierarchy;
  {
    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    int n_vertices = 5;
    PetscOptionsGetInt(NULL, NULL, "-n_vertices", &n_vertices, NULL);

    Coordinate lower_left{0, 0};
    Coordinate upper_right{1, 1};

    DM dm;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR,
                           n_vertices,
                           n_vertices,
                           PETSC_DECIDE,
                           PETSC_DECIDE,
                           dof_per_node,
                           stencil_width,
                           NULL,
                           NULL,
                           &dm));

    PetscCall(DMSetUp(dm));
    PetscCall(DMDASetUniformCoordinates(
        dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0));

    PetscInt n_levels = 4;
    PetscOptionsGetInt(NULL, NULL, "-n_levels", &n_levels, NULL);

    dm_hierarchy = std::make_shared<DMHierarchy>(dm, n_levels);
    // PetscCall(dm_hierarchy->print_info());
  }

  const int n_chains = 8;

  // Setup random number generator
  pcg32 engine;
  {
    int seed;
    if (rank == 0) {
      seed = std::random_device{}();
      PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL);
    }

    // Send seed to all other processes
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    engine.seed(seed);
    engine.set_stream(rank);
  }

  // RHS used in samplers
  Vec sample_rhs;
  PetscCall(DMCreateGlobalVector(dm_hierarchy->get_fine(), &sample_rhs));
  PetscCall(fill_vec_rand(sample_rhs, engine));

  NormQOI qoi;

  PetscInt n_smooth = 2;
  PetscOptionsGetInt(NULL, NULL, "-n_smooth", &n_smooth, NULL);

  MGMCParameters params;
  params.n_smooth = n_smooth;
  params.cycle_type = MGMCCycleType::V;
  params.smoothing_type = MGMCSmoothingType::Symmetric;

  // Setup Multigrid sampler (using rediscretisation on each level)
  {
    PetscCall(
        PetscPrintf(MPI_COMM_WORLD,
                    "Setting up multigrid sampler with rediscretisation..."));

    using Chain = SampleChain<MultigridSampler<pcg32>, NormQOI>;
    Chain chain(
        qoi, n_chains, sample_rhs, dm_hierarchy, assemble, &engine, params);

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "done.\n"));

    PetscCall(iact("MGMC (Rediscretisation)", chain, sample_rhs));
  }

  // Setup Multigrid sampler (using Galerkin product for coarse operators)
  {
    PetscCall(PetscPrintf(
        MPI_COMM_WORLD,
        "Setting up multigrid sampler with Galerkin projection..."));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dm_hierarchy->get_fine(), &mat));
    auto linear_operator = std::make_shared<LinearOperator>(mat);

    using Chain = SampleChain<MultigridSampler<pcg32>, NormQOI>;
    Chain chain(qoi,
                n_chains,
                sample_rhs,
                linear_operator,
                dm_hierarchy,
                &engine,
                params);

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "done.\n"));

    PetscCall(iact("MGMC (Galerkin)", chain, sample_rhs));
  }

  // Setup Gibbs sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up Gibbs sampler..."));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dm_hierarchy->get_fine(), &mat));
    auto linear_operator = std::make_shared<LinearOperator>(mat);

    PetscReal omega = 1.; // SOR parameter
    PetscOptionsGetReal(NULL, NULL, "-omega", &omega, NULL);

    using Chain = SampleChain<GibbsSampler<pcg32>, NormQOI>;
    Chain chain(qoi,
                n_chains,
                sample_rhs,
                linear_operator,
                &engine,
                omega,
                GibbsSweepType::Symmetric);
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "done.\n"));

    PetscCall(iact("Gibbs", chain, sample_rhs));
  }

  PetscCall(VecDestroy(&sample_rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
