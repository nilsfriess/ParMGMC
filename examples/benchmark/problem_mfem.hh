#pragma once

#include "params.hh"
#include "problems.hh"

#include <memory>
#include <mfem/fem/fe_coll.hpp>
#include <mfem/linalg/petsc.hpp>
#include <mfem/mfem.hpp>
#include <petscsys.h>
#include <petscsystypes.h>

class MFEMProblem : public Problem {
public:
  MFEMProblem(Parameters params)
  {
    mfem::Hypre::Init();

    MakeMesh();

    fec     = new mfem::H1_FECollection(1, mesh->Dimension());
    fespace = std::make_unique<mfem::ParFiniteElementSpace>(mesh, fec);
    if (mesh->bdr_attributes.Size()) {
      mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    PetscScalar kappa = 1;
    PetscCallVoid(PetscOptionsGetReal(nullptr, nullptr, "-matern_kappa", &kappa, nullptr));
    mfem::ConstantCoefficient kappa2(kappa * kappa);

    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
    a.AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
    a.SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    a.SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    a.Assemble(0);

    mfem::ParLinearForm       b(fespace.get());
    mfem::ConstantCoefficient zero(0);
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(zero));
    b.Assemble();

    mfem::ParGridFunction u(fespace.get());
    u = 0.0;

    mfem::Vector X;
    mfem::Vector F;
    a.FormLinearSystem(ess_tdof_list, u, b, A, X, F);
    B = std::make_unique<mfem::PetscParVector>(MPI_COMM_WORLD, F, true);

    PetscCallVoid(MatSetOption(A, MAT_SPD, PETSC_TRUE));

    PetscCallVoid(CreateMeasurementVec());

    if (params->with_lr) {
      PetscInt   nobs, cdim = mesh->Dimension(), nobs_given;
      PetscReal *obs_coords, *obs_radii, *obs_values, obs_sigma2 = 1e-4;
      PetscBool  flag = PETSC_FALSE;

      PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-nobs", &nobs, &flag));
      PetscCheckAbort(flag, MPI_COMM_WORLD, PETSC_ERR_ARG_NULL, "Must provide observations");

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

      std::vector<mfem::Vector> obs_coords_mv(nobs);
      for (int i = 0; i < nobs; ++i) obs_coords_mv[i].SetDataAndSize(&obs_coords[i * cdim], cdim);
      std::vector<double> radii_v(obs_radii, obs_radii + nobs);
      std::vector<double> obs_values_v(obs_values, obs_values + nobs);
      AddObservations(obs_sigma2, obs_coords_mv, radii_v, obs_values_v);
    }
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
    *v = *B;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode GetMeasurementVec(Vec *v) override
  {
    PetscFunctionBeginUser;
    *v = *meas_vec;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode VisualiseResults(Vec sample, Vec mean, Vec var) override
  {
    PetscFunctionBeginUser;
    mfem::ParGridFunction mean_gf(fespace.get());
    mean_gf.SetFromTrueDofs(mfem::PetscParVector{mean, true});
    mfem::ParGridFunction var_gf(fespace.get());
    var_gf.SetFromTrueDofs(mfem::PetscParVector{var, true});
    mfem::ParGridFunction sample_gf(fespace.get());
    sample_gf.SetFromTrueDofs(mfem::PetscParVector{sample, true});
    mfem::ParGridFunction meas_vec_gf(fespace.get());
    meas_vec_gf.SetFromTrueDofs(*meas_vec);

    PetscMPIInt rank;
    mfem::L2_FECollection pw_const_fec(0, mesh->Dimension());
    mfem::ParFiniteElementSpace pw_const_fes(mesh, &pw_const_fec);
    mfem::ParGridFunction mpi_rank_gf(&pw_const_fes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mpi_rank_gf = rank;

    mfem::ParaViewDataCollection pd("Results", mesh);
    pd.SetPrefixPath("ParaView");
    pd.RegisterField("mean", &mean_gf);
    pd.RegisterField("var", &var_gf);
    pd.RegisterField("sample", &sample_gf);
    pd.RegisterField("measurement vector", &meas_vec_gf);
    pd.RegisterField("rank", &mpi_rank_gf);

    pd.SetLevelsOfDetail(1);
    pd.SetDataFormat(mfem::VTKFormat::BINARY);
    pd.SetHighOrderOutput(true);
    pd.SetCycle(0);
    pd.SetTime(0.0);
    pd.Save();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void AddObservations(double sigma2, const std::vector<mfem::Vector> &coords, std::vector<double> radii, std::vector<double> obsvals)
  {
    mfem::PetscParVector g(A);
    PetscInt             lsize, gsize;
    PetscCallVoid(VecGetLocalSize(g, &lsize));
    PetscCallVoid(VecGetSize(g, &gsize));

    Mat _BM;
    PetscCallVoid(MatCreateDense(A.GetComm(), lsize, PETSC_DECIDE, gsize, radii.size(), nullptr, &_BM));
    mfem::PetscParMatrix BM(_BM, false);

    mfem::PetscParVector S(BM), f(BM, true), y(S);
    S = 1. / sigma2;

    mfem::PetscParVector  meas(A);
    mfem::ParGridFunction proj(fespace.get());
    for (std::size_t i = 0; i < coords.size(); ++i) {
      meas = 0;
      proj = 0;

      mfem::FunctionCoefficient obs([&](const mfem::Vector &coord) {
        if (coord.DistanceTo(coords[i]) < radii[i]) return 1 / VolumeOfSphere(radii[i]);
        else return 0.;
      });

      proj.ProjectCoefficient(obs);
      mfem::PetscParVector proj_vec(M);
      proj.GetTrueDofs(proj_vec);
      M.Mult(proj_vec, meas);

      Vec col;
      PetscCallVoid(MatDenseGetColumnVec(BM, i, &col));
      PetscCallVoid(VecCopy(meas, col));
      PetscCallVoid(MatDenseRestoreColumnVec(BM, i, &col));
      // y[i] = obsvals[i];
      PetscCallVoid(VecSetValue(y, i, obsvals[i], INSERT_VALUES));
    }
    PetscCallVoid(VecAssemblyBegin(y));
    PetscCallVoid(VecAssemblyEnd(y));

    PetscCallVoid(VecPointwiseMult(y, y, S));
    PetscCallVoid(MatMult(BM, y, f));

    Mat ALRC;
    Mat Ao = A.ReleaseMat(false);
    PetscCallVoid(MatCreateLRC(Ao, BM, S, nullptr, &ALRC));
    PetscCallVoid(PetscObjectDereference((PetscObject)Ao));

    A.SetMat(ALRC);
    PetscCallVoid(PetscObjectDereference((PetscObject)ALRC));
    *B = f;
  }

  ~MFEMProblem()
  {
    delete mesh;
    delete fec;
  }

private:
  PetscErrorCode CreateMeasurementVec()
  {
    PetscFunctionBeginUser;
    AssembleMassMatrix();
    meas_vec = std::make_unique<mfem::PetscParVector>(A);
    mfem::ParGridFunction proj(fespace.get());
    mfem::Vector          centre(mesh->Dimension()), start(mesh->Dimension()), end(mesh->Dimension());
    double                radius = 1;

    char      qoi_type[64] = "sphere";
    PetscBool valid_type, flag;
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-qoi_type", qoi_type, 64, nullptr));
    PetscCall(PetscStrcmpAny(qoi_type, &valid_type, "sphere", "rect", ""));
    PetscCheckAbort(valid_type, MPI_COMM_WORLD, PETSC_ERR_SUP, "-qoi_type must be sphere or rect");

    PetscCall(PetscStrcmp(qoi_type, "sphere", &flag));
    if (flag) {
      double  *centre_data;
      PetscInt got_dim = mesh->Dimension();
      PetscCall(PetscCalloc1(got_dim, &centre_data));
      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_centre", centre_data, &got_dim, nullptr));
      PetscCheckAbort(got_dim == 0 or got_dim == mesh->Dimension(), MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed, expected %d\n", mesh->Dimension());
      centre.SetData(centre_data);

      PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-qoi_radius", &radius, nullptr));
    } else {
      PetscInt got_dim = mesh->Dimension();
      double  *start_data, *end_data;

      PetscCall(PetscCalloc1(got_dim, &start_data));
      PetscCall(PetscCalloc1(got_dim, &end_data));
      for (PetscInt i = 0; i < got_dim; ++i) end_data[i] = 1;

      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_start", start_data, &got_dim, nullptr));
      PetscCheckAbort(got_dim == 0 or got_dim == mesh->Dimension(), MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for start, expected %d", mesh->Dimension());
      got_dim = mesh->Dimension();
      PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_end", end_data, &got_dim, nullptr));
      PetscCheckAbort(got_dim == 0 or got_dim == mesh->Dimension(), MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for end, expected %d", mesh->Dimension());

      start.SetData(start_data);
      end.SetData(end_data);
    }

    mfem::Vector              diff(mesh->Dimension());
    mfem::FunctionCoefficient obs([&](const mfem::Vector &coord) {
      if (flag) { // QOI type is sphere
        if (coord.DistanceTo(centre) < radius) return 1.;
        else return 0.;
      } else { // QOI type is rectangle
        for (int i = 0; i < mesh->Dimension(); ++i)
          if (!(coord[i] >= start[i] && coord[i] <= end[i])) return 0.;
        return 1.;
      }
    });

    proj.ProjectCoefficient(obs);
    mfem::PetscParVector proj_vec(*meas_vec);
    proj.GetTrueDofs(proj_vec);
    M.Mult(proj_vec, *meas_vec);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void MakeMesh()
  {
    char       mesh_file[256];
    PetscBool  flag;
    mfem::Mesh serial_mesh;
    PetscInt   refine = 0, parRefine = 0;

    PetscCallVoid(PetscOptionsGetString(nullptr, nullptr, "-mesh_file", mesh_file, 256, &flag));

    if (!flag) {
      PetscInt  faces_per_dim = 4;
      PetscInt  faces_per_rank = 0;
      PetscBool fpr_flag = PETSC_FALSE;

      PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-box_faces_per_rank", &faces_per_rank, &fpr_flag));
      if (fpr_flag) {
        PetscMPIInt nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        double sqrt_nprocs = std::sqrt((double)nprocs);
        if (std::round(sqrt_nprocs) * std::round(sqrt_nprocs) != nprocs)
          PetscCallVoid(PetscPrintf(MPI_COMM_WORLD, "Warning: number of processes (%d) is not a perfect square; grid size will be approximated as %dx%d elements\n", nprocs, (PetscInt)std::round(faces_per_rank * sqrt_nprocs), (PetscInt)std::round(faces_per_rank * sqrt_nprocs)));
        faces_per_dim = (PetscInt)std::round(faces_per_rank * sqrt_nprocs);
      } else {
        PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-box_faces", &faces_per_dim, nullptr));
      }
      serial_mesh = mfem::Mesh::MakeCartesian2D(faces_per_dim, faces_per_dim, mfem::Element::Type::TRIANGLE);
    } else {
      serial_mesh = mfem::Mesh::LoadFromFile(mesh_file);
    }
    PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-dm_refine", &refine, nullptr));
    for (PetscInt i = 0; i < refine; ++i) serial_mesh.UniformRefinement();

    mesh = new mfem::ParMesh(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    PetscCallVoid(PetscOptionsGetInt(nullptr, nullptr, "-dm_par_refine", &parRefine, nullptr));
    for (PetscInt i = 0; i < parRefine; ++i) mesh->UniformRefinement();
  }

  void AssembleMassMatrix()
  {
    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::MassIntegrator);
    a.SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    a.SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    a.Assemble();
    a.FormSystemMatrix(ess_tdof_list, M);
  }

  double VolumeOfSphere(double radius)
  {
    switch (fespace->GetParMesh()->Dimension()) {
    case 2:
      return PETSC_PI * radius * radius;
    case 3:
      return 4 * PETSC_PI * radius * radius * radius;
    default:
      return 0;
    }
  }

  mfem::Array<int>                             ess_tdof_list;
  mfem::PetscParMatrix                         A;
  mfem::PetscParMatrix                         M;
  std::unique_ptr<mfem::PetscParVector>        B;
  std::unique_ptr<mfem::PetscParVector>        meas_vec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fespace;

  mfem::ParMesh         *mesh;
  mfem::H1_FECollection *fec;
};
