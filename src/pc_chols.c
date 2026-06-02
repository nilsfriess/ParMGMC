#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscblaslapack.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsftypes.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <mpi.h>

#if defined(PETSC_HAVE_MKL_PARDISO)
  #define PARMGMC_DEFAULT_SEQ_CHOLESKY MATSOLVERMKL_PARDISO
#else
  #define PARMGMC_DEFAULT_SEQ_CHOLESKY MATSOLVERPETSC
#endif

#if defined(PETSC_HAVE_MKL_CPARDISO)
  #define PARMGMC_DEFAULT_PAR_CHOLESKY MATSOLVERMKL_CPARDISO
#else
  #define PARMGMC_DEFAULT_PAR_CHOLESKY MATSOLVERPETSC
#endif

typedef struct {
  Vec           r, v, v_cache, xl, yl;
  Mat           F;
  PetscScalar  *dense_L;       /* Lower Cholesky factor (column-major, leading dimension dense_n)
                                  used for small sequential blocks, e.g. ASM patch smoothers. */
  PetscBLASInt  dense_n;       /* Size of the dense factor; 0 if the dense fast path is not used. */
  PetscInt      dense_threshold; /* Blocks of size <= this are factored and solved densely. */
  PetscRandom   prand;
  MatSolverType st;
  PetscBool     richardson, is_gamg_coarse; /* is_gamg_coarse should be set if the sampler is used as
                                               coarse grid sampler in GAMGMC. GAMG reduces the number
                                               of MPI ranks that participate on the coarser levels, down
								                               to 1 on the coarsest. However, it doesn't use sub-communicators
								                               for that but just leaves some values of the MPIAIJ matrices empty.
								                               It turns out that the Intel MKL C/Pardiso solver is horribly slow
								                               in this case, so what we do instead is extract the (sequential)
								                               matrix that contains the actual values and use a sequential
								                               sampler. This involves additional copies but scales much better.
								     */
  PetscBool     in_solve;
  PetscInt      sample_index;
  void         *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_CholSampler;

static PetscErrorCode PCCholSamplerNotifySample(PC pc, Vec y)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  if (chol->scb) PetscCall(chol->scb(chol->sample_index++, y, chol->cbctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  if (chol->del_scb) PetscCall(chol->del_scb(chol->cbctx));
  PetscCall(PetscRandomDestroy(&chol->prand));
  PetscCall(MatDestroy(&chol->F));
  PetscCall(PetscFree(chol->dense_L));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  PetscCall(VecDestroy(&chol->v_cache));
  PetscCall(VecDestroy(&chol->xl));
  PetscCall(VecDestroy(&chol->yl));
  PetscCall(PetscFree(chol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&chol->prand));
  PetscCall(MatDestroy(&chol->F));
  PetscCall(PetscFree(chol->dense_L));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  PetscCall(VecDestroy(&chol->v_cache));
  PetscCall(VecDestroy(&chol->xl));
  PetscCall(VecDestroy(&chol->yl));
  chol->dense_n      = 0;
  chol->sample_index = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;
  Mat            S, P;
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  PetscInt       N = 0;
  MatType        type;
  PetscBool      flag;
  IS             rowperm, colperm;
  MatFactorInfo  info;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)pc);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatFactorInfoInitialize(&info));
  if (!chol->prand) PetscCall(ParMGMCGetPetscRandom(&chol->prand));

  PetscCall(MatGetType(pc->pmat, &type));
  PetscCall(PetscStrcmp(type, MATLRC, &flag));
  if (flag) {
    Mat A, B, Bs, Bs_S, BSBt;
    Vec D;

    PetscCall(MatLRCGetMats(pc->pmat, &A, &B, &D, NULL));
    PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &Bs));
    PetscCall(MatDuplicate(Bs, MAT_COPY_VALUES, &Bs_S));

    { // Scatter D into a distributed vector
      PetscInt   sctsize;
      IS         sctis;
      Vec        Sd;
      VecScatter sct;

      PetscCall(VecGetSize(D, &sctsize));
      PetscCall(ISCreateStride(comm, sctsize, 0, 1, &sctis));
      PetscCall(MatCreateVecs(Bs_S, &Sd, NULL));
      PetscCall(VecScatterCreate(D, sctis, Sd, NULL, &sct));
      PetscCall(VecScatterBegin(sct, D, Sd, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sct, D, Sd, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterDestroy(&sct));
      PetscCall(ISDestroy(&sctis));
      D = Sd;
    }

    PetscCall(MatDiagonalScale(Bs_S, NULL, D));
    PetscCall(MatMatTransposeMult(Bs_S, Bs, MAT_INITIAL_MATRIX, PETSC_DECIDE, &BSBt));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &P));
    PetscCall(MatAXPY(P, 1., BSBt, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Bs));
    PetscCall(MatDestroy(&Bs_S));
    PetscCall(MatDestroy(&BSBt));
    PetscCall(VecDestroy(&D));
  } else {
    P = pc->pmat;
  }

  PetscCallMPI(MPI_Comm_size(comm, &size));

  // Set symmetric flag to allow conversion and help with factorization
  PetscCall(MatSetOption(P, MAT_SYMMETRIC, PETSC_TRUE));

  if (size != 1) {
    if (chol->is_gamg_coarse) PetscCall(MatMPIAIJGetSeqAIJ(P, &S, NULL, NULL));
    else PetscCall(MatConvert(P, MATSBAIJ, MAT_INITIAL_MATRIX, &S));
  } else {
    // For sequential, keep as AIJ but mark as symmetric
    S = P;
  }
  PetscCall(MatSetOption(S, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateVecs(S, &chol->r, &chol->v));
  PetscCall(VecDuplicate(chol->v, &chol->v_cache));

  if (size == 1 && chol->is_gamg_coarse == PETSC_FALSE) PetscCall(MatGetSize(S, &N, NULL));
  if (N > 0 && N <= chol->dense_threshold) {
    /* Small sequential block (e.g. an ASM star/patch smoother block): factor and
       solve densely.  These blocks are tiny and structurally near-dense, so a
       LAPACK Cholesky plus BLAS triangular solves avoids the sparse-factor
       indirection and is markedly faster than the sparse path. */
    Mat                D;
    const PetscScalar *darr;
    PetscBLASInt       n, linfo;

    PetscCall(MatConvert(S, MATSEQDENSE, MAT_INITIAL_MATRIX, &D));
    PetscCall(PetscBLASIntCast(N, &n));
    PetscCall(PetscMalloc1((size_t)N * N, &chol->dense_L));
    PetscCall(MatDenseGetArrayRead(D, &darr));
    PetscCall(PetscArraycpy(chol->dense_L, darr, (size_t)N * N));
    PetscCall(MatDenseRestoreArrayRead(D, &darr));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("L", &n, chol->dense_L, &n, &linfo));
    PetscCall(PetscFPTrapPop());
    PetscCheck(linfo == 0, comm, PETSC_ERR_MAT_CH_ZRPVT, "Dense Cholesky failed: leading minor of order %" PetscBLASInt_FMT " is not positive definite", linfo);
    chol->dense_n = n;
    PetscCall(MatDestroy(&D));
  } else {
    PetscCall(MatGetFactor(S, chol->st, MAT_FACTOR_CHOLESKY, &chol->F));
    if (size == 1 || chol->is_gamg_coarse) PetscCall(MatGetOrdering(S, MATORDERINGMETISND, &rowperm, &colperm));
    else PetscCall(MatGetOrdering(S, MATORDERINGEXTERNAL, &rowperm, &colperm));
    if (!chol->is_gamg_coarse || rank == 0) {
      PetscCall(MatCholeskyFactorSymbolic(chol->F, S, rowperm, &info));
      PetscCall(MatCholeskyFactorNumeric(chol->F, S, &info));
    }
    if (chol->is_gamg_coarse) {
      PetscCall(MatCreateVecs(chol->F, &chol->xl, NULL));
      PetscCall(MatCreateVecs(chol->F, &chol->yl, NULL));
    }
    PetscCall(ISDestroy(&rowperm));
    PetscCall(ISDestroy(&colperm));
  }
  if (size != 1 && !chol->is_gamg_coarse) PetscCall(MatDestroy(&S));
  if (flag) PetscCall(MatDestroy(&P));

  /* pc->setupcalled         = PETSC_TRUE; */
  /* pc->reusepreconditioner = PETSC_TRUE; */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solve L w = in for w (forward substitution).  Uses the dense LAPACK factor
   for small sequential blocks and the sparse factor otherwise. */
static PetscErrorCode CholSamplerForwardSolve(PC_CholSampler chol, Vec in, Vec out)
{
  PetscFunctionBeginUser;
  if (chol->dense_n) {
    const PetscScalar *ina;
    PetscScalar       *outa;
    PetscBLASInt       one = 1;

    PetscCall(VecGetArrayRead(in, &ina));
    PetscCall(VecGetArray(out, &outa));
    PetscCall(PetscArraycpy(outa, ina, chol->dense_n));
    PetscCallBLAS("BLAStrsv", BLAStrsv_("L", "N", "N", &chol->dense_n, chol->dense_L, &chol->dense_n, outa, &one));
    PetscCall(VecRestoreArray(out, &outa));
    PetscCall(VecRestoreArrayRead(in, &ina));
  } else {
    PetscCall(MatForwardSolve(chol->F, in, out));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solve L^T y = in for y (backward substitution), the transpose counterpart of
   CholSamplerForwardSolve(). */
static PetscErrorCode CholSamplerBackwardSolve(PC_CholSampler chol, Vec in, Vec out)
{
  PetscFunctionBeginUser;
  if (chol->dense_n) {
    const PetscScalar *ina;
    PetscScalar       *outa;
    PetscBLASInt       one = 1;

    PetscCall(VecGetArrayRead(in, &ina));
    PetscCall(VecGetArray(out, &outa));
    PetscCall(PetscArraycpy(outa, ina, chol->dense_n));
    PetscCallBLAS("BLAStrsv", BLAStrsv_("L", "T", "N", &chol->dense_n, chol->dense_L, &chol->dense_n, outa, &one));
    PetscCall(VecRestoreArray(out, &outa));
    PetscCall(VecRestoreArrayRead(in, &ina));
  } else {
    PetscCall(MatBackwardSolve(chol->F, in, out));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_CholSampler(PC pc, Vec x, Vec y)
{
  PC_CholSampler chol = pc->data;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCheck(chol->richardson || !chol->scb || chol->in_solve, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Setting a sample callback is only supported for Cholesky sampler during KSPSolve");

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));

  if (chol->is_gamg_coarse) {
    if (rank == 0) {
      PetscCall(VecGetLocalVectorRead(x, chol->xl));
      PetscCall(MatForwardSolve(chol->F, chol->xl, chol->v));
      PetscCall(VecRestoreLocalVectorRead(x, chol->xl));
      PetscCall(VecSetRandomStandardNormal(chol->r, chol->prand));
      PetscCall(VecAXPY(chol->v, 1., chol->r));
      PetscCall(VecGetLocalVector(y, chol->yl));
      PetscCall(MatBackwardSolve(chol->F, chol->v, chol->yl));
      PetscCall(VecRestoreLocalVector(y, chol->yl));
    }
  } else {
    PetscCall(CholSamplerForwardSolve(chol, x, chol->v));
    PetscCall(VecSetRandomStandardNormal(chol->r, chol->prand));
    PetscCall(VecAXPY(chol->v, 1., chol->r));
    PetscCall(CholSamplerBackwardSolve(chol, chol->v, y));
  }
  PetscCall(PCCholSamplerNotifySample(pc, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_CholSampler(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;
  (void)w;

  PC_CholSampler chol = pc->data;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  chol->richardson = PETSC_TRUE;
  if (its == 1) {
    PetscCall(PCApply_CholSampler(pc, b, y));
  } else {
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
    if (chol->is_gamg_coarse) {
      if (rank == 0) {
        PetscCall(VecGetLocalVectorRead(b, chol->xl));
        PetscCall(MatForwardSolve(chol->F, chol->xl, chol->v_cache));
        PetscCall(VecRestoreLocalVectorRead(b, chol->xl));
      }
    } else {
      PetscCall(CholSamplerForwardSolve(chol, b, chol->v_cache));
    }
    for (PetscInt it = 0; it < its; ++it) {
      if (chol->is_gamg_coarse) {
        if (rank == 0) {
          PetscCall(VecCopy(chol->v_cache, chol->v));
          PetscCall(VecSetRandomStandardNormal(chol->r, chol->prand));
          PetscCall(VecAXPY(chol->v, 1., chol->r));
          PetscCall(VecGetLocalVector(y, chol->yl));
          PetscCall(MatBackwardSolve(chol->F, chol->v, chol->yl));
          PetscCall(VecRestoreLocalVector(y, chol->yl));
        }
      } else {
        PetscCall(VecCopy(chol->v_cache, chol->v));
        PetscCall(VecSetRandomStandardNormal(chol->r, chol->prand));
        PetscCall(VecAXPY(chol->v, 1., chol->r));
        PetscCall(CholSamplerBackwardSolve(chol, chol->v, y));
      }
      PetscCall(PCCholSamplerNotifySample(pc, y));
    }
  }
  *outits          = its;
  *reason          = PCRICHARDSON_CONVERGED_ITS;
  chol->richardson = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_CholSampler(PC pc, KSP ksp, Vec b, Vec x)
{
  (void)ksp;
  (void)b;
  (void)x;

  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  chol->in_solve     = PETSC_TRUE;
  chol->sample_index = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_CholSampler(PC pc, KSP ksp, Vec b, Vec x)
{
  (void)ksp;
  (void)b;
  (void)x;

  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  chol->in_solve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_Cholsampler(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  if (chol->del_scb) PetscCall(chol->del_scb(chol->cbctx));
  chol->scb     = cb;
  chol->cbctx   = ctx;
  chol->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_CholSampler(PC pc, PetscViewer viewer)
{
  PC_CholSampler chol = pc->data;
  MatInfo        info;

  PetscFunctionBeginUser;
  if (chol && chol->dense_n) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Dense Cholesky factor for sequential block of size %" PetscBLASInt_FMT "\n", chol->dense_n));
  } else if (chol && chol->F) {
    PetscCall(MatGetInfo(chol->F, MAT_GLOBAL_SUM, &info));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Nonzeros in factored matrix: allocated %f\n", info.nz_allocated));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCholSamplerSetIsCoarseGAMG(PC pc, PetscBool flag)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  chol->is_gamg_coarse = flag;
  if (flag) chol->st = PARMGMC_DEFAULT_SEQ_CHOLESKY;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_CholSampler(PC pc, PetscOptionItems_ARG PetscOptionsObject)
{
  PC_CholSampler chol = pc->data;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Cholesky options");
  PetscCall(PetscOptionsBool("-pc_cholsampler_coarse_gamg", "Sampler is coarse GAMGMC sampler", NULL, flag, &flag, NULL));
  if (flag) PetscCall(PCCholSamplerSetIsCoarseGAMG(pc, PETSC_TRUE));
  PetscCall(PetscOptionsInt("-pc_cholsampler_dense_threshold", "Sequential blocks of size <= this are factored and solved densely", NULL, chol->dense_threshold, &chol->dense_threshold, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_CholSampler(PC pc)
{
  PC_CholSampler chol;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&chol));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size == 1) chol->st = PARMGMC_DEFAULT_SEQ_CHOLESKY;
  else chol->st = PARMGMC_DEFAULT_PAR_CHOLESKY;
  chol->dense_threshold = 64;

  pc->data                 = chol;
  pc->ops->destroy         = PCDestroy_CholSampler;
  pc->ops->reset           = PCReset_CholSampler;
  pc->ops->setup           = PCSetUp_CholSampler;
  pc->ops->apply           = PCApply_CholSampler;
  pc->ops->applyrichardson = PCApplyRichardson_CholSampler;
  pc->ops->view            = PCView_CholSampler;
  pc->ops->setfromoptions  = PCSetFromOptions_CholSampler;
  pc->ops->presolve        = PCPreSolve_CholSampler;
  pc->ops->postsolve       = PCPostSolve_CholSampler;
  chol->is_gamg_coarse     = PETSC_FALSE;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Cholsampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}
