#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsftypes.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>
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
  Vec           r, v, xl, yl;
  Mat           F;
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
  void         *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_CholSampler;

static PetscErrorCode PCDestroy_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&chol->prand));
  PetscCall(MatDestroy(&chol->F));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  if (chol->is_gamg_coarse) {
    PetscCall(VecDestroy(&chol->xl));
    PetscCall(VecDestroy(&chol->yl));
  }
  PetscCall(PetscFree(chol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&chol->prand));
  PetscCall(MatDestroy(&chol->F));
  PetscCall(VecDestroy(&chol->r));
  PetscCall(VecDestroy(&chol->v));
  if (chol->is_gamg_coarse) {
    PetscCall(VecDestroy(&chol->xl));
    PetscCall(VecDestroy(&chol->yl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_CholSampler(PC pc)
{
  PC_CholSampler chol = pc->data;
  Mat            S, P;
  PetscMPIInt    size, rank, grank;
  MatType        type;
  PetscBool      flag;
  IS             rowperm, colperm;
  MatFactorInfo  info;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &grank));
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
      PetscCall(ISCreateStride(MPI_COMM_WORLD, sctsize, 0, 1, &sctis));
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

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  if (size != 1) {
    if (chol->is_gamg_coarse) PetscCall(MatMPIAIJGetSeqAIJ(P, &S, NULL, NULL));
    else PetscCall(MatConvert(P, MATSBAIJ, MAT_INITIAL_MATRIX, &S));
  } else {
    S = P;
  }
  PetscCall(MatSetOption(S, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateVecs(S, &chol->r, &chol->v));
  PetscCall(MatGetFactor(S, chol->st, MAT_FACTOR_CHOLESKY, &chol->F));

  if (size == 1 || chol->is_gamg_coarse) PetscCall(MatGetOrdering(S, MATORDERINGMETISND, &rowperm, &colperm));
  else PetscCall(MatGetOrdering(S, MATORDERINGEXTERNAL, &rowperm, &colperm));
  if (!chol->is_gamg_coarse || (chol->is_gamg_coarse && rank == 0)) {
    PetscCall(MatCholeskyFactorSymbolic(chol->F, S, rowperm, &info));
    PetscCall(MatCholeskyFactorNumeric(chol->F, S, &info));
  }
  if (chol->is_gamg_coarse) {
    PetscCall(MatCreateVecs(chol->F, &chol->xl, NULL));
    PetscCall(MatCreateVecs(chol->F, &chol->yl, NULL));
  }
  if (size != 1 && !chol->is_gamg_coarse) PetscCall(MatDestroy(&S));
  if (flag) PetscCall(MatDestroy(&P));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));

  /* pc->setupcalled         = PETSC_TRUE; */
  /* pc->reusepreconditioner = PETSC_TRUE; */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_CholSampler(PC pc, Vec x, Vec y)
{
  PC_CholSampler chol = pc->data;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCheck(chol->richardson || !chol->scb, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Setting a sample callback is not supported for Cholesky sampler in PREONLY mode. Use KSPRICHARDSON instead");

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));

  if (chol->is_gamg_coarse) {
    PetscCall(VecGetLocalVectorRead(x, chol->xl));
    if (rank == 0) PetscCall(MatForwardSolve(chol->F, chol->xl, chol->v));
    PetscCall(VecRestoreLocalVectorRead(x, chol->xl));
    PetscCall(VecSetRandom(chol->r, chol->prand));
    PetscCall(VecAXPY(chol->v, 1., chol->r));
    PetscCall(VecGetLocalVector(y, chol->yl));
    if (rank == 0) PetscCall(MatBackwardSolve(chol->F, chol->v, chol->yl));
    PetscCall(VecRestoreLocalVector(y, chol->yl));
  } else {
    PetscCall(MatForwardSolve(chol->F, x, chol->v));
    PetscCall(VecSetRandom(chol->r, chol->prand));
    PetscCall(VecAXPY(chol->v, 1., chol->r));
    PetscCall(MatBackwardSolve(chol->F, chol->v, y));
  }
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

  PetscFunctionBeginUser;
  chol->richardson = PETSC_TRUE;
  for (PetscInt it = 0; it < its; ++it) {
    if (chol->scb) PetscCall(chol->scb(it, y, chol->cbctx));
    PetscCall(PCApply_CholSampler(pc, b, y));
  }
  if (chol->scb) PetscCall(chol->scb(its, y, chol->cbctx));
  *outits          = its;
  *reason          = PCRICHARDSON_CONVERGED_ITS;
  chol->richardson = PETSC_FALSE;
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
  PetscCall(MatGetInfo(chol->F, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Nonzeros in factored matrix: allocated %f\n", info.nz_allocated));
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

static PetscErrorCode PCSetFromOptions_CholSampler(PC pc, PetscOptionItems PetscOptionsObject)
{
  PetscBool flag = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Cholesky options");
  PetscCall(PetscOptionsBool("-pc_cholsampler_coarse_gamg", "Sampler is coarse GAMGMC sampler", NULL, flag, &flag, NULL));
  if (flag) PetscCall(PCCholSamplerSetIsCoarseGAMG(pc, PETSC_TRUE));
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

  pc->data                 = chol;
  pc->ops->destroy         = PCDestroy_CholSampler;
  pc->ops->reset           = PCReset_CholSampler;
  pc->ops->setup           = PCSetUp_CholSampler;
  pc->ops->apply           = PCApply_CholSampler;
  pc->ops->applyrichardson = PCApplyRichardson_CholSampler;
  pc->ops->view            = PCView_CholSampler;
  pc->ops->setfromoptions  = PCSetFromOptions_CholSampler;
  chol->is_gamg_coarse     = PETSC_FALSE;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Cholsampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}
