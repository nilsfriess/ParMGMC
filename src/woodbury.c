#include "parmgmc/pc/woodbury.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>

typedef struct {
  PC solver;
  PC sampler;

  PetscRandom prand;

  Mat B, G;

  Vec wk, sqrtS, zn, swork;

  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
} *PC_Woodbury;

static PetscErrorCode PCWoodburyBuildLRCCorrection(PC pc)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;
  Mat         A, B, C, tmp, Id, Sb;
  KSP         ksp;
  Vec         S, x, Si;
  IS          sctis;
  VecScatter  sct;
  PetscInt    cols, sctsize;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCall(MatLRCGetMats(pc->pmat, &A, &B, &S, NULL));

  // Step 1: C = wb->solver(B), column by column
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &C));
  PetscCall(MatGetSize(B, NULL, &cols));
  PetscCall(MatCreateVecs(A, &x, NULL));
  for (PetscInt i = 0; i < cols; ++i) {
    Vec b, c;

    PetscCall(VecZeroEntries(x));
    PetscCall(MatDenseGetColumnVecRead(B, i, &b));
    PetscCall(PCApply(wb->solver, b, x));
    PetscCall(MatDenseRestoreColumnVecRead(B, i, &b));

    PetscCall(MatDenseGetColumnVecWrite(C, i, &c));
    PetscCall(VecCopy(x, c));
    PetscCall(MatDenseRestoreColumnVecWrite(C, i, &c));
  }
  PetscCall(VecDestroy(&x));

  // Step 2: form tmp = S^-1 + B^T C and invert (k x k).
  PetscCall(MatTransposeMatMult(B, C, MAT_INITIAL_MATRIX, 1, &tmp)); // tmp = B^T M_A^-1 B

  // Scatter S into a vec compatible with C's column layout.
  PetscCall(VecGetSize(S, &sctsize));
  PetscCall(ISCreateStride(comm, sctsize, 0, 1, &sctis));
  PetscCall(MatCreateVecs(C, &Si, NULL));
  PetscCall(VecScatterCreate(S, sctis, Si, NULL, &sct));
  PetscCall(VecScatterBegin(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sct, S, Si, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&sct));
  PetscCall(ISDestroy(&sctis));
  PetscCall(VecReciprocal(Si));

  PetscCall(MatDiagonalSet(tmp, Si, ADD_VALUES)); // tmp = S^-1 + B^T M_A^-1 B
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, tmp, tmp));
  PetscCall(MatDuplicate(tmp, MAT_DO_NOT_COPY_VALUES, &Id));
  PetscCall(MatShift(Id, 1));
  PetscCall(MatDuplicate(tmp, MAT_DO_NOT_COPY_VALUES, &Sb));
  PetscCall(KSPMatSolve(ksp, Id, Sb)); // Sb = (S^-1 + B^T M_A^-1 B)^-1

  PetscCall(MatMatMult(C, Sb, MAT_INITIAL_MATRIX, 1, &wb->G)); // G = C * Sb
  PetscCall(MatCreateVecs(wb->G, NULL, &wb->zn));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&Si));
  PetscCall(MatDestroy(&Id));
  PetscCall(MatDestroy(&Sb));
  PetscCall(MatDestroy(&tmp));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_Woodbury(PC pc)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&wb->prand));
  PetscCall(VecDestroy(&wb->wk));
  PetscCall(VecDestroy(&wb->sqrtS));
  PetscCall(VecDestroy(&wb->zn));
  PetscCall(VecDestroy(&wb->swork));
  PetscCall(MatDestroy(&wb->G));
  if (wb->solver) PetscCall(PCReset(wb->solver));
  if (wb->sampler) PetscCall(PCReset(wb->sampler));
  wb->B = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Woodbury(PC pc)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&wb->prand));
  PetscCall(VecDestroy(&wb->wk));
  PetscCall(VecDestroy(&wb->sqrtS));
  PetscCall(VecDestroy(&wb->zn));
  PetscCall(VecDestroy(&wb->swork));
  PetscCall(MatDestroy(&wb->G));
  PetscCall(PCDestroy(&wb->solver));
  PetscCall(PCDestroy(&wb->sampler));
  if (wb->del_scb) {
    PetscCall(wb->del_scb(wb->cbctx));
    wb->del_scb = NULL;
  }
  PetscCall(PetscFree(wb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_Woodbury(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;

  PetscFunctionBeginUser;
  if (wb->del_scb) {
    PetscCall(wb->del_scb(wb->cbctx));
    wb->del_scb = NULL;
  }
  wb->scb     = cb;
  wb->cbctx   = ctx;
  wb->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Woodbury(PC pc)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;
  MatType     mtype;
  PetscBool   is_lrc;
  Mat         A;
  Vec         S;

  PetscFunctionBegin;
  PetscCheck(wb->solver && wb->sampler, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Must provide sampler and solver");
  PetscCall(VecDestroy(&wb->wk));
  PetscCall(VecDestroy(&wb->sqrtS));
  PetscCall(VecDestroy(&wb->zn));
  PetscCall(VecDestroy(&wb->swork));
  PetscCall(MatDestroy(&wb->G));
  wb->B = NULL;
  if (!wb->prand) PetscCall(ParMGMCGetPetscRandom(&wb->prand));
  PetscCall(MatGetType(pc->pmat, &mtype));
  PetscCall(PetscStrcmp(mtype, MATLRC, &is_lrc));
  PetscCheck(is_lrc, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCWoodbury only supports matrices of type LRC");
  PetscCall(MatLRCGetMats(pc->pmat, &A, &wb->B, &S, NULL));
  PetscCall(MatCreateVecs(wb->B, &wb->wk, NULL));
  PetscCall(MatCreateVecs(A, &wb->swork, NULL));
  PetscCall(VecDuplicate(wb->wk, &wb->sqrtS));
  {
    const PetscScalar *Sarr;
    PetscScalar       *sqrtSarr;
    PetscInt           istart, iend;

    PetscCall(VecGetOwnershipRange(wb->sqrtS, &istart, &iend));
    PetscCall(VecGetArrayRead(S, &Sarr));
    PetscCall(VecGetArray(wb->sqrtS, &sqrtSarr));
    for (PetscInt i = istart; i < iend; ++i) sqrtSarr[i - istart] = Sarr[i];
    PetscCall(VecRestoreArrayRead(S, &Sarr));
    PetscCall(VecRestoreArray(wb->sqrtS, &sqrtSarr));
  }
  PetscCall(VecSqrtAbs(wb->sqrtS));
  PetscCall(PCSetOperators(wb->solver, A, A));
  PetscCall(PCSetOperators(wb->sampler, A, A));
  PetscCall(PCSetUp(wb->solver));
  PetscCall(PCSetUp(wb->sampler));
  PetscCall(PCWoodburyBuildLRCCorrection(pc));
  PetscCall(PCDestroy(&wb->solver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCWoodburySetSolver(PC pc, PC solver)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PCSetOptionsPrefix(solver, prefix));
  PetscCall(PCAppendOptionsPrefix(solver, "pc_woodbury_solver_"));
  PetscCall(PetscObjectReference((PetscObject)solver));
  wb->solver = solver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCWoodburySetSampler(PC pc, PC sampler)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PCSetOptionsPrefix(sampler, prefix));
  PetscCall(PCAppendOptionsPrefix(sampler, "pc_woodbury_sampler"));
  PetscCall(PetscObjectReference((PetscObject)sampler));
  wb->sampler = sampler;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCWoodbury_SetSolverType(PC pc, PCType type)
{
  PC solver;

  PetscFunctionBegin;
  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &solver));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)solver, (PetscObject)pc, 1));
  PetscCall(PCWoodburySetSolver(pc, solver));
  PetscCall(PCSetType(solver, type));
  PetscCall(PCDestroy(&solver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCWoodbury_SetSamplerType(PC pc, PCType type)
{
  PC sampler;

  PetscFunctionBegin;
  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &sampler));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)sampler, (PetscObject)pc, 1));
  PetscCall(PCWoodburySetSampler(pc, sampler));
  PetscCall(PCSetType(sampler, type));
  PetscCall(PCDestroy(&sampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Woodbury(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_Woodbury wb = (PC_Woodbury)pc->data;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Woodbury preconditioner/ sampler options");
  PetscCall(PetscOptionsString("-pc_woodbury_solver", "Solver for the Woodbury preconditioner", NULL, name, name, 256, &flg));
  if (flg) PetscCall(PCWoodbury_SetSolverType(pc, name));
  PetscCall(PetscOptionsString("-pc_woodbury_sampler", "Sampler for the Woodbury preconditioner", NULL, name, name, 256, &flg));
  if (flg) PetscCall(PCWoodbury_SetSamplerType(pc, name));
  PetscOptionsHeadEnd();
  if (wb->solver) PetscCall(PCSetFromOptions(wb->solver));
  if (wb->sampler) PetscCall(PCSetFromOptions(wb->sampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_Woodbury(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_Woodbury                 wb = (PC_Woodbury)pc->data;
  PetscInt                    sits;
  PCRichardsonConvergedReason sreason;

  PetscFunctionBegin;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(VecSetRandomStandardNormal(wb->wk, wb->prand));
    PetscCall(VecPointwiseMult(wb->wk, wb->wk, wb->sqrtS));
    PetscCall(MatMultAdd(wb->B, wb->wk, b, w));
    PetscCall(PCApplyRichardson(wb->sampler, w, y, wb->swork, 0., 0., 0., 1, PETSC_FALSE, &sits, &sreason));

    PetscCall(MatMultTranspose(wb->B, y, wb->wk));
    PetscCall(MatMult(wb->G, wb->wk, wb->zn));
    PetscCall(VecAXPY(y, -1., wb->zn));

    if (wb->scb) PetscCall(wb->scb(it, y, wb->cbctx));
  }
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Woodbury(PC pc)
{
  PC_Woodbury wb;

  PetscFunctionBegin;
  PetscCall(PetscNew(&wb));
  pc->data                 = wb;
  pc->ops->setup           = PCSetUp_Woodbury;
  pc->ops->reset           = PCReset_Woodbury;
  pc->ops->destroy         = PCDestroy_Woodbury;
  pc->ops->setfromoptions  = PCSetFromOptions_Woodbury;
  pc->ops->applyrichardson = PCApplyRichardson_Woodbury;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_Woodbury));
  PetscFunctionReturn(PETSC_SUCCESS);
}
