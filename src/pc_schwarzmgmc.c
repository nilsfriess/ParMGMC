/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/** @file pc_schwarzmgmc.c
    @brief Two-level Multigrid Monte Carlo sampler with an overlapping Schwarz (block Gibbs) sampler
    on the fine level and a user-provided coarse space.

    A thin wrapper around PCGAMGMC: two levels (PCMG), the interpolation is the coarse basis Z
    (PCSchwarzMGMCSetCoarseSpace()), the coarse operator is the Galerkin product Z^T A Z (sampled
    exactly with PCCHOLSAMPLER on rank 0), and the fine-level sampler is PCSCHWARZGIBBS, on the
    subdomains given with PCSchwarzMGMCSetSubdomains() or on algebraic ones. One application is the
    cycle: Schwarz-Gibbs sweep, coarse sample, and (with up_sweeps 1) a second sweep. The second
    sweep is not needed for the sampler to be exact, so the default is V(1,0).

    # Options database keys
    - `-pc_schwarzmgmc_up_sweeps <0|1>` - sweeps after the coarse step (default 0)
    - `-pc_schwarzmgmc_deterministic` - no noise on either level: the deterministic two-level
      multiplicative Schwarz cycle (for convergence-rate estimates)
    - The fine-level sampler is configured with the prefix `gamgmc_mg_levels_`
      (e.g. `-gamgmc_mg_levels_pc_schwarzgibbs_overlap`), the coarse one with `gamgmc_mg_coarse_`.

    Use with `-ksp_type richardson`; each KSP iteration produces one sample.
*/

#include "parmgmc/pc/pc_schwarzmgmc.h"
#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/pc/pc_schwarzgibbs.h"

#include <petsc/private/pcimpl.h>
#include <petscksp.h>

typedef struct {
  PC        gamgmc; /* inner PCGAMGMC (PCMG with two levels) */
  Mat       Z;      /* coarse basis, rows x coarse functions */
  PetscInt  nsub;
  IS       *subs;
  PetscBool subs_set;
  PetscInt  up_sweeps;
  PetscBool deterministic;
} *PC_SchwarzMGMC;

/* Up "smoother" of the V(1,0) cycle: leaves the iterate unchanged without computing a residual. */
static PetscErrorCode NoopRichardson(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)pc;
  (void)b;
  (void)r;
  (void)rtol;
  (void)abstol;
  (void)dtol;

  PetscFunctionBegin;
  if (guesszero) PetscCall(VecZeroEntries(x));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set -<prefix><name> to value unless the user has set it. */
static PetscErrorCode SetDefaultOption(const char *prefix, const char *name, const char *value)
{
  PetscBool set;
  char      opt[512];

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(NULL, prefix, name, &set));
  if (!set) {
    PetscCall(PetscSNPrintf(opt, sizeof(opt), "-%s%s", prefix ? prefix : "", name + 1));
    PetscCall(PetscOptionsSetValue(NULL, opt, value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The coarse Cholesky sampler factorises on rank 0, so rank 0 must own all coarse rows, i.e. all
   columns of Z. Returns Z itself (new reference) or a copy with that column layout. */
static PetscErrorCode CoarseColumnsOnRankZero(Mat Z, Mat *Z0)
{
  PetscMPIInt rank;
  PetscInt    N, cs, ce, rs, re;
  int         ok, allok;
  IS          isrow, iscol;
  MPI_Comm    comm = PetscObjectComm((PetscObject)Z);

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetSize(Z, NULL, &N));
  PetscCall(MatGetOwnershipRangeColumn(Z, &cs, &ce));
  ok = (rank == 0) ? (ce - cs == N) : (ce == cs);
  PetscCallMPI(MPIU_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, comm));
  if (allok) {
    PetscCall(PetscObjectReference((PetscObject)Z));
    *Z0 = Z;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatGetOwnershipRange(Z, &rs, &re));
  PetscCall(ISCreateStride(comm, re - rs, rs, 1, &isrow));
  PetscCall(ISCreateStride(comm, rank == 0 ? N : 0, 0, 1, &iscol));
  PetscCall(MatCreateSubMatrix(Z, isrow, iscol, MAT_INITIAL_MATRIX, Z0));
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_SchwarzMGMC(PC pc)
{
  PC_SchwarzMGMC sm = pc->data;
  const char    *prefix;
  char           mgprefix[256], upprefix[256];
  PC             mg, sp;
  KSP            ks;
  Mat            Z;
  int            set, anyset;

  PetscFunctionBegin;
  if (pc->setupcalled) { /* new operator values: refactor everything */
    PetscCall(PCSetOperators(sm->gamgmc, pc->mat, pc->pmat));
    PetscCall(PCSetUp(sm->gamgmc));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(sm->Z, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Set the coarse space with PCSchwarzMGMCSetCoarseSpace() first");
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PetscSNPrintf(mgprefix, sizeof(mgprefix), "%sgamgmc_", prefix ? prefix : ""));
  PetscCall(PetscSNPrintf(upprefix, sizeof(upprefix), "%sschwarzmgmc_up_", prefix ? prefix : ""));

  /* Two-level PCMG inside PCGAMGMC; the prefix must be set before the levels exist */
  PetscCall(PCSetOptionsPrefix(sm->gamgmc, prefix));
  PetscCall(PCGAMGMCSetMGType(sm->gamgmc, PCMG));
  PetscCall(PCGAMGMCGetInternalPC(sm->gamgmc, &mg));
  PetscCall(PCSetOptionsPrefix(mg, mgprefix));
  PetscCall(PCSetType(mg, PCMG));
  PetscCall(PCMGSetLevels(mg, 2, NULL));
  PetscCall(CoarseColumnsOnRankZero(sm->Z, &Z));
  PetscCall(PCMGSetInterpolation(mg, 1, Z));
  PetscCall(MatDestroy(&Z));

  /* Fine level: Schwarz-Gibbs sampler, on the user's subdomains if any rank gave some */
  PetscCall(PCMGGetSmoother(mg, 1, &ks));
  PetscCall(KSPSetType(ks, KSPRICHARDSON));
  PetscCall(KSPGetPC(ks, &sp));
  PetscCall(PCSetType(sp, PCSCHWARZGIBBS));
  set = sm->subs_set ? 1 : 0;
  PetscCallMPI(MPIU_Allreduce(&set, &anyset, 1, MPI_INT, MPI_MAX, PetscObjectComm((PetscObject)pc)));
  if (anyset) PetscCall(PCSchwarzGibbsSetSubdomains(sp, sm->nsub, sm->subs));

  /* PCSetUp_GAMGMC sets its defaults through the options database; keep ours */
  PetscCall(SetDefaultOption(mgprefix, "-mg_levels_pc_type", PCSCHWARZGIBBS));
  if (sm->deterministic) {
    PetscCall(SetDefaultOption(mgprefix, "-mg_levels_pc_schwarzgibbs_deterministic", "1"));
    PetscCall(SetDefaultOption(mgprefix, "-mg_coarse_pc_cholsampler_deterministic", "1"));
  }

  /* V(1,0): an up "smoother" that does nothing (own prefix, so the level options don't apply) */
  if (sm->up_sweeps == 0) {
    KSP ku;
    PC  pu;

    PetscCall(PCMGGetSmootherUp(mg, 1, &ku));
    PetscCall(KSPSetOptionsPrefix(ku, upprefix));
    PetscCall(KSPSetType(ku, KSPRICHARDSON));
    PetscCall(KSPSetTolerances(ku, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1));
    PetscCall(KSPGetPC(ku, &pu));
    PetscCall(PCSetType(pu, PCSHELL));
    PetscCall(PCShellSetApplyRichardson(pu, NoopRichardson));
  }

  PetscCall(PCSetOperators(sm->gamgmc, pc->mat, pc->pmat));
  PetscCall(PCSetUp(sm->gamgmc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_SchwarzMGMC(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_SchwarzMGMC sm = pc->data;

  PetscFunctionBegin;
  PetscCall(PCApplyRichardson(sm->gamgmc, b, y, w, rtol, abstol, dtol, its, guesszero, outits, reason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_SchwarzMGMC(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_SchwarzMGMC sm = pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PCSCHWARZMGMC options");
  PetscCall(PetscOptionsRangeInt("-pc_schwarzmgmc_up_sweeps", "Fine-level sweeps after the coarse step", NULL, sm->up_sweeps, &sm->up_sweeps, NULL, 0, 1));
  PetscCall(PetscOptionsBool("-pc_schwarzmgmc_deterministic", "No noise: deterministic two-level Schwarz cycle", NULL, sm->deterministic, &sm->deterministic, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_SchwarzMGMC(PC pc, PetscViewer viewer)
{
  PC_SchwarzMGMC sm = pc->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "  two-level MGMC, V(1,%" PetscInt_FMT ") cycle%s\n", sm->up_sweeps, sm->deterministic ? ", deterministic" : ""));
  PetscCall(PCView(sm->gamgmc, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_SchwarzMGMC(PC pc)
{
  PC_SchwarzMGMC sm = pc->data;

  PetscFunctionBegin;
  PetscCall(PCDestroy(&sm->gamgmc));
  PetscCall(MatDestroy(&sm->Z));
  for (PetscInt i = 0; i < sm->nsub; ++i) PetscCall(ISDestroy(&sm->subs[i]));
  PetscCall(PetscFree(sm->subs));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSchwarzMGMCCast(PC pc, PC_SchwarzMGMC *sm)
{
  PetscBool same;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCSCHWARZMGMC, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC is not of type %s", PCSCHWARZMGMC);
  *sm = pc->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSchwarzMGMCSetCoarseSpace - Sets the coarse basis Z (fine rows x coarse functions, same row
  layout as the operator). The coarse operator is Z^T A Z.
@*/
PetscErrorCode PCSchwarzMGMCSetCoarseSpace(PC pc, Mat Z)
{
  PC_SchwarzMGMC sm = NULL;

  PetscFunctionBegin;
  PetscCall(PCSchwarzMGMCCast(pc, &sm));
  PetscCall(PetscObjectReference((PetscObject)Z));
  PetscCall(MatDestroy(&sm->Z));
  sm->Z = Z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSchwarzMGMCSetSubdomains - Sets the (overlapping) fine-level subdomains owned by this rank, as
  global row indices. Optional; without it PCSCHWARZGIBBS builds algebraic subdomains.
@*/
PetscErrorCode PCSchwarzMGMCSetSubdomains(PC pc, PetscInt n, const IS is[])
{
  PC_SchwarzMGMC sm = NULL;

  PetscFunctionBegin;
  PetscCall(PCSchwarzMGMCCast(pc, &sm));
  for (PetscInt i = 0; i < sm->nsub; ++i) PetscCall(ISDestroy(&sm->subs[i]));
  PetscCall(PetscFree(sm->subs));
  PetscCall(PetscMalloc1(n, &sm->subs));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCall(PetscObjectReference((PetscObject)is[i]));
    sm->subs[i] = is[i];
  }
  sm->nsub     = n;
  sm->subs_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSchwarzMGMCGetMG - Returns the internal two-level PCMG (borrowed reference), e.g. to inspect
  the fine-level sampler (PCMGGetSmoother(mg, 1, ...)) or the coarse operator.
@*/
PetscErrorCode PCSchwarzMGMCGetMG(PC pc, PC *mg)
{
  PC_SchwarzMGMC sm = NULL;

  PetscFunctionBegin;
  PetscCall(PCSchwarzMGMCCast(pc, &sm));
  PetscCall(PCGAMGMCGetInternalPC(sm->gamgmc, mg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_SchwarzMGMC(PC pc)
{
  PC_SchwarzMGMC sm = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sm));
  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &sm->gamgmc));
  PetscCall(PCSetType(sm->gamgmc, PCGAMGMC));
  sm->up_sweeps = 0;
  pc->data      = sm;

  pc->ops->setup           = PCSetUp_SchwarzMGMC;
  pc->ops->applyrichardson = PCApplyRichardson_SchwarzMGMC;
  pc->ops->setfromoptions  = PCSetFromOptions_SchwarzMGMC;
  pc->ops->view            = PCView_SchwarzMGMC;
  pc->ops->destroy         = PCDestroy_SchwarzMGMC;
  PetscFunctionReturn(PETSC_SUCCESS);
}
