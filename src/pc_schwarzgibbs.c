/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/** @file pc_schwarzgibbs.c
    @brief Parallel multiplicative overlapping Schwarz (block Gibbs) sampler.

    The subdomains are built algebraically, as in PCASM with `-pc_asm_local_blocks`: the local
    rows of each rank are split into `n` blocks by graph partitioning (PCASMCreateSubdomains) and
    then extended by `k` layers of overlap through the matrix graph (MatIncreaseOverlap).

    The subdomains are coloured such that two subdomains of the same colour neither overlap nor
    are coupled through A: the conflict graph is C = B^T |A| B, where B is the (rows x subdomains)
    incidence matrix and |A| has the sparsity pattern of A with all entries 1. Subdomains of the
    same colour are then conditionally independent, and updating them simultaneously is exactly a
    sequential block Gibbs sweep. One sweep visits the colours in order; for colour c it computes
    the residual r = b - A x and, on every subdomain S of colour c, adds
    A_SS^{-1} r_S + L_S^{-T} xi (an exact sample of the conditional, PCCHOLSAMPLER on A_SS) to x.
    With `-pc_schwarzgibbs_deterministic` the noise is omitted (exact local solves, PCCHOLESKY),
    which gives the multiplicative overlapping Schwarz smoother.

    Unlike PCASM (additive across ranks), this is multiplicative across ranks, hence an exact
    sampler in parallel.

    # Options database keys
    - `-pc_schwarzgibbs_local_blocks <n>` - number of subdomains per rank (default 1)
    - `-pc_schwarzgibbs_overlap <k>` - layers of overlap (default 1)
    - `-pc_schwarzgibbs_deterministic` - no noise (deterministic smoother)
    - The subdomain PCs use the prefix `sub_` (e.g. `-sub_pc_factor_mat_solver_type`).
*/

#include "parmgmc/pc/pc_schwarzgibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>

static PetscLogEvent SCHWARZGIBBS_SETUP = 0, SCHWARZGIBBS_SWEEP = 0;

typedef struct {
  /* Options */
  PetscInt  nblocks, overlap;
  PetscBool deterministic;

  /* Set up in PCSetUp */
  PetscInt   n;       /* number of local subdomains */
  IS        *is;      /* overlapping subdomains (global indices, sorted) */
  Mat       *submats; /* A(is_i, is_i) */
  PC        *subpc;   /* one sampler (or solver) per subdomain */
  PetscInt  *colors;  /* colour of each local subdomain */
  PetscInt   ncolors; /* global number of colours */
  PetscInt  *offs;    /* subdomain i occupies [offs[i], offs[i+1]) of the local work vectors */
  VecScatter sct;     /* global vector -> concatenated local subdomains */
  Vec        lr, ld;  /* local work vectors: restricted residual, local corrections */
  Vec       *xs, *ys; /* per-subdomain views (VecPlaceArray) */
  Vec        r, d;    /* global work vectors */
} *PC_SchwarzGibbs;

static PetscErrorCode PCSchwarzGibbsResetSetup(PC_SchwarzGibbs sg)
{
  PetscFunctionBegin;
  if (sg->n > 0) {
    for (PetscInt i = 0; i < sg->n; ++i) {
      PetscCall(PCDestroy(&sg->subpc[i]));
      PetscCall(VecDestroy(&sg->xs[i]));
      PetscCall(VecDestroy(&sg->ys[i]));
    }
    PetscCall(MatDestroySubMatrices(sg->n, &sg->submats));
  }
  if (sg->is) PetscCall(PCASMDestroySubdomains(sg->n, &sg->is, NULL));
  PetscCall(PetscFree(sg->subpc));
  PetscCall(PetscFree3(sg->offs, sg->xs, sg->ys));
  PetscCall(PetscFree(sg->colors));
  PetscCall(VecScatterDestroy(&sg->sct));
  PetscCall(VecDestroy(&sg->lr));
  PetscCall(VecDestroy(&sg->ld));
  PetscCall(VecDestroy(&sg->r));
  PetscCall(VecDestroy(&sg->d));
  sg->n       = 0;
  sg->ncolors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_SchwarzGibbs(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCSchwarzGibbsResetSetup(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_SchwarzGibbs(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCSchwarzGibbsResetSetup(pc->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSchwarzGibbsGetSubdomains_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A copy of A with all stored entries set to one (same sparsity pattern, no cancellation in products). */
static PetscErrorCode MatCreatePatternOnes(Mat A, Mat *P)
{
  Mat          blocks[2] = {NULL, NULL};
  PetscBool    ismpi, isseq;
  PetscInt     nb = 0;
  PetscScalar *v;
  MatInfo      info;

  PetscFunctionBegin;
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, P));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)*P, MATMPIAIJ, &ismpi));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)*P, MATSEQAIJ, &isseq));
  PetscCheck(ismpi || isseq, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "PCSCHWARZGIBBS requires an AIJ matrix");
  if (ismpi) {
    PetscCall(MatMPIAIJGetSeqAIJ(*P, &blocks[0], &blocks[1], NULL));
    nb = 2;
  } else {
    blocks[0] = *P;
    nb        = 1;
  }
  for (PetscInt k = 0; k < nb; ++k) {
    PetscCall(MatGetInfo(blocks[k], MAT_LOCAL, &info));
    PetscCall(MatSeqAIJGetArray(blocks[k], &v));
    for (PetscInt i = 0; i < (PetscInt)info.nz_used; ++i) v[i] = 1.0;
    PetscCall(MatSeqAIJRestoreArray(blocks[k], &v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Colour the subdomains: C = B^T |A| B with B the (rows x subdomains) incidence matrix. */
static PetscErrorCode PCSchwarzGibbsColor(PC pc, Mat A)
{
  PC_SchwarzGibbs sg = pc->data;
  MPI_Comm        comm;
  PetscInt        first = 0, nsub, m, M, *nnz, *onnz, cstart, cend, maxcolor = -1;
  Mat             Bt, B, Aone, C;
  MatColoring     mc;
  ISColoring      isc;
  PetscInt        ncol_isc, nloc;
  const ISColoringValue *cv;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCallMPI(MPI_Scan(&sg->n, &first, 1, MPIU_INT, MPI_SUM, comm));
  first -= sg->n;
  PetscCallMPI(MPIU_Allreduce(&sg->n, &nsub, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, &M, NULL));

  /* B^T: one row per subdomain, ones in the columns of its DOFs (all rows local) */
  PetscCall(PetscMalloc2(sg->n, &nnz, sg->n, &onnz));
  PetscCall(MatGetOwnershipRangeColumn(A, &cstart, &cend));
  for (PetscInt i = 0; i < sg->n; ++i) {
    const PetscInt *idx;
    PetscInt        ni;

    PetscCall(ISGetLocalSize(sg->is[i], &ni));
    PetscCall(ISGetIndices(sg->is[i], &idx));
    nnz[i] = 0;
    for (PetscInt j = 0; j < ni; ++j)
      if (idx[j] >= cstart && idx[j] < cend) nnz[i]++;
    onnz[i] = ni - nnz[i];
    PetscCall(ISRestoreIndices(sg->is[i], &idx));
  }
  PetscCall(MatCreate(comm, &Bt));
  PetscCall(MatSetSizes(Bt, sg->n, m, nsub, M));
  PetscCall(MatSetType(Bt, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Bt, 0, nnz));
  PetscCall(MatMPIAIJSetPreallocation(Bt, 0, nnz, 0, onnz));
  for (PetscInt i = 0; i < sg->n; ++i) nnz[i] += onnz[i]; /* total entries per row, used below */
  for (PetscInt i = 0; i < sg->n; ++i) {
    const PetscInt *idx;
    PetscScalar    *ones;
    PetscInt        row = first + i;

    PetscCall(PetscMalloc1(nnz[i], &ones));
    for (PetscInt j = 0; j < nnz[i]; ++j) ones[j] = 1.0;
    PetscCall(ISGetIndices(sg->is[i], &idx));
    PetscCall(MatSetValues(Bt, 1, &row, nnz[i], idx, ones, INSERT_VALUES));
    PetscCall(ISRestoreIndices(sg->is[i], &idx));
    PetscCall(PetscFree(ones));
  }
  PetscCall(PetscFree2(nnz, onnz));
  PetscCall(MatAssemblyBegin(Bt, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Bt, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTranspose(Bt, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatDestroy(&Bt));

  PetscCall(MatCreatePatternOnes(A, &Aone));
  PetscCall(MatPtAP(Aone, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatDestroy(&Aone));
  PetscCall(MatDestroy(&B));

  PetscCall(MatViewFromOptions(C, (PetscObject)pc, "-pc_schwarzgibbs_conflict_view"));
  PetscCall(MatColoringCreate(C, &mc));
  PetscCall(MatColoringSetDistance(mc, 1));
  PetscCall(MatColoringSetType(mc, MATCOLORINGJP));
  PetscCall(MatColoringApply(mc, &isc));
  PetscCall(ISColoringViewFromOptions(isc, (PetscObject)pc, "-pc_schwarzgibbs_coloring_view"));
  /* Same-colour subdomains must neither overlap nor be coupled; this is what makes the sampler exact. */
  PetscCall(MatColoringTest(mc, isc));
  PetscCall(ISColoringGetColors(isc, &nloc, &ncol_isc, &cv));
  PetscCheck(nloc == sg->n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Colouring has %" PetscInt_FMT " local entries, expected %" PetscInt_FMT, nloc, sg->n);
  PetscCall(PetscMalloc1(sg->n, &sg->colors));
  for (PetscInt i = 0; i < sg->n; ++i) {
    sg->colors[i] = (PetscInt)cv[i];
    maxcolor      = PetscMax(maxcolor, sg->colors[i]);
  }
  PetscCallMPI(MPIU_Allreduce(&maxcolor, &sg->ncolors, 1, MPIU_INT, MPI_MAX, comm));
  sg->ncolors += 1;
  PetscCall(ISColoringDestroy(&isc));
  PetscCall(MatColoringDestroy(&mc));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_SchwarzGibbs(PC pc)
{
  PC_SchwarzGibbs sg = pc->data;
  Mat             A  = pc->pmat;
  MPI_Comm        comm;
  PetscInt        m, ntot = 0;
  IS              isall;
  const char     *prefix;
  PetscReal       cmin;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCall(PetscLogEventBegin(SCHWARZGIBBS_SETUP, pc, 0, 0, 0));
  PetscCall(PCSchwarzGibbsResetSetup(sg));

  /* Algebraic subdomains: graph partitioning of the local rows, then overlap */
  PetscCall(MatGetLocalSize(A, &m, NULL));
  sg->n = PetscMin(sg->nblocks, m);
  if (sg->n > 0) PetscCall(PCASMCreateSubdomains(A, sg->n, &sg->is));
  {
    IS dummy = NULL; /* ranks without rows still take part in the collective calls */

    PetscCall(MatIncreaseOverlap(A, sg->n, sg->n ? sg->is : &dummy, sg->overlap));
    for (PetscInt i = 0; i < sg->n; ++i) PetscCall(ISSort(sg->is[i]));
    PetscCall(MatCreateSubMatrices(A, sg->n, sg->n ? sg->is : &dummy, sg->n ? sg->is : &dummy, MAT_INITIAL_MATRIX, &sg->submats));
  }

  /* One exact sampler (or solver) per subdomain */
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PetscMalloc1(sg->n, &sg->subpc));
  PetscCall(PetscMalloc3(sg->n + 1, &sg->offs, sg->n, &sg->xs, sg->n, &sg->ys));
  sg->offs[0] = 0;
  for (PetscInt i = 0; i < sg->n; ++i) {
    PetscInt ni;

    PetscCall(MatSetOption(sg->submats[i], MAT_SPD, PETSC_TRUE));
    PetscCall(PCCreate(PETSC_COMM_SELF, &sg->subpc[i]));
    PetscCall(PCSetOptionsPrefix(sg->subpc[i], prefix));
    PetscCall(PCAppendOptionsPrefix(sg->subpc[i], "sub_"));
    PetscCall(PCSetType(sg->subpc[i], sg->deterministic ? PCCHOLESKY : PCCHOLSAMPLER));
    PetscCall(PCSetOperators(sg->subpc[i], sg->submats[i], sg->submats[i]));
    PetscCall(PCSetFromOptions(sg->subpc[i]));
    PetscCall(PCSetUp(sg->subpc[i]));
    PetscCall(ISGetLocalSize(sg->is[i], &ni));
    sg->offs[i + 1] = sg->offs[i] + ni;
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, ni, NULL, &sg->xs[i]));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, ni, NULL, &sg->ys[i]));
  }
  ntot = sg->offs[sg->n];

  /* Scatter from the global vector to all local subdomains (one contiguous range each) */
  PetscCall(MatCreateVecs(A, &sg->r, &sg->d));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ntot, &sg->lr));
  PetscCall(VecDuplicate(sg->lr, &sg->ld));
  PetscCall(ISConcatenate(PETSC_COMM_SELF, sg->n, sg->is, &isall));
  PetscCall(VecScatterCreate(sg->r, isall, sg->lr, NULL, &sg->sct));
  PetscCall(ISDestroy(&isall));

  /* Coverage: every row must lie in a subdomain, otherwise the chain is not ergodic */
  PetscCall(VecSet(sg->ld, 1.0));
  PetscCall(VecZeroEntries(sg->d));
  PetscCall(VecScatterBegin(sg->sct, sg->ld, sg->d, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(sg->sct, sg->ld, sg->d, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecMin(sg->d, NULL, &cmin));
  PetscCheck(cmin >= 1, comm, PETSC_ERR_PLIB, "Some rows are not covered by any subdomain");

  PetscCall(PCSchwarzGibbsColor(pc, A));
  PetscCall(PetscLogEventEnd(SCHWARZGIBBS_SETUP, pc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One forward sweep over the colours, in place on x, right-hand side b. */
static PetscErrorCode PCSchwarzGibbsSweep(PC pc, Vec b, Vec x)
{
  PC_SchwarzGibbs sg = pc->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(SCHWARZGIBBS_SWEEP, pc, 0, 0, 0));
  for (PetscInt c = 0; c < sg->ncolors; ++c) {
    const PetscScalar *lr;
    PetscScalar       *ld;

    PetscCall(MatMult(pc->pmat, x, sg->r));
    PetscCall(VecAYPX(sg->r, -1.0, b));
    PetscCall(VecScatterBegin(sg->sct, sg->r, sg->lr, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sg->sct, sg->r, sg->lr, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecZeroEntries(sg->ld));
    PetscCall(VecGetArrayRead(sg->lr, &lr));
    PetscCall(VecGetArray(sg->ld, &ld));
    for (PetscInt i = 0; i < sg->n; ++i) {
      if (sg->colors[i] != c) continue;
      PetscCall(VecPlaceArray(sg->xs[i], lr + sg->offs[i]));
      PetscCall(VecPlaceArray(sg->ys[i], ld + sg->offs[i]));
      PetscCall(PCApply(sg->subpc[i], sg->xs[i], sg->ys[i]));
      PetscCall(VecResetArray(sg->xs[i]));
      PetscCall(VecResetArray(sg->ys[i]));
    }
    PetscCall(VecRestoreArray(sg->ld, &ld));
    PetscCall(VecRestoreArrayRead(sg->lr, &lr));
    PetscCall(VecZeroEntries(sg->d));
    PetscCall(VecScatterBegin(sg->sct, sg->ld, sg->d, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(sg->sct, sg->ld, sg->d, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecAXPY(x, 1.0, sg->d));
  }
  PetscCall(PetscLogEventEnd(SCHWARZGIBBS_SWEEP, pc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_SchwarzGibbs(PC pc, Vec b, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(y));
  PetscCall(PCSchwarzGibbsSweep(pc, b, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_SchwarzGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)w;
  (void)rtol;
  (void)abstol;
  (void)dtol;

  PetscFunctionBegin;
  if (guesszero) PetscCall(VecZeroEntries(y));
  for (PetscInt it = 0; it < its; ++it) PetscCall(PCSchwarzGibbsSweep(pc, b, y));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_SchwarzGibbs(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_SchwarzGibbs sg = pc->data;
  PetscInt        nb = sg->nblocks, ov = sg->overlap;
  PetscBool       det = sg->deterministic;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Schwarz Gibbs options");
  PetscCall(PetscOptionsInt("-pc_schwarzgibbs_local_blocks", "Number of subdomains per rank", NULL, nb, &nb, NULL));
  PetscCall(PetscOptionsInt("-pc_schwarzgibbs_overlap", "Layers of overlap", NULL, ov, &ov, NULL));
  PetscCall(PetscOptionsBool("-pc_schwarzgibbs_deterministic", "Deterministic smoother (no noise)", NULL, det, &det, NULL));
  PetscOptionsHeadEnd();
  PetscCheck(nb >= 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Need at least one block per rank");
  PetscCheck(ov >= 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Overlap must be non-negative");
  if (nb != sg->nblocks || ov != sg->overlap || det != sg->deterministic) pc->setupcalled = PETSC_FALSE;
  sg->nblocks       = nb;
  sg->overlap       = ov;
  sg->deterministic = det;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_SchwarzGibbs(PC pc, PetscViewer viewer)
{
  PC_SchwarzGibbs sg = pc->data;
  PetscBool       isascii;
  MPI_Comm        comm;
  PetscInt        loc[3] = {sg->n, PETSC_INT_MAX, 0}, nmin, nmax, smin, smax, ntot;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  for (PetscInt i = 0; i < sg->n; ++i) {
    loc[1] = PetscMin(loc[1], sg->offs[i + 1] - sg->offs[i]);
    loc[2] = PetscMax(loc[2], sg->offs[i + 1] - sg->offs[i]);
  }
  PetscCallMPI(MPIU_Allreduce(&loc[0], &ntot, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(&loc[0], &nmin, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&loc[0], &nmax, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&loc[1], &smin, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&loc[2], &smax, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  %" PetscInt_FMT " subdomains (%" PetscInt_FMT "-%" PetscInt_FMT " per rank, requested %" PetscInt_FMT "), overlap %" PetscInt_FMT ", sizes %" PetscInt_FMT "-%" PetscInt_FMT ", %" PetscInt_FMT " colours%s\n", ntot, nmin, nmax, sg->nblocks, sg->overlap, smin, smax, sg->ncolors, sg->deterministic ? ", deterministic (no noise)" : ""));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSchwarzGibbsGetSubdomains_SchwarzGibbs(PC pc, PetscInt *n, const IS *is[], const PetscInt *colors[])
{
  PC_SchwarzGibbs sg = pc->data;

  PetscFunctionBegin;
  PetscCheck(pc->setupcalled, PetscObjectComm((PetscObject)pc), PETSC_ERR_ORDER, "Must call PCSetUp() first");
  if (n) *n = sg->n;
  if (is) *is = sg->is;
  if (colors) *colors = sg->colors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSchwarzGibbsGetSubdomains - Returns the local overlapping subdomains and their colours.

  Not Collective

  Input Parameter:
. pc - the PC (after PCSetUp())

  Output Parameters:
+ n      - number of local subdomains
. is     - the subdomains (global indices, sorted); owned by the PC
- colors - the colour of each subdomain; owned by the PC
@*/
PetscErrorCode PCSchwarzGibbsGetSubdomains(PC pc, PetscInt *n, const IS *is[], const PetscInt *colors[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCSchwarzGibbsGetSubdomains_C", (PC, PetscInt *, const IS *[], const PetscInt *[]), (pc, n, is, colors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_SchwarzGibbs(PC pc)
{
  PC_SchwarzGibbs sg;

  PetscFunctionBegin;
  if (!SCHWARZGIBBS_SWEEP) {
    PetscCall(PetscLogEventRegister("SchwarzGibbsSweep", PARMGMC_CLASSID, &SCHWARZGIBBS_SWEEP));
    PetscCall(PetscLogEventRegister("SchwarzGibbsSetUp", PARMGMC_CLASSID, &SCHWARZGIBBS_SETUP));
  }
  PetscCall(PetscNew(&sg));
  sg->nblocks       = 1;
  sg->overlap       = 1;
  sg->deterministic = PETSC_FALSE;
  pc->data          = sg;

  pc->ops->setup           = PCSetUp_SchwarzGibbs;
  pc->ops->reset           = PCReset_SchwarzGibbs;
  pc->ops->destroy         = PCDestroy_SchwarzGibbs;
  pc->ops->apply           = PCApply_SchwarzGibbs;
  pc->ops->applyrichardson = PCApplyRichardson_SchwarzGibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_SchwarzGibbs;
  pc->ops->view            = PCView_SchwarzGibbs;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSchwarzGibbsGetSubdomains_C", PCSchwarzGibbsGetSubdomains_SchwarzGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
