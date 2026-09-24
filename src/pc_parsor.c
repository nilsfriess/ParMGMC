/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/pc/pc_parsor.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsftypes.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petsc/private/hashmapi.h>

typedef struct {
  PetscScalar data;
  PetscInt    id;
} MidIDData;

static PetscErrorCode MatGetDiagonalPointers_SeqAIJ(Mat A, PetscInt **diagptrs)
{
  PetscInt        rows;
  const PetscInt *i, *j;
  PetscReal      *a;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &rows, NULL));
  PetscCall(PetscMalloc1(rows, diagptrs));
  PetscCall(MatSeqAIJGetCSRAndMemType(A, &i, &j, &a, NULL));

  for (PetscInt row = 0; row < rows; ++row) {
    (*diagptrs)[row] = i[row + 1];
    for (PetscInt k = i[row]; k < i[row + 1]; ++k) {
      if (j[k] == row) {
        (*diagptrs)[row] = k;
        break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LocalMatInvertDiagonalForSOR(Mat A, PetscScalar omega, PetscScalar fshift, PetscInt **diag_out, Vec *idiag_out)
{
  PetscInt         i, m;
  const MatScalar *v;
  PetscScalar     *idiag_arr;
  PetscInt        *diag;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, NULL));
  PetscCall(MatGetDiagonalPointers_SeqAIJ(A, &diag));

  if (!*idiag_out) PetscCall(MatCreateVecs(A, NULL, idiag_out));

  PetscCall(VecGetArray(*idiag_out, &idiag_arr));
  PetscCall(MatSeqAIJGetArrayRead(A, &v));

  if (omega == 1.0 && PetscRealPart(fshift) <= 0.0) {
    for (i = 0; i < m; i++) {
      if (!PetscAbsScalar(v[diag[i]])) {
        PetscCheck(PetscRealPart(fshift), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Zero diagonal on row %" PetscInt_FMT, i);
        PetscCall(PetscInfo(A, "Zero diagonal on row %" PetscInt_FMT "\n", i));
      }
      idiag_arr[i] = 1.0 / v[diag[i]];
    }
    PetscCall(PetscLogFlops(m));
  } else {
    for (i = 0; i < m; i++) { idiag_arr[i] = omega / (fshift + v[diag[i]]); }
    PetscCall(PetscLogFlops(2.0 * m));
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &v));
  PetscCall(VecRestoreArray(*idiag_out, &idiag_arr));
  *diag_out = diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void SparseDenseMinusDot(PetscScalar *sum, const PetscScalar *x, const MatScalar *v, const PetscInt *idx, PetscInt n)
{
  for (PetscInt i = 0; i < n; i++) *sum -= v[i] * x[idx[i]];
}

typedef struct {
  PetscInt *proccols;

  VecScatter topsct;
  VecScatter botsct;
  IS         top;
  IS         bot;
  IS         mid;
  IS         int1, int2;

  PetscInt *mid_ready; /* stack of MID nodes whose dependencies are all satisfied */

  PetscInt      n_mid_recv_nbs;
  PetscInt      n_mid_send_nbs;
  PetscMPIInt  *mid_recv_nbs;
  PetscMPIInt  *mid_send_nbs;
  MPI_Request  *mid_recv_reqs;
  MPI_Request  *send_reqs; /* all sends of one sweep, completed at its end */
  PetscInt      n_send_reqs_max;
  PetscInt     *mid_recv_buf_size;  /* number of ghost values received from a neighbour per sweep */
  PetscInt     *mid_recv_remaining; /* ... of which still outstanding in the current sweep */
  MidIDData   **mid_recv_bufs;
  MidIDData   **mid_send_bufs;
  PetscInt     *mid_node_n_deps;
  PetscInt     *mid_node_n_send_to;
  PetscMPIInt **mid_node_send_to_nb;
  PetscInt     *mid_local_dep_count;
  PetscInt    **mid_local_deps;

  PetscHMapI global_to_lvec;
  PetscInt   n_lvec_cols;
  PetscInt  *lvec_to_mid_count;
  PetscInt **lvec_to_mid_nodes;

  Vec xx;
  Vec lvec;

  /* Cached apply-time data (computed once in setup) */
  PetscInt        *mid_send_start, *mid_send_cursor; /* per rank: first unsent and next free buffer entry */
  PetscInt        *mid_dep_left;
  const PetscInt  *arowptr, *acolind, *browptr, *bcolind, *colmap;
  const MatScalar *aa, *ba;
  const PetscInt  *midnodes;
  PetscInt         nmid;
  PetscInt         rstart;
  PetscMPIInt      rank;
  PetscInt         size;
  MPI_Comm         comm;

  PetscMPIInt tag;

  /* Debug statistics (-pc_parsor_stats) */
  PetscInt       stat_ncolors, stat_ntop, stat_nbot, stat_nint;
  PetscInt       stat_bot_mid_edges; /* couplings of a local BOT node to a MID node on a higher-colour rank */
  PetscInt       stat_napply, stat_nrounds, stat_nscans;
  PetscLogDouble stat_ttotal, stat_tphase[10]; /* see ParallelSORPhaseNames */

  PetscBool debug_no_mid_wait; /* INCORRECT, for timing only: MID nodes use old remote values, no MID messages */
} ParallelSORData;

/* Phases of one sweep, timed with -pc_parsor_stats */
enum {
  PH_TOPSCATTER,
  PH_TOP,
  PH_COPY,
  PH_INT1,
  PH_BOTSCATTER,
  PH_MID,
  PH_INT2,
  PH_BOTWAIT,
  PH_BOT,
  PH_SENDWAIT,
  PH_N
};
static const char *const ParallelSORPhaseNames[PH_N] = {"zero + TOP scatter", "TOP sweep", "VecCopy + BOT scatter begin", "INT1 sweep", "BOT scatter end (wait)", "MID loop", "INT2 sweep", "wait for BOT values", "BOT sweep", "wait for sends"};

typedef struct {
  ParallelSORData *parsor_data;
  PetscReal        omega;
  PetscInt         its;
  Vec              idiag_vec;
  PetscInt        *diag;
  PetscBool        stats;
  PetscBool        debug_no_mid_wait;
} *PC_PARSOR;

static PetscErrorCode CreateGhostCommunication(Mat matin, Vec *lvec_out, VecScatter *mvctx_out)
{
  Mat             Ad, Ao;
  const PetscInt *colmap;
  PetscInt        ncols;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(matin, &Ad, &Ao, &colmap));
  PetscCall(MatGetSize(Ao, NULL, &ncols));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ncols, lvec_out));

  {
    Vec       xcol;
    IS        from_is, to_is;
    PetscInt *from_indices, *to_indices;

    PetscCall(MatCreateVecs(matin, &xcol, NULL));
    PetscCall(PetscMalloc1(ncols, &from_indices));
    PetscCall(PetscMalloc1(ncols, &to_indices));

    for (PetscInt i = 0; i < ncols; i++) {
      from_indices[i] = colmap[i];
      to_indices[i]   = i;
    }

    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), ncols, from_indices, PETSC_OWN_POINTER, &from_is));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncols, to_indices, PETSC_OWN_POINTER, &to_is));
    PetscCall(VecScatterCreate(xcol, from_is, *lvec_out, to_is, mvctx_out));
    PetscCall(ISDestroy(&from_is));
    PetscCall(ISDestroy(&to_is));
    PetscCall(VecDestroy(&xcol));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ColorProcessors(Mat matin, PetscInt **proccols)
{
  PetscMPIInt     rank, size;
  PetscInt       *procmap, n, N, n_nb_total = 0, *proc_cols;
  PetscScalar    *proc_cols_vals;
  Mat             Ap, Ao, P;
  const PetscInt *colmap, *ii, *jj;
  PetscLayout     layout;
  MatColoring     proc_coloring;
  ISColoring      isc;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)matin), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)matin), &rank));
  PetscCall(PetscCalloc1(size, &procmap));

  PetscCall(MatMPIAIJGetSeqAIJ(matin, &Ap, &Ao, &colmap));
  PetscCall(MatGetSize(Ao, &n, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetLayouts(matin, NULL, &layout));
  PetscCall(MatGetSize(matin, &N, NULL));

  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      procmap[owner] = 1;
    }
  }

  for (PetscInt i = 0; i < size; ++i)
    if (procmap[i] > 0) n_nb_total++;

  PetscCall(PetscCalloc1(n_nb_total, &proc_cols));
  PetscCall(PetscCalloc1(n_nb_total, &proc_cols_vals));
  for (PetscInt i = 0, j = 0; i < size; ++i) {
    if (procmap[i] > 0) {
      proc_cols[j]      = i;
      proc_cols_vals[j] = 1;
      ++j;
    }
  }

  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)matin), 1, 1, size, size, 0, NULL, n_nb_total, NULL, &P));
  {
    const PetscInt row = rank;
    PetscCall(MatSetValues(P, 1, &row, n_nb_total, proc_cols, proc_cols_vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatColoringCreate(P, &proc_coloring));
  PetscCall(MatColoringSetDistance(proc_coloring, 1));
  PetscCall(MatColoringSetType(proc_coloring, MATCOLORINGJP));
  PetscCall(MatColoringApply(proc_coloring, &isc));
  PetscCall(ISColoringSetType(isc, IS_COLORING_GLOBAL));
  PetscCall(ISColoringViewFromOptions(isc, NULL, "-pc_parsor_proc_coloring_view"));
  PetscCall(PetscCalloc1(size, proccols));
  {
    IS      *iss;
    PetscInt ncols;

    PetscCall(ISColoringGetIS(isc, PETSC_USE_POINTER, &ncols, &iss));
    for (PetscInt c = 0; c < ncols; ++c) {
      const PetscInt *idxs;
      IS              gis;

      PetscCall(ISAllGather(iss[c], &gis));
      PetscCall(ISGetSize(gis, &n));
      PetscCall(ISGetIndices(gis, &idxs));
      for (PetscInt j = 0; j < n; ++j) (*proccols)[idxs[j]] = c;
      PetscCall(ISRestoreIndices(gis, &idxs));
      PetscCall(ISDestroy(&gis));
    }
    PetscCall(ISColoringRestoreIS(isc, PETSC_USE_POINTER, &iss));
  }
  PetscCall(ISColoringDestroy(&isc));
  PetscCall(MatColoringDestroy(&proc_coloring));
  PetscCall(PetscFree(proc_cols));
  PetscCall(PetscFree(proc_cols_vals));
  PetscCall(PetscFree(procmap));
  PetscCall(MatDestroy(&P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Scatter of the ghost values coupled to the given local rows, each ghost value sent once */
static PetscErrorCode CreateGhostSubScatter(Mat matin, Vec xcol, Vec lvec, PetscInt n1, const PetscInt rows1[], PetscInt n2, const PetscInt rows2[], VecScatter *sct)
{
  Mat             Ao;
  const PetscInt *colmap, *ii, *jj;
  PetscInt        ncols, cnt = 0, *from, *to;
  PetscBool      *used;
  IS              ix, iy;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(matin, NULL, &Ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetSize(Ao, NULL, &ncols));
  PetscCall(PetscCalloc1(ncols, &used));
  for (PetscInt k = 0; k < n1 + n2; ++k) {
    const PetscInt row = k < n1 ? rows1[k] : rows2[k - n1];
    for (PetscInt j = ii[row]; j < ii[row + 1]; ++j) used[jj[j]] = PETSC_TRUE;
  }
  for (PetscInt c = 0; c < ncols; ++c) cnt += used[c];
  PetscCall(PetscMalloc1(cnt, &from));
  PetscCall(PetscMalloc1(cnt, &to));
  cnt = 0;
  for (PetscInt c = 0; c < ncols; ++c) {
    if (!used[c]) continue;
    from[cnt] = colmap[c];
    to[cnt]   = c;
    ++cnt;
  }
  PetscCall(PetscFree(used));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), cnt, from, PETSC_OWN_POINTER, &ix));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cnt, to, PETSC_OWN_POINTER, &iy));
  PetscCall(VecScatterCreate(xcol, ix, lvec, iy, sct));
  PetscCall(ISDestroy(&ix));
  PetscCall(ISDestroy(&iy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParallelSORPartitionNodes(Mat matin, ParallelSORData *parsor)
{
  Mat                Ad, Ao;
  PetscLayout        layout;
  Vec                xcol;
  const PetscInt    *colmap, *ii, *jj, *ai, *aj;
  PetscMPIInt        rank, size;
  PetscInt           n, ncols, nrows, cnt = 0, ntop = 0, nbot = 0, nmid = 0, nint = 0, intcnt = 0, topcnt = 0, botcnt = 0, midcnt = 0, intcost = 0, topcost = 0, botcost = 0, tgt_int1_cost, curr_int1_cost = 0, splitidx;
  PetscBool         *needed;
  PetscInt          *nodes, *topnodes, *botnodes, *midnodes, *intnodes, *n_mid_recv_buf_size, *mid_send_ranks, *cnt_arr, *row_to_mid, *lvec_cursor, *local_cursor;
  PetscScalar       *xarr;
  const PetscScalar *larr;
  enum {
    INT,
    TOP,
    MID,
    BOT
  };

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)matin), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)matin), &size));
  PetscCall(MatGetLayouts(matin, NULL, &layout));
  PetscCall(MatMPIAIJGetSeqAIJ(matin, &Ad, &Ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetSize(Ao, &n, NULL));

  /* Create ghost communication infrastructure */
  Vec        lvec;
  VecScatter Mvctx;
  PetscCall(CreateGhostCommunication(matin, &lvec, &Mvctx));
  parsor->lvec = lvec;

  PetscCall(PetscCalloc1(n, &nodes));
  for (PetscInt i = 0; i < n; ++i) {
    PetscBool istop = PETSC_FALSE, isbot = PETSC_FALSE;
    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      if (parsor->proccols[owner] < parsor->proccols[rank]) istop = PETSC_TRUE;
      if (parsor->proccols[owner] > parsor->proccols[rank]) isbot = PETSC_TRUE;
    }

    if (!istop && !isbot) {
      nint++;
      nodes[i] = INT;
    } else if (!istop && isbot) {
      nbot++;
      nodes[i] = BOT;
    } else if (istop && !isbot) {
      ntop++;
      nodes[i] = TOP;
    } else {
      nmid++;
      nodes[i] = MID;
    }
  }

  PetscCall(PetscMalloc1(ntop, &topnodes));
  PetscCall(PetscMalloc1(nbot, &botnodes));
  PetscCall(PetscMalloc1(nmid, &midnodes));
  PetscCall(PetscMalloc1(nint, &intnodes));
  for (PetscInt i = 0; i < n; ++i) {
    switch (nodes[i]) {
    case TOP:
      topnodes[topcnt++] = i;
      break;
    case BOT:
      botnodes[botcnt++] = i;
      break;
    case MID:
      midnodes[midcnt++] = i;
      break;
    case INT:
      intnodes[intcnt++] = i;
      break;
    }
  }
  PetscCall(PetscFree(nodes));
  PetscCall(PetscInfo(NULL, "PCPARSOR: Partitioned nodes, have %" PetscInt_FMT " top, %" PetscInt_FMT " bot, %" PetscInt_FMT " mid and %" PetscInt_FMT " int\n", topcnt, botcnt, midcnt, intcnt));

  /* Split the interior so that TOP + INT1 and INT2 + BOT cost about the same; INT1 hides the BOT scatter.
     The cost of a row is its number of nonzeros (diagonal and off-process block). */
  PetscCall(MatSeqAIJGetCSRAndMemType(Ad, &ai, &aj, NULL, NULL));
#define ROW_COST(r) ((ai[(r) + 1] - ai[r]) + (ii[(r) + 1] - ii[r]))
  for (PetscInt i = 0; i < ntop; ++i) topcost += ROW_COST(topnodes[i]);
  for (PetscInt i = 0; i < nbot; ++i) botcost += ROW_COST(botnodes[i]);
  for (PetscInt i = 0; i < nint; ++i) intcost += ROW_COST(intnodes[i]);

  tgt_int1_cost = (intcost + botcost - topcost) / 2;
  for (splitidx = 0; splitidx < nint; ++splitidx) {
    curr_int1_cost += ROW_COST(intnodes[splitidx]);
    if (curr_int1_cost > tgt_int1_cost) break;
  }
#undef ROW_COST

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), splitidx, intnodes, PETSC_COPY_VALUES, &parsor->int1));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), nint - splitidx, intnodes + splitidx, PETSC_COPY_VALUES, &parsor->int2));
  PetscCall(ISViewFromOptions(parsor->int1, NULL, "-pc_parsor_int1_view"));
  PetscCall(ISViewFromOptions(parsor->int2, NULL, "-pc_parsor_int2_view"));

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), ntop, topnodes, PETSC_COPY_VALUES, &parsor->top));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), nbot, botnodes, PETSC_COPY_VALUES, &parsor->bot));
  PetscCall(MatCreateVecs(matin, &xcol, NULL));

  PetscCall(CreateGhostSubScatter(matin, xcol, parsor->lvec, ntop, topnodes, 0, NULL, &parsor->topsct));
  PetscCall(CreateGhostSubScatter(matin, xcol, parsor->lvec, nbot, botnodes, nmid, midnodes, &parsor->botsct));

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)matin), nmid, midnodes, PETSC_COPY_VALUES, &parsor->mid));
  PetscCall(MatCreateVecs(matin, NULL, &parsor->xx));
  PetscCall(PetscCalloc1(nmid, &parsor->mid_ready));
  PetscCall(PetscCalloc1(nmid, &parsor->mid_node_n_deps));
  PetscCall(PetscCalloc1(nmid, &parsor->mid_node_n_send_to));
  PetscCall(PetscCalloc1(size, &n_mid_recv_buf_size));
  PetscCall(PetscCalloc1(size, &mid_send_ranks));
  PetscCall(PetscCalloc1(nmid, &cnt_arr));

  /* Tell the neighbours which of our nodes are MID (1) and BOT (2). The new value of a MID node is needed by
     the MID and BOT nodes coupled to it on lower-colour ranks, since those are updated after it. */
  PetscCall(VecZeroEntries(xcol));
  PetscCall(VecZeroEntries(lvec));
  PetscCall(VecGetArray(xcol, &xarr));
  for (PetscInt i = 0; i < nmid; ++i) xarr[midnodes[i]] = 1.0;
  for (PetscInt i = 0; i < nbot; ++i) xarr[botnodes[i]] = 2.0;
  PetscCall(VecRestoreArray(xcol, &xarr));
  PetscCall(VecScatterBegin(Mvctx, xcol, lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Mvctx, xcol, lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(lvec, &larr));
#define GHOST_IS_MID(c)        (PetscRealPart(larr[c]) == 1.0)
#define GHOST_IS_MID_OR_BOT(c) (PetscRealPart(larr[c]) > 0.0)

  PetscCall(MatGetSize(Ao, NULL, &ncols));
  PetscCall(PetscCalloc1(ncols, &parsor->lvec_to_mid_count));
  PetscCall(PetscCalloc1(ncols, &needed));
  PetscCall(PetscHMapICreate(&parsor->global_to_lvec));
  parsor->n_lvec_cols = ncols;
  parsor->stat_ntop   = ntop;
  parsor->stat_nbot   = nbot;
  parsor->stat_nint   = nint;
  for (PetscInt i = 0; i < nbot; ++i) {
    for (PetscInt j = ii[botnodes[i]]; j < ii[botnodes[i] + 1]; ++j) {
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, colmap[jj[j]], &owner));
      if (parsor->proccols[owner] > parsor->proccols[rank] && GHOST_IS_MID(jj[j])) {
        parsor->stat_bot_mid_edges++;
        needed[jj[j]] = PETSC_TRUE;
      }
    }
  }
  for (PetscInt i = 0; i < nmid; ++i) {
    const PetscInt row = midnodes[i];
    for (PetscInt j = ii[row]; j < ii[row + 1]; ++j) {
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, colmap[jj[j]], &owner));
      if (parsor->proccols[owner] > parsor->proccols[rank] && GHOST_IS_MID(jj[j])) {
        parsor->mid_node_n_deps[i]++;
        parsor->lvec_to_mid_count[jj[j]]++;
        needed[jj[j]] = PETSC_TRUE;
      } else if (parsor->proccols[owner] < parsor->proccols[rank] && GHOST_IS_MID_OR_BOT(jj[j])) {
        mid_send_ranks[owner]++;
        parsor->mid_node_n_send_to[i]++;
      }
    }
  }
  /* Each needed ghost value arrives exactly once per sweep */
  for (PetscInt c = 0; c < ncols; ++c) {
    PetscMPIInt owner;

    if (!needed[c]) continue;
    PetscCall(PetscLayoutFindOwner(layout, colmap[c], &owner));
    n_mid_recv_buf_size[owner]++;
    PetscCall(PetscHMapISet(parsor->global_to_lvec, colmap[c], c));
  }
  PetscCall(PetscFree(needed));

  PetscCall(PetscCalloc1(ncols, &parsor->lvec_to_mid_nodes));
  PetscCall(PetscCalloc1(ncols, &lvec_cursor));
  for (PetscInt j = 0; j < ncols; ++j) {
    if (parsor->lvec_to_mid_count[j] > 0) PetscCall(PetscCalloc1(parsor->lvec_to_mid_count[j], &parsor->lvec_to_mid_nodes[j]));
  }
  for (PetscInt i = 0; i < nmid; ++i) {
    const PetscInt row = midnodes[i];
    for (PetscInt j = ii[row]; j < ii[row + 1]; ++j) {
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, colmap[jj[j]], &owner));
      if (parsor->proccols[owner] > parsor->proccols[rank] && GHOST_IS_MID(jj[j])) parsor->lvec_to_mid_nodes[jj[j]][lvec_cursor[jj[j]]++] = i;
    }
  }
  PetscCall(PetscFree(lvec_cursor));

  parsor->n_mid_recv_nbs = 0;
  for (PetscMPIInt i = 0; i < size; ++i)
    if (n_mid_recv_buf_size[i] > 0) parsor->n_mid_recv_nbs++;
  PetscCall(PetscCalloc1(parsor->n_mid_recv_nbs, &parsor->mid_recv_nbs));
  cnt = 0;
  for (PetscMPIInt i = 0; i < size; ++i)
    if (n_mid_recv_buf_size[i] > 0) parsor->mid_recv_nbs[cnt++] = i;

  PetscCall(PetscCalloc1(parsor->n_mid_recv_nbs, &parsor->mid_recv_bufs));
  PetscCall(PetscCalloc1(parsor->n_mid_recv_nbs, &parsor->mid_recv_buf_size));
  PetscCall(PetscCalloc1(parsor->n_mid_recv_nbs, &parsor->mid_recv_remaining));
  PetscCall(PetscCalloc1(parsor->n_mid_recv_nbs, &parsor->mid_recv_reqs));
  for (PetscInt i = 0; i < parsor->n_mid_recv_nbs; ++i) {
    parsor->mid_recv_buf_size[i] = n_mid_recv_buf_size[parsor->mid_recv_nbs[i]];
    PetscCall(PetscCalloc1(parsor->mid_recv_buf_size[i], &parsor->mid_recv_bufs[i]));
  }

  PetscCall(PetscCalloc1(nmid, &parsor->mid_node_send_to_nb));
  for (PetscInt i = 0; i < nmid; ++i) PetscCall(PetscCalloc1(parsor->mid_node_n_send_to[i], &parsor->mid_node_send_to_nb[i]));
  for (PetscInt i = 0; i < nmid; ++i) {
    const PetscInt row = midnodes[i];
    for (PetscInt j = ii[row]; j < ii[row + 1]; ++j) {
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, colmap[jj[j]], &owner));
      if (parsor->proccols[owner] < parsor->proccols[rank] && GHOST_IS_MID_OR_BOT(jj[j])) {
        PetscBool rank_already_added = PETSC_FALSE;
        for (PetscInt k = 0; k < cnt_arr[i]; ++k) {
          if (parsor->mid_node_send_to_nb[i][k] == owner) {
            rank_already_added = PETSC_TRUE;
            break;
          }
        }
        if (!rank_already_added) {
          parsor->mid_node_send_to_nb[i][cnt_arr[i]] = owner;
          cnt_arr[i]++;
        }
      }
    }
  }
#undef GHOST_IS_MID
#undef GHOST_IS_MID_OR_BOT
  for (PetscInt i = 0; i < nmid; ++i) {
    parsor->mid_node_n_send_to[i] = cnt_arr[i];
    PetscCall(PetscRealloc(parsor->mid_node_n_send_to[i] * sizeof(PetscMPIInt), &parsor->mid_node_send_to_nb[i]));
  }

  /* Send buffers are filled front to back during a sweep and never overwritten while a send is in flight, so
     they must hold everything sent to a rank in one sweep; there is at most one message per entry. */
  PetscCall(PetscCalloc1(size, &parsor->mid_send_bufs));
  for (PetscMPIInt i = 0; i < size; ++i) PetscCall(PetscCalloc1(mid_send_ranks[i], &parsor->mid_send_bufs[i]));
  parsor->n_send_reqs_max = 0;
  for (PetscMPIInt i = 0; i < size; ++i) parsor->n_send_reqs_max += mid_send_ranks[i];
  PetscCall(PetscCalloc1(parsor->n_send_reqs_max, &parsor->send_reqs));

  parsor->n_mid_send_nbs = 0;
  for (PetscMPIInt i = 0; i < size; ++i)
    if (mid_send_ranks[i] > 0) parsor->n_mid_send_nbs++;
  PetscCall(PetscCalloc1(parsor->n_mid_send_nbs, &parsor->mid_send_nbs));
  cnt = 0;
  for (PetscMPIInt i = 0; i < size; ++i)
    if (mid_send_ranks[i] > 0) parsor->mid_send_nbs[cnt++] = i;

  PetscCall(VecRestoreArrayRead(lvec, &larr));
  PetscCall(VecDestroy(&xcol));
  PetscCall(VecScatterDestroy(&Mvctx));

  /* Get CSR arrays for diagonal block using public API */
  PetscCall(MatGetLocalSize(matin, &nrows, NULL));
  PetscCall(PetscMalloc1(nrows, &row_to_mid));
  for (PetscInt i = 0; i < nrows; ++i) row_to_mid[i] = -1;
  for (PetscInt m = 0; m < nmid; ++m) row_to_mid[midnodes[m]] = m;

  PetscCall(PetscCalloc1(nmid, &parsor->mid_local_dep_count));
  for (PetscInt m = 0; m < nmid; ++m) {
    const PetscInt row = midnodes[m];
    for (PetscInt j = ai[row]; j < ai[row + 1]; ++j) {
      PetscInt c  = aj[j];
      PetscInt m2 = row_to_mid[c];
      if (m2 < 0 || m2 == m) continue;
      if (midnodes[m2] < row) {
        parsor->mid_node_n_deps[m]++;
      } else {
        parsor->mid_local_dep_count[m]++;
      }
    }
  }

  PetscCall(PetscCalloc1(nmid, &local_cursor));
  PetscCall(PetscCalloc1(nmid, &parsor->mid_local_deps));
  for (PetscInt m = 0; m < nmid; ++m) {
    if (parsor->mid_local_dep_count[m] > 0) PetscCall(PetscMalloc1(parsor->mid_local_dep_count[m], &parsor->mid_local_deps[m]));
  }
  for (PetscInt m = 0; m < nmid; ++m) {
    const PetscInt row = midnodes[m];
    for (PetscInt j = ai[row]; j < ai[row + 1]; ++j) {
      PetscInt c  = aj[j];
      PetscInt m2 = row_to_mid[c];
      if (m2 < 0 || m2 == m) continue;
      if (midnodes[m2] > row) {
        PetscBool already = PETSC_FALSE;
        for (PetscInt k = 0; k < local_cursor[m]; ++k) {
          if (parsor->mid_local_deps[m][k] == m2) {
            already = PETSC_TRUE;
            break;
          }
        }
        if (!already) parsor->mid_local_deps[m][local_cursor[m]++] = m2;
      }
    }
  }
  for (PetscInt m = 0; m < nmid; ++m) parsor->mid_local_dep_count[m] = local_cursor[m];
  PetscCall(PetscFree(local_cursor));
  PetscCall(PetscFree(row_to_mid));
  PetscCall(PetscFree(n_mid_recv_buf_size));
  PetscCall(PetscFree(mid_send_ranks));
  PetscCall(PetscFree(cnt_arr));
  PetscCall(PetscFree(topnodes));
  PetscCall(PetscFree(botnodes));
  PetscCall(PetscFree(midnodes));
  PetscCall(PetscFree(intnodes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParallelSORSetUp(Mat matin, ParallelSORData *parsor)
{
  Mat          A, B;
  PetscScalar *aa_tmp, *ba_tmp;

  PetscFunctionBegin;
  PetscCall(ColorProcessors(matin, &parsor->proccols));
  PetscCall(ParallelSORPartitionNodes(matin, parsor));
  {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)matin), &size));
    for (PetscMPIInt p = 0; p < size; ++p) parsor->stat_ncolors = PetscMax(parsor->stat_ncolors, parsor->proccols[p] + 1);
  }
  PetscCall(PetscFree(parsor->proccols));
  PetscCall(PetscCommGetNewTag(PetscObjectComm((PetscObject)matin), &parsor->tag));

  /* Cache values that don't change between applies */
  parsor->comm = PetscObjectComm((PetscObject)matin);
  PetscCallMPI(MPI_Comm_rank(parsor->comm, &parsor->rank));
  PetscCallMPI(MPI_Comm_size(parsor->comm, (int *)&parsor->size));
  PetscCall(MatMPIAIJGetSeqAIJ(matin, &A, &B, &parsor->colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(A, &parsor->arowptr, &parsor->acolind, &aa_tmp, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(B, &parsor->browptr, &parsor->bcolind, &ba_tmp, NULL));
  parsor->aa = aa_tmp;
  parsor->ba = ba_tmp;
  PetscCall(ISGetLocalSize(parsor->mid, &parsor->nmid));
  PetscCall(ISGetIndices(parsor->mid, &parsor->midnodes));
  PetscCall(MatGetOwnershipRange(matin, &parsor->rstart, NULL));
  PetscCall(PetscCalloc1(parsor->size, &parsor->mid_send_start));
  PetscCall(PetscCalloc1(parsor->size, &parsor->mid_send_cursor));
  PetscCall(PetscCalloc1(parsor->nmid, &parsor->mid_dep_left));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParallelSORDestroy(ParallelSORData *parsor)
{
  PetscInt nmid;

  PetscFunctionBegin;
  if (!parsor) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISDestroy(&parsor->top));
  PetscCall(ISDestroy(&parsor->bot));
  nmid = parsor->nmid;
  if (parsor->midnodes) PetscCall(ISRestoreIndices(parsor->mid, &parsor->midnodes));
  PetscCall(ISDestroy(&parsor->mid));
  PetscCall(ISDestroy(&parsor->int1));
  PetscCall(ISDestroy(&parsor->int2));
  PetscCall(VecScatterDestroy(&parsor->topsct));
  PetscCall(VecScatterDestroy(&parsor->botsct));
  PetscCall(VecDestroy(&parsor->lvec));
  PetscCall(PetscFree(parsor->mid_ready));
  PetscCall(PetscFree(parsor->mid_node_n_deps));
  PetscCall(PetscFree(parsor->mid_node_n_send_to));
  for (PetscInt i = 0; i < nmid; ++i) PetscCall(PetscFree(parsor->mid_node_send_to_nb[i]));
  PetscCall(PetscFree(parsor->mid_node_send_to_nb));
  PetscCall(PetscFree(parsor->mid_local_dep_count));
  for (PetscInt i = 0; i < nmid; ++i) PetscCall(PetscFree(parsor->mid_local_deps[i]));
  PetscCall(PetscFree(parsor->mid_local_deps));
  PetscCall(PetscFree(parsor->mid_recv_nbs));
  PetscCall(PetscFree(parsor->mid_recv_reqs));
  PetscCall(PetscFree(parsor->mid_recv_buf_size));
  PetscCall(PetscFree(parsor->mid_recv_remaining));
  for (PetscInt i = 0; i < parsor->n_mid_recv_nbs; ++i) PetscCall(PetscFree(parsor->mid_recv_bufs[i]));
  PetscCall(PetscFree(parsor->mid_recv_bufs));
  for (PetscInt p = 0; p < parsor->n_mid_send_nbs; ++p) PetscCall(PetscFree(parsor->mid_send_bufs[parsor->mid_send_nbs[p]]));
  PetscCall(PetscFree(parsor->mid_send_nbs));
  PetscCall(PetscFree(parsor->send_reqs));
  PetscCall(PetscFree(parsor->mid_send_bufs));
  PetscCall(PetscHMapIDestroy(&parsor->global_to_lvec));
  PetscCall(PetscFree(parsor->lvec_to_mid_count));
  for (PetscInt j = 0; j < parsor->n_lvec_cols; ++j) PetscCall(PetscFree(parsor->lvec_to_mid_nodes[j]));
  PetscCall(PetscFree(parsor->lvec_to_mid_nodes));
  PetscCall(VecDestroy(&parsor->xx));
  PetscCall(PetscFree(parsor->mid_send_start));
  PetscCall(PetscFree(parsor->mid_send_cursor));
  PetscCall(PetscFree(parsor->mid_dep_left));
  PetscCall(PetscFree(parsor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SORLocalForwardSweepIS(const PetscInt *rowptr, const PetscInt *colind, const MatScalar *matvals, const PetscInt *diag, const PetscScalar *idiag, PetscReal omega, IS is, const PetscScalar *b, PetscScalar *x, const PetscInt *browptr, const PetscInt *bcolind, const MatScalar *bmatvals, const PetscScalar *lv)
{
  PetscInt        isn;
  const PetscInt *isptr;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &isn));
  if (isn == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISGetIndices(is, &isptr));
  for (PetscInt j = 0; j < isn; ++j) {
    const PetscInt   i = isptr[j];
    const MatScalar *v;
    const PetscInt  *idx;
    PetscScalar      sum;
    PetscInt         n;

    n   = diag[i] - rowptr[i];
    idx = colind + rowptr[i];
    v   = matvals + rowptr[i];
    sum = b[i];
    SparseDenseMinusDot(&sum, x, v, idx, n);
    n   = rowptr[i + 1] - diag[i] - 1;
    idx = colind + diag[i] + 1;
    v   = matvals + diag[i] + 1;
    SparseDenseMinusDot(&sum, x, v, idx, n);
    if (browptr) {
      n   = browptr[i + 1] - browptr[i];
      idx = bcolind + browptr[i];
      v   = bmatvals + browptr[i];
      SparseDenseMinusDot(&sum, lv, v, idx, n);
    }
    x[i] = (1. - omega) * x[i] + sum * idiag[i];
  }
  PetscCall(ISRestoreIndices(is, &isptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Store a received batch of (global id, value) pairs in the ghost vector, release the MID nodes waiting for them
   and, if more values are expected from this neighbour, post the next receive. */
static PetscErrorCode ParallelSORProcessMessage(ParallelSORData *parsor, PetscInt p, PetscMPIInt count, PetscScalar *lv, PetscInt *nready)
{
  PetscFunctionBegin;
  for (PetscMPIInt i = 0; i < count; ++i) {
    PetscInt      gid = parsor->mid_recv_bufs[p][i].id, lidx;
    PetscBool     found;
    PetscHashIter hit;

    PetscCall(PetscHMapIFind(parsor->global_to_lvec, gid, &hit, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Received MID update for global index %" PetscInt_FMT " not found in lvec map", gid);
    PetscCall(PetscHMapIIterGet(parsor->global_to_lvec, hit, &lidx));
    lv[lidx] = parsor->mid_recv_bufs[p][i].data;
    for (PetscInt k = 0; k < parsor->lvec_to_mid_count[lidx]; ++k) {
      const PetscInt m = parsor->lvec_to_mid_nodes[lidx][k];
      if (--parsor->mid_dep_left[m] == 0) parsor->mid_ready[(*nready)++] = m;
    }
  }
  parsor->mid_recv_remaining[p] -= count;
  PetscCheck(parsor->mid_recv_remaining[p] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Received more MID updates than expected from rank %d", parsor->mid_recv_nbs[p]);
  if (parsor->mid_recv_remaining[p] > 0) PetscCallMPI(MPI_Irecv(parsor->mid_recv_bufs[p], (PetscMPIInt)(parsor->mid_recv_remaining[p] * sizeof(MidIDData)), MPI_BYTE, parsor->mid_recv_nbs[p], parsor->tag, parsor->comm, &parsor->mid_recv_reqs[p]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParallelSORWaitMessage(ParallelSORData *parsor, PetscScalar *lv, PetscInt *nready)
{
  PetscMPIInt completed, bytes;
  MPI_Status  status;

  PetscFunctionBegin;
  parsor->stat_nrounds++;
  PetscCallMPI(MPI_Waitany((PetscMPIInt)parsor->n_mid_recv_nbs, parsor->mid_recv_reqs, &completed, &status));
  PetscCheck(completed != MPI_UNDEFINED, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Waiting for MID updates but no receive is posted");
  PetscCallMPI(MPI_Get_count(&status, MPI_BYTE, &bytes));
  PetscCheck(bytes % (PetscMPIInt)sizeof(MidIDData) == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Received data size not a multiple of MidIDData");
  PetscCall(ParallelSORProcessMessage(parsor, completed, bytes / (PetscMPIInt)sizeof(MidIDData), lv, nready));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One forward SOR sweep in a consistent global ordering (Adams 2001): processors are coloured, and of two coupled
   nodes on different processors the one on the higher-colour processor is updated first. Locally the nodes are
   updated in the order TOP (only lower-colour neighbours), INT1, MID (both), INT2, BOT (only higher-colour
   neighbours). TOP and BOT nodes get the values they need through two scatters; MID nodes exchange their new values
   with point-to-point messages as soon as they are computed. */
static PetscErrorCode ParallelSORApply(ParallelSORData *parsor, const PetscInt *diag, const PetscScalar *idiag_arr, Vec bb, PetscReal omega, PetscInt its, PetscBool zero_initial_guess, Vec xx)
{
  const MatScalar   *aa = parsor->aa, *ba = parsor->ba;
  const PetscInt    *arowptr = parsor->arowptr, *acolind = parsor->acolind;
  const PetscInt    *browptr = parsor->browptr, *bcolind = parsor->bcolind;
  const PetscInt    *midnodes = parsor->midnodes;
  PetscScalar       *x, *lv;
  const PetscScalar *b1;
  PetscInt           nmid = parsor->nmid, rstart = parsor->rstart, mid_remaining, recv_remaining, nready;
  PetscMPIInt        nsend;
  PetscBool          first_iter = zero_initial_guess;
  PetscLogDouble     t_start, t_lap, t_now;

  PetscFunctionBegin;
  PetscCall(PetscTime(&t_start));
#define LAP(ph) \
  do { \
    PetscCall(PetscTime(&t_now)); \
    parsor->stat_tphase[ph] += t_now - t_lap; \
    t_lap = t_now; \
  } while (0)
  while (its--) {
    parsor->stat_napply++;
    PetscCall(PetscTime(&t_lap));
    PetscCall(VecZeroEntries(parsor->lvec));
    if (first_iter) {
      PetscCall(VecZeroEntries(xx));
    } else {
      PetscCall(VecScatterBegin(parsor->topsct, xx, parsor->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(parsor->topsct, xx, parsor->lvec, INSERT_VALUES, SCATTER_FORWARD));
    }
    first_iter = PETSC_FALSE;
    LAP(PH_TOPSCATTER);

    /* TOP nodes: their neighbours on other processors are all updated later, so they use the old values */
    {
      const PetscScalar *lvread;

      PetscCall(VecGetArrayRead(parsor->lvec, &lvread));
      PetscCall(VecGetArray(xx, &x));
      PetscCall(VecGetArrayRead(bb, &b1));
      PetscCall(SORLocalForwardSweepIS(arowptr, acolind, aa, diag, idiag_arr, omega, parsor->top, b1, x, browptr, bcolind, ba, lvread));
      PetscCall(VecRestoreArrayRead(bb, &b1));
      PetscCall(VecRestoreArray(xx, &x));
      PetscCall(VecRestoreArrayRead(parsor->lvec, &lvread));
    }
    LAP(PH_TOP);

    /* Send the new TOP values (and old MID/BOT values) to the neighbours, overlapped with the first interior sweep */
    PetscCall(VecCopy(xx, parsor->xx));
    PetscCall(VecScatterBegin(parsor->botsct, parsor->xx, parsor->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(xx, &x));
    PetscCall(VecGetArrayRead(bb, &b1));
    LAP(PH_COPY);
    PetscCall(SORLocalForwardSweepIS(arowptr, acolind, aa, diag, idiag_arr, omega, parsor->int1, b1, x, NULL, NULL, NULL, NULL));
    LAP(PH_INT1);
    PetscCall(VecScatterEnd(parsor->botsct, parsor->xx, parsor->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(parsor->lvec, &lv));
    LAP(PH_BOTSCATTER);

    /* MID nodes */
    nsend = 0;
    if (parsor->debug_no_mid_wait) {
      PetscCall(SORLocalForwardSweepIS(arowptr, acolind, aa, diag, idiag_arr, omega, parsor->mid, b1, x, browptr, bcolind, ba, lv));
      mid_remaining = 0;
      for (PetscInt p = 0; p < parsor->n_mid_recv_nbs; ++p) parsor->mid_recv_remaining[p] = 0;
    }
    nready = 0;
    for (PetscInt m = 0; m < nmid && !parsor->debug_no_mid_wait; ++m) {
      parsor->mid_dep_left[m] = parsor->mid_node_n_deps[m];
      if (parsor->mid_dep_left[m] == 0) parsor->mid_ready[nready++] = m;
    }
    recv_remaining = 0;
    for (PetscInt p = 0; p < parsor->n_mid_recv_nbs && !parsor->debug_no_mid_wait; ++p) {
      parsor->mid_recv_remaining[p] = parsor->mid_recv_buf_size[p];
      recv_remaining += parsor->mid_recv_buf_size[p];
      PetscCallMPI(MPI_Irecv(parsor->mid_recv_bufs[p], (PetscMPIInt)(parsor->mid_recv_buf_size[p] * sizeof(MidIDData)), MPI_BYTE, parsor->mid_recv_nbs[p], parsor->tag, parsor->comm, &parsor->mid_recv_reqs[p]));
    }
    for (PetscInt p = 0; p < parsor->n_mid_send_nbs; ++p) parsor->mid_send_start[parsor->mid_send_nbs[p]] = parsor->mid_send_cursor[parsor->mid_send_nbs[p]] = 0;
    mid_remaining = parsor->debug_no_mid_wait ? 0 : nmid;
    while (mid_remaining > 0) {
      parsor->stat_nscans++;
      while (nready > 0) {
        const PetscInt   m   = parsor->mid_ready[--nready];
        const PetscInt   row = midnodes[m];
        PetscScalar      sum = b1[row];
        const MatScalar *v;
        const PetscInt  *idx;
        PetscInt         n;

        n   = diag[row] - arowptr[row];
        idx = acolind + arowptr[row];
        v   = aa + arowptr[row];
        SparseDenseMinusDot(&sum, x, v, idx, n);
        n   = arowptr[row + 1] - diag[row] - 1;
        idx = acolind + diag[row] + 1;
        v   = aa + diag[row] + 1;
        SparseDenseMinusDot(&sum, x, v, idx, n);
        n   = browptr[row + 1] - browptr[row];
        idx = bcolind + browptr[row];
        v   = ba + browptr[row];
        SparseDenseMinusDot(&sum, lv, v, idx, n);
        x[row] = (1. - omega) * x[row] + sum * idiag_arr[row];
        mid_remaining--;

        for (PetscInt k = 0; k < parsor->mid_local_dep_count[m]; ++k) {
          const PetscInt m2 = parsor->mid_local_deps[m][k];
          if (--parsor->mid_dep_left[m2] == 0) parsor->mid_ready[nready++] = m2;
        }
        for (PetscInt k = 0; k < parsor->mid_node_n_send_to[m]; ++k) {
          const PetscMPIInt dest = parsor->mid_node_send_to_nb[m][k];
          MidIDData        *e    = &parsor->mid_send_bufs[dest][parsor->mid_send_cursor[dest]++];

          e->id   = rstart + row;
          e->data = x[row];
        }
      }

      /* Send what was computed in this round; the buffer regions are not touched again in this sweep */
      for (PetscInt p = 0; p < parsor->n_mid_send_nbs; ++p) {
        const PetscMPIInt dest  = parsor->mid_send_nbs[p];
        const PetscInt    start = parsor->mid_send_start[dest], count = parsor->mid_send_cursor[dest] - start;

        if (count == 0) continue;
        PetscCheck(nsend < parsor->n_send_reqs_max, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Too many MID messages");
        PetscCallMPI(MPI_Isend(parsor->mid_send_bufs[dest] + start, (PetscMPIInt)(count * sizeof(MidIDData)), MPI_BYTE, dest, parsor->tag, parsor->comm, &parsor->send_reqs[nsend++]));
        parsor->mid_send_start[dest] = parsor->mid_send_cursor[dest];
      }
      if (mid_remaining > 0) PetscCall(ParallelSORWaitMessage(parsor, lv, &nready));
    }
    LAP(PH_MID);

    /* INT2 nodes, then wait for the new values of MID nodes on higher-colour processors that our BOT nodes need */
    PetscCall(SORLocalForwardSweepIS(arowptr, acolind, aa, diag, idiag_arr, omega, parsor->int2, b1, x, NULL, NULL, NULL, NULL));
    LAP(PH_INT2);
    recv_remaining = 0;
    for (PetscInt p = 0; p < parsor->n_mid_recv_nbs; ++p) recv_remaining += parsor->mid_recv_remaining[p];
    while (recv_remaining > 0) {
      PetscCall(ParallelSORWaitMessage(parsor, lv, &nready));
      recv_remaining = 0;
      for (PetscInt p = 0; p < parsor->n_mid_recv_nbs; ++p) recv_remaining += parsor->mid_recv_remaining[p];
    }
    LAP(PH_BOTWAIT);
    PetscCheck(nready == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MID node became ready after all MID nodes were updated");

    /* BOT nodes: all their neighbours on other processors have been updated */
    PetscCall(SORLocalForwardSweepIS(arowptr, acolind, aa, diag, idiag_arr, omega, parsor->bot, b1, x, browptr, bcolind, ba, lv));
    LAP(PH_BOT);
    PetscCallMPI(MPI_Waitall(nsend, parsor->send_reqs, MPI_STATUSES_IGNORE));
    PetscCall(VecRestoreArray(parsor->lvec, &lv));
    PetscCall(VecRestoreArrayRead(bb, &b1));
    PetscCall(VecRestoreArray(xx, &x));
    LAP(PH_SENDWAIT);
  }
#undef LAP
  PetscCall(PetscTime(&t_now));
  parsor->stat_ttotal += t_now - t_start;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Collective. Prints min/max/sum over ranks of the node partition (at setup) or of the apply counters. */
static PetscErrorCode ParallelSORPrintStats(PC pc, PetscBool apply_stats)
{
  PC_PARSOR        parsor = (PC_PARSOR)pc->data;
  ParallelSORData *d      = parsor->parsor_data;
  MPI_Comm         comm   = PetscObjectComm((PetscObject)pc);
  const char      *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  if (!apply_stats) {
    PetscInt nloc, nint1, nint2, loc[9], mx[9], mn[9], sm[9];

    PetscCall(MatGetLocalSize(pc->pmat, &nloc, NULL));
    PetscCall(ISGetLocalSize(d->int1, &nint1));
    PetscCall(ISGetLocalSize(d->int2, &nint2));
    loc[0] = nloc, loc[1] = d->stat_ntop, loc[2] = nint1, loc[3] = nint2, loc[4] = d->nmid, loc[5] = d->stat_nbot;
    loc[6] = d->n_mid_recv_nbs, loc[7] = d->n_mid_send_nbs, loc[8] = d->stat_bot_mid_edges;
    PetscCallMPI(MPIU_Allreduce(loc, mx, 9, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(loc, mn, 9, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPIU_Allreduce(loc, sm, 9, MPIU_INT, MPI_SUM, comm));
    {
      PetscInt    active = nloc > 0, nactive;
      PetscMPIInt size;

      PetscCallMPI(MPI_Comm_size(comm, &size));
      PetscCallMPI(MPIU_Allreduce(&active, &nactive, 1, MPIU_INT, MPI_SUM, comm));
      PetscCall(PetscPrintf(comm, "PARSOR %s setup: %" PetscInt_FMT " processor colours, %" PetscInt_FMT " of %d ranks own rows\n", prefix ? prefix : "", d->stat_ncolors, nactive, size));
    }
    PetscCall(PetscPrintf(comm, "  %-26s %10s %10s %12s\n", "", "min", "max", "sum"));
    {
      const char *names[9] = {"rows", "top", "int1", "int2", "mid", "bot", "mid recv neighbours", "mid send neighbours", "BOT<-MID couplings"};
      for (PetscInt k = 0; k < 9; ++k) PetscCall(PetscPrintf(comm, "  %-26s %10" PetscInt_FMT " %10" PetscInt_FMT " %12" PetscInt_FMT "\n", names[k], mn[k], mx[k], sm[k]));
    }
  } else {
    PetscReal loc[PH_N + 3], mx[PH_N + 3], mn[PH_N + 3];
    PetscInt  napply = PetscMax(d->stat_napply, 1);

    loc[0] = (PetscReal)d->stat_nrounds / napply, loc[1] = (PetscReal)d->stat_nscans / napply;
    for (PetscInt k = 0; k < PH_N; ++k) loc[2 + k] = 1e3 * d->stat_tphase[k] / napply;
    loc[PH_N + 2] = 1e3 * d->stat_ttotal / napply;
    PetscCallMPI(MPIU_Allreduce(loc, mx, PH_N + 3, MPIU_REAL, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(loc, mn, PH_N + 3, MPIU_REAL, MPI_MIN, comm));
    PetscCall(PetscPrintf(comm, "PARSOR %s apply (%" PetscInt_FMT " sweeps), per sweep:\n", prefix ? prefix : "", d->stat_napply));
    PetscCall(PetscPrintf(comm, "  %-30s %10s %10s\n", "", "min", "max"));
    PetscCall(PetscPrintf(comm, "  %-30s %10.3g %10.3g\n", "message rounds", (double)mn[0], (double)mx[0]));
    PetscCall(PetscPrintf(comm, "  %-30s %10.3g %10.3g\n", "scans over mid list", (double)mn[1], (double)mx[1]));
    for (PetscInt k = 0; k < PH_N; ++k) PetscCall(PetscPrintf(comm, "  %-30s %10.3g %10.3g  [ms]\n", ParallelSORPhaseNames[k], (double)mn[2 + k], (double)mx[2 + k]));
    PetscCall(PetscPrintf(comm, "  %-30s %10.3g %10.3g  [ms]\n", "total", (double)mn[PH_N + 2], (double)mx[PH_N + 2]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PARSOR(PC pc, Vec b, Vec x)
{
  PC_PARSOR          parsor = (PC_PARSOR)pc->data;
  const PetscScalar *idiag_arr;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(parsor->idiag_vec, &idiag_arr));
  PetscCall(ParallelSORApply(parsor->parsor_data, parsor->diag, idiag_arr, b, parsor->omega, parsor->its, PETSC_TRUE, x));
  PetscCall(VecRestoreArrayRead(parsor->idiag_vec, &idiag_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCPARSORApplySOR(PC pc, Vec b, PetscInt its, PetscBool zero_initial_guess, Vec x)
{
  PC_PARSOR          parsor;
  const PetscScalar *idiag_arr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  parsor = (PC_PARSOR)pc->data;
  PetscCall(VecGetArrayRead(parsor->idiag_vec, &idiag_arr));
  PetscCall(ParallelSORApply(parsor->parsor_data, parsor->diag, idiag_arr, b, parsor->omega, its, zero_initial_guess, x));
  PetscCall(VecRestoreArrayRead(parsor->idiag_vec, &idiag_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sweeps directly on the current iterate, which saves KSPRICHARDSON the residual computation and PC application per step */
static PetscErrorCode PCApplyRichardson_PARSOR(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_PARSOR          parsor = (PC_PARSOR)pc->data;
  const PetscScalar *idiag_arr;

  PetscFunctionBegin;
  (void)w, (void)rtol, (void)abstol, (void)dtol;
  PetscCall(VecGetArrayRead(parsor->idiag_vec, &idiag_arr));
  PetscCall(ParallelSORApply(parsor->parsor_data, parsor->diag, idiag_arr, b, parsor->omega, its * parsor->its, guesszero, y));
  PetscCall(VecRestoreArrayRead(parsor->idiag_vec, &idiag_arr));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_PARSOR(PC pc)
{
  PC_PARSOR parsor = (PC_PARSOR)pc->data;

  PetscFunctionBegin;
  if (parsor->parsor_data) {
    if (parsor->stats) PetscCall(ParallelSORPrintStats(pc, PETSC_TRUE));
    PetscCall(ParallelSORDestroy(parsor->parsor_data));
    parsor->parsor_data = NULL;
  }
  PetscCall(VecDestroy(&parsor->idiag_vec));
  PetscCall(PetscFree(parsor->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_PARSOR(PC pc)
{
  PC_PARSOR parsor = (PC_PARSOR)pc->data;

  PetscFunctionBegin;
  if (parsor->parsor_data) {
    if (parsor->stats) PetscCall(ParallelSORPrintStats(pc, PETSC_TRUE));
    PetscCall(ParallelSORDestroy(parsor->parsor_data));
  }
  PetscCall(VecDestroy(&parsor->idiag_vec));
  PetscCall(PetscFree(parsor->diag));
  PetscCall(PetscFree(parsor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_PARSOR(PC pc)
{
  PC_PARSOR parsor = (PC_PARSOR)pc->data;
  MatType   mtype;
  PetscBool is_mpiaij, is_seqaij;
  PetscReal omega_save;
  PetscInt  its_save;

  PetscFunctionBegin;
  PetscCall(MatGetType(pc->pmat, &mtype));
  PetscCall(PetscStrcmp(mtype, MATMPIAIJ, &is_mpiaij));
  PetscCall(PetscStrcmp(mtype, MATSEQAIJ, &is_seqaij));

  if (is_seqaij) {
    /* Save values before changing PC type (which will free parsor) */
    omega_save = parsor->omega;
    its_save   = parsor->its;
    PetscCall(PetscInfo(pc, "PCPARSOR: Matrix is SEQAIJ, falling back to standard PCSOR\n"));
    PetscCall(PCSetType(pc, PCSOR));
    PetscCall(PCSORSetOmega(pc, omega_save));
    PetscCall(PCSORSetIterations(pc, its_save, 1));
    PetscCall(PCSORSetSymmetric(pc, SOR_FORWARD_SWEEP)); /* PARSOR is a forward sweep */
    PetscCall(PCSetUp(pc));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCheck(is_mpiaij, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCPARSOR only supports MATMPIAIJ and MATSEQAIJ matrices, got %s", mtype);

  if (parsor->parsor_data && pc->flag == DIFFERENT_NONZERO_PATTERN) {
    if (parsor->stats) PetscCall(ParallelSORPrintStats(pc, PETSC_TRUE));
    PetscCall(ParallelSORDestroy(parsor->parsor_data));
    parsor->parsor_data = NULL;
  }
  if (!parsor->parsor_data) {
    PetscCall(PetscNew(&parsor->parsor_data));
    PetscCall(ParallelSORSetUp(pc->pmat, parsor->parsor_data));
    parsor->parsor_data->debug_no_mid_wait = parsor->debug_no_mid_wait;
    if (parsor->stats) PetscCall(ParallelSORPrintStats(pc, PETSC_FALSE));
  } else {
    /* Same nonzero pattern, possibly new values: refresh the cached value arrays */
    Mat          A, B;
    PetscScalar *aa, *ba;

    PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &A, &B, NULL));
    PetscCall(MatSeqAIJGetCSRAndMemType(A, NULL, NULL, &aa, NULL));
    PetscCall(MatSeqAIJGetCSRAndMemType(B, NULL, NULL, &ba, NULL));
    parsor->parsor_data->aa = aa;
    parsor->parsor_data->ba = ba;
  }
  {
    Mat A;

    PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &A, NULL, NULL));
    PetscCall(MatSetOption(A, MAT_USE_INODES, PETSC_FALSE));
    PetscCall(PetscFree(parsor->diag));
    PetscCall(LocalMatInvertDiagonalForSOR(A, parsor->omega, 0., &parsor->diag, &parsor->idiag_vec));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_PARSOR(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_PARSOR parsor = (PC_PARSOR)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Parallel SOR options");
  PetscCall(PetscOptionsReal("-pc_parsor_omega", "Relaxation factor", "PCPARSORSetOmega", parsor->omega, &parsor->omega, NULL));
  PetscCall(PetscOptionsInt("-pc_parsor_its", "Number of SOR iterations", "PCPARSORSetIterations", parsor->its, &parsor->its, NULL));
  PetscCall(PetscOptionsBool("-pc_parsor_stats", "Print partition and communication statistics (debugging)", NULL, parsor->stats, &parsor->stats, NULL));
  PetscCall(PetscOptionsBool("-pc_parsor_debug_no_mid_wait", "INCORRECT sweep for timing only: MID nodes do not wait for remote updates", NULL, parsor->debug_no_mid_wait, &parsor->debug_no_mid_wait, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_PARSOR(PC pc, PetscViewer viewer)
{
  PC_PARSOR parsor = (PC_PARSOR)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Omega: %g\n", (double)parsor->omega));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Iterations: %" PetscInt_FMT "\n", parsor->its));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Sweep type: Forward\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCPARSORSetOmega(PC pc, PetscReal omega)
{
  PC_PARSOR parsor;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveReal(pc, omega, 2);
  parsor        = (PC_PARSOR)pc->data;
  parsor->omega = omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCPARSORSetIterations(PC pc, PetscInt its)
{
  PC_PARSOR parsor;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, its, 2);
  parsor      = (PC_PARSOR)pc->data;
  parsor->its = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_PARSOR(PC pc)
{
  PC_PARSOR parsor;

  PetscFunctionBegin;
  PetscCall(PetscNew(&parsor));
  pc->data            = parsor;
  parsor->omega       = 1.0;
  parsor->its         = 1;
  parsor->parsor_data = NULL;

  pc->ops->apply           = PCApply_PARSOR;
  pc->ops->applyrichardson = PCApplyRichardson_PARSOR;
  pc->ops->destroy         = PCDestroy_PARSOR;
  pc->ops->reset           = PCReset_PARSOR;
  pc->ops->setup           = PCSetUp_PARSOR;
  pc->ops->setfromoptions  = PCSetFromOptions_PARSOR;
  pc->ops->view            = PCView_PARSOR;
  PetscFunctionReturn(PETSC_SUCCESS);
}
