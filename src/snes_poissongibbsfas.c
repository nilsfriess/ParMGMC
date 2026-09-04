/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.

    Non-linear Gibbs sampler for posterior obtained by contitioning a Gaussian prior
    on a Poisson process.
*/

#include "parmgmc/snes/snes_poissongibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/snesimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

typedef struct {  
  SNES fas;         // FAS
  PC mg;            // multigrid 
  PetscInt nlevels; // number of multigrid levels
  PoissonGibbsCtx *smoother_ctx;
}  SNES_PoissonGibbsFAS;

/* Generate a new sample (computational routine) */
static PetscErrorCode SNESSample_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscInt nlevels;

  PetscFunctionBeginUser;
  if (poissongibbsfas->smoother_ctx) {
    PetscCall(PCMGGetLevels(poissongibbsfas->mg,&nlevels));
    for (PetscInt ell=0;ell<nlevels;++ell) {
      PetscCall(MatDestroy(&poissongibbsfas->smoother_ctx[ell].Q_prec));
      PetscCall(MatDestroy(&poissongibbsfas->smoother_ctx[ell].B_meas));
      PetscCall(VecDestroy(&poissongibbsfas->smoother_ctx[ell].event_counts));
    }
    PetscCall(PetscFree(poissongibbsfas->smoother_ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESDestroy_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;

  PetscFunctionBeginUser;
  PetscCall(PCDestroy(&poissongibbsfas->mg));
  PetscCall(SNESDestroy(&poissongibbsfas->fas));
  PetscCall(PetscFree(poissongibbsfas));
  PetscFunctionReturn(PETSC_SUCCESS);  
}

static PetscErrorCode setup_multigrid(SNES snes) {
  
  PetscInt nlevels;
  Mat* P;

  PetscFunctionBeginUser;
  
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PoissonGibbsCtx* ctx;
  PetscCall(SNESGetApplicationContext(snes, &ctx));

  PetscCall(PCSetDM(poissongibbsfas->mg, snes->dm));
  PetscCall(PCSetOperators(poissongibbsfas->mg,ctx->Q_prec,ctx->Q_prec));
  PetscCall(PCSetUp(poissongibbsfas->mg));
  PetscCall(PCMGGetLevels(poissongibbsfas->mg,&nlevels));
  PetscCall(PetscMalloc1(nlevels,&poissongibbsfas->smoother_ctx));
  // Extract prolongation operators
  PetscCall(PCGetInterpolations(poissongibbsfas->mg, &nlevels, &P));
  // Construct precision matrices on all levels
  for (PetscInt ell=nlevels-1;ell>=0;--ell) {
    poissongibbsfas->smoother_ctx[ell].event_counts = ctx->event_counts;    
    PetscCall(PetscObjectReference((PetscObject)ctx->event_counts));
    // On finest level, just point to already existing matrices
    if (ell==nlevels-1) {
      poissongibbsfas->smoother_ctx[ell].Q_prec = ctx->Q_prec;
      PetscCall(PetscObjectReference((PetscObject)ctx->Q_prec));
      poissongibbsfas->smoother_ctx[ell].B_meas = ctx->B_meas;
      PetscCall(PetscObjectReference((PetscObject)ctx->B_meas));
    } else {
      // event_counts is just copied
      // Q_c = P^T Q P
      Mat Q_prec_P;
      PetscCall(MatMatMult(poissongibbsfas->smoother_ctx[ell+1].Q_prec, P[ell],
                           MAT_INITIAL_MATRIX, PETSC_DEFAULT, 
                           &Q_prec_P));
      PetscCall(MatTransposeMatMult(P[ell], Q_prec_P, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
                           &poissongibbsfas->smoother_ctx[ell].Q_prec));
      // B_c = P^T B
      PetscCall(MatTransposeMatMult(P[ell], poissongibbsfas->smoother_ctx[ell+1].B_meas,
                                    MAT_INITIAL_MATRIX, PETSC_DEFAULT,
                                    &poissongibbsfas->smoother_ctx[ell].B_meas)); 
    }
  }
  poissongibbsfas->nlevels = nlevels;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode setup_fas(SNES snes) {
  
  PetscInt nlevels;
  Mat* P;

  PetscFunctionBeginUser;
  
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PoissonGibbsCtx* ctx;
  PetscCall(SNESGetApplicationContext(snes, &ctx));

  nlevels = poissongibbsfas->nlevels;
  
  // Set up FAS
  PetscCall(SNESSetType(poissongibbsfas->fas, SNESFAS));
  PetscCall(SNESFASSetType(poissongibbsfas->fas, SNES_FAS_MULTIPLICATIVE));
  PetscCall(SNESFASSetLevels(poissongibbsfas->fas, nlevels, NULL));

  // Set the smoothers on all levels
  for (PetscInt ell=0;ell<nlevels;++ell) {
    SNES smoother;
    Vec solution;
    if (ell == 0) {
      PetscCall(SNESFASGetCoarseSolve(poissongibbsfas->fas, &smoother));
    } else {
      PetscCall(SNESFASGetSmoother(poissongibbsfas->fas, ell, &smoother));
    }    
    PetscCall(SNESSetApplicationContext(smoother, &poissongibbsfas->smoother_ctx[ell]));    
    PetscCall(SNESSetType(smoother, SNESPOISSONGIBBS));        
    PetscCall(SNESSetUp(smoother));
  }
    
  // Set intergrid operators on all levels
  Mat Id;
  PetscInt ndof, nobs;
  PetscCall(MatGetSize(ctx->B_meas, &ndof, &nobs));
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, nobs, nobs, PETSC_DECIDE, PETSC_DECIDE, 1.0, &Id));
  PetscCall(PCGetInterpolations(poissongibbsfas->mg, &nlevels, &P));
  for (PetscInt ell=1;ell<nlevels;++ell) {
    Mat R;
    Mat P_T;
    PetscCall(SNESFASSetInterpolation(poissongibbsfas->fas, ell, P[ell-1]));
    PetscCall(MatTranspose(P[ell-1], MAT_INITIAL_MATRIX, &P_T));
    Mat blocks[4] = {P_T, NULL, NULL, Id};
    PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, blocks, &R));
    PetscCall(SNESFASSetRestriction(poissongibbsfas->fas, ell, R));
  }

  Vec b_rhs;
  SNESFunctionFn *f;
  SNES fine_smoother;
  if (nlevels == 1) {
      PetscCall(SNESFASGetCoarseSolve(poissongibbsfas->fas, &fine_smoother));
    } else {
      PetscCall(SNESFASGetSmoother(poissongibbsfas->fas, nlevels-1, &fine_smoother));
    }
  PetscCall(SNESGetFunction(fine_smoother, &b_rhs, &f, ctx));
  PetscCall(SNESSetFunction(poissongibbsfas->fas, b_rhs, f, ctx));
  PetscCall(SNESSetFunction(snes, b_rhs, f, ctx));

  PetscCall(SNESSetUp(poissongibbsfas->fas));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetUp_PoissonGibbsFAS(SNES snes)
{
  PetscFunctionBeginUser;
  PetscCall(setup_multigrid(snes));
  PetscCall(setup_fas(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_PoissonGibbsFAS(SNES snes, PetscOptionItems PetscOptionsObject)
{
  const char *pc_type;
  PetscBool isgamg, ismg;
  
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscFunctionBegin;
  // Set multigrid from options
  PetscCall(PCSetFromOptions(poissongibbsfas->mg));
  PetscCall(PCGetType(poissongibbsfas->mg, &pc_type));
  // Check that PC is 'mg' or 'gamg'
  PetscCall(PetscStrcmp(pc_type, PCGAMG, &isgamg));
  PetscCall(PetscStrcmp(pc_type, PCMG, &ismg));
  PetscCheck(isgamg || ismg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PC type must be mg or gamg, but got %s", pc_type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_PoissonGibbsFAS(SNES snes, PetscViewer viewer)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscFunctionBeginUser;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Underlying multigrid\n"));
  PetscCall(PCView(poissongibbsfas->mg,viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Underlying FAS\n"));
  PetscCall(SNESView(poissongibbsfas->fas,viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESCreate_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas;
  
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissongibbsfas));
  snes->data       = (void*)poissongibbsfas;
  
  snes->ops->solve           = SNESSample_PoissonGibbsFAS;
  snes->ops->destroy         = SNESDestroy_PoissonGibbsFAS;
  snes->ops->reset           = SNESReset_PoissonGibbsFAS;
  snes->ops->setup           = SNESSetUp_PoissonGibbsFAS;
  snes->ops->setfromoptions  = SNESSetFromOptions_PoissonGibbsFAS;
  snes->ops->view            = SNESView_PoissonGibbsFAS;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  
  // Create multigrid PC
  PetscCall(PCCreate(PETSC_COMM_WORLD,&poissongibbsfas->mg));
  // Create SNES
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&poissongibbsfas->fas));
  PetscFunctionReturn(PETSC_SUCCESS);
}
