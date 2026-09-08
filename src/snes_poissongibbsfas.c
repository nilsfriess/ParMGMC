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
  SNES fas;            // internal FAS
  PC mg;               // internal multigrid 
  PetscInt nlevels;    // number of multigrid levels
  PetscInt its;        // Number of iterations (=FAS cycles)
  PetscInt its_up;     // Number of pre-smoother iterations
  PetscInt its_down;   // Number of post-smoother iterations
  PetscInt its_coarse; // Number of coarse-smoother iterations
  PoissonGibbsCtx *smoother_ctx;
}  SNES_PoissonGibbsFAS;

/* Generate a new sample (computational routine) */
static PetscErrorCode SNESSample_PoissonGibbsFAS(SNES snes)
{
  Vec vec_rhs, vec_sol, z;
  SNES_PoissonGibbsFAS* poissongibbsfas;
  PoissonGibbsCtx* ctx;

  PetscFunctionBeginUser;  
  
  poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));  
  PetscCall(VecDuplicate(ctx->nu, &z));
  PetscCall(VecSet(z, 0.0));

  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, (Vec[]){snes->vec_rhs, ctx->nu}, &vec_rhs));
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, (Vec[]){snes->vec_sol, z}, &vec_sol));

  PetscCall(SNESSolve(poissongibbsfas->fas, vec_rhs, vec_sol));
  snes->reason = SNES_CONVERGED_ITS;  
  PetscCall(VecDestroy(&z));
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
      PoissonGibbsCtx ctx = poissongibbsfas->smoother_ctx[ell];
      if (ctx.Q_prec)
        PetscCall(MatDestroy(&ctx.Q_prec));
      if (ctx.B_meas)
        PetscCall(MatDestroy(&ctx.B_meas));
      if (ctx.event_counts)
        PetscCall(VecDestroy(&ctx.event_counts));      
      if (ctx.nu)
        PetscCall(VecDestroy(&ctx.nu));
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
    PetscCall(VecDuplicate(ctx->nu, &poissongibbsfas->smoother_ctx[ell].nu));
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

static PetscErrorCode SNESPoissonGibbs_Function(SNES snes, Vec y, Vec b, void* ctx) 
{
  Vec theta, f_rhs, nu;

  PoissonGibbsCtx* poissongibbs = (PoissonGibbsCtx*)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecNestGetSubVec(y, 0, &theta));
  PetscCall(VecNestGetSubVec(b, 0, &f_rhs));
  PetscCall(VecNestGetSubVec(b, 1, &nu));
  PetscCall(MatMult(poissongibbs->Q_prec,theta,f_rhs));
  PetscCall(MatMultTranspose(poissongibbs->B_meas,theta,nu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode set_function(SNES snes) {

  PoissonGibbsCtx* ctx;
  PetscInt ndof, nobs;
  Vec f_rhs, nu, b_rhs;

  PetscFunctionBeginUser;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  PetscCall(MatGetSize(ctx->B_meas,&ndof,&nobs));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ndof, &f_rhs));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, nobs, &nu));
  Vec subvecs[2] = {f_rhs, nu};
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, subvecs, &b_rhs));
  PetscCall(SNESSetFunction(snes, b_rhs, SNESPoissonGibbs_Function, ctx));    
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
    SNES smoother_up, smoother_down, smoother_coarse, level_snes;
    if (ell == 0) {
      PetscCall(SNESFASGetCoarseSolve(poissongibbsfas->fas, &smoother_coarse));
      PetscCall(SNESSetApplicationContext(smoother_coarse, &poissongibbsfas->smoother_ctx[ell]));
      PetscCall(SNESSetType(smoother_coarse, SNESPOISSONGIBBS));
      PetscCall(SNESPoissonGibbsSetIterations(smoother_coarse,poissongibbsfas->its_coarse));      
      PetscCall(set_function(smoother_coarse));              
      PetscCall(SNESSetUp(smoother_coarse));            
    } else {
      // Down smoother
      PetscCall(SNESFASGetSmootherDown(poissongibbsfas->fas, ell, &smoother_down));
      PetscCall(SNESSetApplicationContext(smoother_down, &poissongibbsfas->smoother_ctx[ell]));
      PetscCall(SNESSetType(smoother_down, SNESPOISSONGIBBS));
      PetscCall(SNESPoissonGibbsSetIterations(smoother_down,poissongibbsfas->its_down));
      PetscCall(set_function(smoother_down));
      PetscCall(SNESSetUp(smoother_down));            
      // Up smoother
      PetscCall(SNESFASGetSmootherUp(poissongibbsfas->fas, ell, &smoother_up));
      PetscCall(SNESSetApplicationContext(smoother_up, &poissongibbsfas->smoother_ctx[ell]));
      PetscCall(SNESSetType(smoother_up, SNESPOISSONGIBBS));
      PetscCall(SNESPoissonGibbsSetIterations(smoother_up,poissongibbsfas->its_up));      
      PetscCall(set_function(smoother_up));     
      PetscCall(SNESSetUp(smoother_up));       
    }
    PetscCall(SNESFASGetCycleSNES(poissongibbsfas->fas, ell, &level_snes));
    PetscCall(SNESSetApplicationContext(level_snes, &poissongibbsfas->smoother_ctx[ell]));    
    PetscCall(set_function(level_snes));
  }  
  // Set intergrid operators on all levels
  Mat Id;
  PetscInt ndof, nobs;
  PetscCall(MatGetSize(ctx->B_meas, &ndof, &nobs));
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, nobs, nobs, PETSC_DECIDE, PETSC_DECIDE, 1.0, &Id));
  PetscCall(PCGetInterpolations(poissongibbsfas->mg, &nlevels, &P));
  for (PetscInt ell=1;ell<nlevels;++ell) {
    Mat P_T, R_hat, R_2x2, P_2x2, I_2x2;
    // Prolongation is given by
    //                          [ P 0 ]
    //                          [ 0 0 ]
    Mat blocks_prolong[4] = {P[ell-1], NULL, NULL, Id};
    PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, blocks_prolong, &P_2x2));
    PetscCall(MatNestSetVecType(P_2x2, VECNEST));
    PetscCall(SNESFASSetInterpolation(poissongibbsfas->fas, ell, P_2x2));
    // Restriction is given by
    //                          [ P^T 0 ]
    //                          [ 0   I ]    
    PetscCall(MatTranspose(P[ell-1], MAT_INITIAL_MATRIX, &P_T));
    Mat blocks_restrict[4] = {P_T, NULL, NULL, Id};
    PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, blocks_restrict, &R_2x2));
    PetscCall(MatNestSetVecType(R_2x2, VECNEST));
    PetscCall(SNESFASSetRestriction(poissongibbsfas->fas, ell, R_2x2));
    // Injection is given by
    //                          [ 0   0 ]
    //                          [ 0   I ]        
    PetscCall(MatDuplicate(P_T, MAT_DO_NOT_COPY_VALUES, &R_hat));
    PetscCall(MatZeroEntries(R_hat));
    Mat blocks_inject[4] = {R_hat, NULL, NULL, Id};
    PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, blocks_inject, &I_2x2));    
    PetscCall(MatNestSetVecType(I_2x2, VECNEST));
    PetscCall(SNESFASSetInjection(poissongibbsfas->fas, ell, I_2x2));
  }  
  // Do exactly one iteration
  PetscCall(SNESSetTolerances(poissongibbsfas->fas, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT, 1, PETSC_DEFAULT));
  PetscCall(SNESSetForceIteration(poissongibbsfas->fas,true));  
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

  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson Gibbs options");  
  PetscCall(PetscOptionsInt("-snes_poissongibbsfas_smoothdown", "Number of Poisson Gibbs pre-smoother iterations", NULL, poissongibbsfas->its_down, &poissongibbsfas->its_down, NULL));  
  PetscCall(PetscOptionsInt("-snes_poissongibbsfas_smoothup", "Number of Poisson Gibbs post-smoother smoother iterations", NULL, poissongibbsfas->its_up, &poissongibbsfas->its_up, NULL));    
  PetscCall(PetscOptionsInt("-snes_poissongibbsfas_smoothcoarse", "Number of Poisson Gibbs coarse-smoother iterations", NULL, poissongibbsfas->its_coarse, &poissongibbsfas->its_coarse, NULL));  
  PetscOptionsHeadEnd();  
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

  poissongibbsfas->its = 1;
  poissongibbsfas->its_up = 1;
  poissongibbsfas->its_down = 1;
  poissongibbsfas->its_coarse = 1;
    
  // Create multigrid PC
  PetscCall(PCCreate(PETSC_COMM_WORLD,&poissongibbsfas->mg));
  // Create SNES
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&poissongibbsfas->fas));
  PetscFunctionReturn(PETSC_SUCCESS);
}
