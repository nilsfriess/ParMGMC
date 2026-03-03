#pragma once

#include <petscsystypes.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscdm.h>

class Problem {
public:
  virtual PetscErrorCode GetPrecisionMat(Mat *)                                                             = 0;
  virtual PetscErrorCode GetRHSVec(Vec *)                                                                   = 0;
  virtual PetscErrorCode GetMeasurementVec(Vec *)                                                           = 0;
  virtual PetscErrorCode VisualiseResults(Vec last_sample = nullptr, Vec mean = nullptr, Vec var = nullptr) = 0;
  virtual PetscErrorCode GetDM(DM *dm)
  {
    PetscFunctionBeginUser;
    *dm = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  virtual ~Problem() = default;
};
