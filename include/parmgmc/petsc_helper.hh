#pragma once

#include <petscsys.h>

namespace parmgmc {
struct PetscHelper {
  PetscHelper(int *argc, char ***argv) {
    PetscInitialize(argc, argv, NULL, NULL);
  }

  ~PetscHelper() { PetscFinalize(); }
};
} // namespace parmgmc
