#pragma once

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

#include <petscsys.h>
#include <petscvec.h>

namespace parmgmc {
template <class Sampler> class SampleChain {
public:
  template <typename... Args>
  SampleChain(Args &&...sampler_args)
      : sampler(std::forward<Args>(sampler_args)...), save_samples{true},
        est_mean_online{true} {}

  PetscErrorCode sample(Vec sample, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    if (samples.size() == 0) {
      PetscCall(VecDuplicate(sample, &mean_));
      PetscCall(VecZeroEntries(mean_));
    }

    for (std::size_t n = 0; n < n_samples; ++n) {
      PetscCall(sampler.sample(sample));

      if (save_samples) {
        PetscCall(add_sample(sample));

        if (est_mean_online)
          PetscCall(update_mean());
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode add_sample(Vec sample) {
    Vec new_sample;
    PetscFunctionBeginUser;
    PetscCall(VecDuplicate(sample, &new_sample));
    PetscCall(VecCopy(sample, new_sample));
    samples.push_back(new_sample);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode get_mean(Vec mean) {
    PetscFunctionBeginUser;

    if (est_mean_online)
      PetscCall(VecCopy(mean_, mean));
    else
      PetscCall(compute_mean(mean));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode update_mean() {
    assert(samples.size() >= 1);

    PetscFunctionBeginUser;

    if (samples.size() == 1)
      PetscCall(VecCopy(samples.back(), mean_));
    else {
      const auto n_sample = static_cast<PetscScalar>(samples.size());
      PetscCall(VecAXPBY(
          mean_, 1. / n_sample, (n_sample - 1) / n_sample, samples.back()));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void disable_save() { save_samples = false; }
  void enable_save() { save_samples = true; }

  void disable_est_mean_online() { est_mean_online = false; }

private:
  PetscErrorCode compute_mean(Vec mean) const {
    assert(samples.size() >= 1);

    PetscFunctionBeginUser;
    VecZeroEntries(mean);

    const auto n_samples = samples.size();
    for (auto sample : samples)
      PetscCall(VecAXPY(mean, 1. / n_samples, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Vec mean_;

  Sampler sampler;

  std::vector<Vec> samples;

  bool save_samples;
  bool est_mean_online;
};
} // namespace parmgmc
