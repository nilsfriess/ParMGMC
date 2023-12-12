#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <petscsys.h>
#include <petscvec.h>

namespace parmgmc {
template <class Sampler> class SampleChain {
public:
  template <typename... Args>
  SampleChain(Args &&...sampler_args)
      : sampler(std::forward<Args>(sampler_args)...), n_samples{0},
        save_samples{false}, est_mean_online{true} {}

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_steps = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_steps; ++n) {
      n_samples++;

      PetscCall(sampler.sample(sample, rhs));

      if (save_samples)
        PetscCall(add_sample(sample));

      if (est_mean_online)
        PetscCall(update_mean(sample));
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

  PetscErrorCode update_mean(Vec new_sample) {
    PetscFunctionBeginUser;

    if (n_samples == 1) {
      PetscCall(VecDuplicate(new_sample, &mean_));
      PetscCall(VecCopy(new_sample, mean_));
    } else {
      PetscCall(VecAXPBY(
          mean_, 1. / n_samples, (n_samples - 1.) / n_samples, new_sample));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void enable_save_samples() { save_samples = true; }
  void disable_save_samples() { save_samples = false; }

  void enable_est_mean_online() { est_mean_online = true; }
  void disable_est_mean_online() { est_mean_online = true; }

  void reset() {
    samples.clear();
    n_samples = 0;
  }

  ~SampleChain() {
    VecDestroy(&mean_);
    for (auto &sample : samples)
      VecDestroy(&sample);
  }

private:
  PetscErrorCode compute_mean(Vec mean) const {
    assert(samples.size() >= 1);

    if (!save_samples)
      throw std::runtime_error("[SampleChain::compute_mean] Cannot compute "
                               "mean when save_samples is not enabled");

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
  std::size_t n_samples;

  bool save_samples;
  bool est_mean_online;
};
} // namespace parmgmc
