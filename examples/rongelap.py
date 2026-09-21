from dataclasses import dataclass
import numpy as np
import scipy as sp
import pandas as pd
import sys
import tqdm
import argparse

import petsc4py

petsc4py.init(sys.argv)

import pymgmc

from petsc4py import PETSc

petsc_options = PETSc.Options()
petsc_options["options_left"] = False


@dataclass
class Rongelap:
    locations: np.array
    event_counts: np.array
    measurement_intervals: np.array

    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.locations = np.stack([df[dim].to_numpy() for dim in ("x", "y")]).T
        self.event_counts = df["gamma_counts"].to_numpy()
        self.measurement_intervals = df["measurement_time"].to_numpy()


def precision_matrix(locations, sigma, delta, alpha):
    n_obs = locations.shape[0]
    distances = np.empty(shape=(n_obs, n_obs))
    for i in range(n_obs):
        distances[i, :] = np.linalg.norm(locations[:, :] - locations[i, :], axis=1)
    covariance_matrix = sigma**2 * np.exp(-((alpha * distances) ** delta))
    precision_matrix = np.linalg.inv(covariance_matrix)
    return precision_matrix


def petsc_mat(A):
    sparse_data = sp.sparse.csr_matrix(A)
    A_sparse = PETSc.Mat().createAIJ(
        size=sparse_data.shape,
        csr=(sparse_data.indptr, sparse_data.indices, sparse_data.data),
    )
    A_sparse.assemblyBegin()
    A_sparse.assemblyEnd()
    return A_sparse


if __name__ == "__main__":
    # Model parameters
    alpha = 1 / 4000
    delta = 0.7
    sigma = 1
    beta = 0.0
    n_samples = 10000
    n_warmup = 100
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="results.txt", help="name of output file"
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="index of location to sample"
    )
    args, _ = parser.parse_known_args()
    # File with data
    filename = "../data/rongelap_gamma.csv"
    rongelap = Rongelap(filename)
    n_obs = len(rongelap.event_counts)
    snes = PETSc.SNES().create()
    snes.setFromOptions()

    event_counts = PETSc.Vec().createWithArray(rongelap.event_counts)
    measurement_intervals = PETSc.Vec().createWithArray(rongelap.measurement_intervals)
    Q_prec = petsc_mat(precision_matrix(rongelap.locations, sigma, delta, alpha))
    B_meas = petsc_mat(np.eye(n_obs))
    pymgmc.SetPoissonCtx(
        snes,
        event_counts,
        measurement_intervals,
        beta,
        Q_prec,
        B_meas,
    )
    f_rhs = PETSc.Vec().createWithArray(np.zeros(shape=n_obs))
    u = PETSc.Vec().createWithArray(np.zeros(shape=n_obs))
    data = np.empty(shape=n_samples, dtype=np.float64)
    for k in tqdm.tqdm(range(n_warmup + n_samples)):
        snes.solve(f_rhs, u)
        theta = u.getArray()
        data[k - n_warmup] = np.exp(theta[args.idx])

    if snes.getType() == "poissonmala":
        acceptance_rate = pymgmc.GetMALAAcceptanceRate(snes)
        print(f"MALA acceptance rate = {100 * acceptance_rate:6.2f} %")

    np.savetxt(args.output, data, delimiter=",")
