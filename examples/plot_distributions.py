import numpy as np
import scipy as sp
import emcee
from matplotlib import pyplot as plt


samplers = ["poissonmala", "poissongibbs", "poissongibbsfas", "vangelis"]
n_samplers = len(samplers)
idx = 4

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
axs = axs.flatten()

data = {}
for sampler in samplers:
    filename = f"results_{sampler}.txt"
    data[sampler] = np.loadtxt(filename, delimiter=",")


for ax, sampler in zip(axs, samplers):
    ax.set_xlabel(sampler)


for i, sampler in enumerate(samplers):
    theta = data[sampler]
    mean = np.average(theta)
    std = np.std(theta)
    iact = emcee.autocorr.integrated_time(theta, quiet=True)[0]
    print(f"==== {sampler} ====")
    print(f"  mean = {mean:8.4f}")
    print(f"  std  = {std:8.4f}")
    print(f"  IACT = {iact:8.4f}")
    print()

    axs[i].hist(theta, bins=32, density=True)

print("KS test:")
for i in range(len(samplers)):
    for j in range(i):
        res = sp.stats.ks_2samp(data[samplers[i]], data[samplers[j]])
        print(
            f"  {samplers[i]:16s} - {samplers[j]:16s} : {res.statistic:8.2e} / {100 * res.pvalue:6.4f}%"
        )

plt.savefig("histograms.pdf", bbox_inches="tight")
