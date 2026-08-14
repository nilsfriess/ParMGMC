# Data source: https://rdrr.io/cran/geoR/man/rongelap.html
# Peter J. Diggle , Paulo J. Ribeiro: "Model based geostatistics", Springer 2007 https://link.springer.com/book/10.1007/978-0-387-48536-2

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

coastline_file = "rongelap_coastline.csv"
measurement_file = "rongelap_gamma.csv"

# Coastline
data_coastline = pd.read_csv(coastline_file)
X_coastline = data_coastline.x.to_numpy()
Y_coastline = data_coastline.y.to_numpy()

# Measurements
data_gamma = pd.read_csv(measurement_file)
X_gamma = data_gamma.x.to_numpy()
Y_gamma = data_gamma.y.to_numpy()
rate = data_gamma.gamma_counts.to_numpy() / data_gamma.measurement_time.to_numpy()
print(f"rate in {np.min(rate)} {np.max(rate)}")

# Plot
plt.clf()
ax = plt.gca()
ax.set_aspect("equal")
plt.plot(X_coastline, Y_coastline, linewidth=1, color="black")
plt.fill(X_coastline, Y_coastline, color="lightgray")
sc = plt.scatter(X_gamma, Y_gamma, c=rate, s=1)
plt.colorbar(sc, ax=ax, orientation="horizontal")
plt.savefig("rongelap.pdf", bbox_inches="tight")
