import numpy as np
from scipy.stats import levy_stable

alpha = 1.5
beta = 0.0
timeseries = levy_stable.rvs(alpha, beta, size=1000000)

outfile = "levy_stable_alpha1.5_T1M.dat"
with open(outfile, 'w') as file:
    for j in range(len(timeseries)):
        file.write(f"{timeseries[j]}\n")

