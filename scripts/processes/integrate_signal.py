import numpy as np

# -------------------------------------------------------------- #

#    randomize time series by shuffling: data points or FFT phases (21.07.2023)
#
#    INPUT PARAMETERS:

# input file
infile = "data/uncorrelated_gaussian_noise_T10k_ver-1.dat"
# data column
column = 1

# ============================================================ 

signal = []

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal)

integrated = np.zeros(ts_lngth)

integrated[0] = signal[0]
for i in range(1,ts_lngth):
    integrated[i] = integrated[i-1] + signal[i]

outfile = infile[:-4] + f'_integrated.dat'
with open(outfile, 'w') as file:
    for i in range(ts_lngth):
        file.write(f"{integrated[i]}\n")

