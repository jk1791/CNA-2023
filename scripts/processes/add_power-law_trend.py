import numpy as np
import math

# -------------------------------------------------------------- #

#    add features to a signal (21.07.2023)
#
#    INPUT PARAMETERS:

# input file
infile = "uncorrelated_gaussian_noise_T100k.dat"
# output file
outfile = infile[:-4] + "_power-law_trend_lambda-2.5.dat"
# data column
column = 1
# scaling exponent
exponent = -2.5

signal = []
scaling_factor = 2.5

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal)

signal_modified = np.zeros(ts_lngth)

signal_modified[0] = signal[0]

for i in range(1,ts_lngth):
    if signal[i] != 0.0:
        signal_modified[i] = signal[i] + (float(i))**exponent * scaling_factor

with open(outfile, 'w') as file:
    for i in range(ts_lngth):
        file.write(f'{signal_modified[i]}\n')

