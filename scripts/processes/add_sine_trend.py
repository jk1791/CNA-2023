import numpy as np
import math

# -------------------------------------------------------------- #

#    add features to a signal (21.07.2023)
#
#    INPUT PARAMETERS:

# input file
infile = "finnegans_wake_sentences.txt"
# output file
outfile = infile[:-4] + "_sine_trend.dat"
# data column
column = 1

signal = []
scaling_factor = 0.5

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal)
std_dev = np.std(signal)

signal_modified = np.zeros(ts_lngth)

for i in range(ts_lngth):
    signal_modified[i] = signal[i] + math.sin(float(i)/200.0) * std_dev * scaling_factor

with open(outfile, 'w') as file:
    for i in range(ts_lngth):
        file.write(f'{signal_modified[i]}\n')

