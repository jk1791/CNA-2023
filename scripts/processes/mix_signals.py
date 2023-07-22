import numpy as np
import math

# -------------------------------------------------------------- #

#    add features to a signal (21.07.2023)
#
#    INPUT PARAMETERS:

infiles = []
# input file
infiles.append("arfima_d0.4_T65k.dat")
infiles.append("arfima_d-0.2_T65k.dat")
# output file
outfile = infiles[0][:-4] + "_" + infiles[1][:-4] + "_segmentwise_mixed6%.dat"
# data column
column = 1

signal = [[],[]]

with open(infiles[0], 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal[0].append(float(items[column-1]))
with open(infiles[1], 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal[1].append(float(items[column-1]))

if len(signal[0]) != len(signal[1]):
    print ("unequal signal lengths - exiting...\n")
    quit()

ts_lngth = len(signal[0])

mixed_signal = signal[0]

mixed_signal[100:500] = signal[1][100:500]
mixed_signal[5600:7500] = signal[1][5600:7500]
mixed_signal[15000:17000] = signal[1][15000:17000]

with open(outfile, 'w') as file:
    for i in range(ts_lngth):
        file.write(f'{mixed_signal[i]}\n')

