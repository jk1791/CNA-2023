import numpy as np
import math
from scipy import special
import matplotlib.pyplot as plt

# -------------------------------------------------------------- #

print("\n     plot time series (20.07.2023)\n\n")

#    INPUT PARAMETERS:

# input file
infile = "levy_stable_alpha1.5_T100k.dat"
# number of columns
num_col = 1
# time series column
column = 1

signal = []

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal) + 1
points = np.arange(1,ts_lngth,1)

fig = plt.figure(figsize=(10, 6))
plt.title(f'{infile}')
plt.xlabel('points',fontsize=15)
plt.ylabel('signal value',fontsize=15)
plt.plot(points,signal)
plt.xlim(0.0,float(ts_lngth))

min_yaxis = 0.0
if min(signal) < 0: min_yaxis = 1.2 * min(signal)
max_yaxis = 1.2 * max(signal)
if max(signal) < 0: max_yaxis = 0.0

plt.ylim(min_yaxis,max_yaxis)
plt.legend()
plt.grid()

plt.show()

