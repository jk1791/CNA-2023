# Author: Jaroslaw Kwapien, IFJ PAN, Kraków, Poland
# MIT License

import numpy as np
import math
from scipy import special
import matplotlib.pyplot as plt
import argparse
import sys

# -------------------------------------------------------------- #

def close_on_key(event):

    plt.close("all")

# -------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Plot time series")
parser.add_argument("--col", help = "data column (default: 1)")
parser.add_argument("--title", help = "plot title (default: empty)")
parser.add_argument('filename', help="name of the data file")

args = parser.parse_args()

if len(sys.argv) == 1: print ("\nNo data file name provided.\n")

#    INPUT PARAMETERS:

if args.filename is None:
    infile = "type_file_name_here.dat"
else:
    infile = args.filename
#   data column
if args.col is None:
    column = 1
else:
    column = int(args.col)
#   plot title
if args.title is None:
    plottitle = ''
else:
    plottitle = args.title

signal = []

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal) + 1
points = np.arange(1,ts_lngth,1)

fig = plt.figure(figsize=(10,6))
plt.title(f'{plottitle}')
plt.xlabel('points',fontsize=15)
plt.ylabel('signal value',fontsize=15)
plt.plot(points,signal)
plt.xlim(0.0,float(ts_lngth))

min_yaxis = 0.0
if min(signal) < 0: min_yaxis = 1.2 * min(signal)
max_yaxis = 1.2 * max(signal)
if max(signal) < 0: max_yaxis = 0.0

plt.ylim(min_yaxis,max_yaxis)
plt.grid()

fig.canvas.mpl_connect('key_press_event',close_on_key)

plt.show()

