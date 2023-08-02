# Author: Jaroslaw Kwapien, IFJ PAN, Kraków, Poland
# MIT License

import numpy as np
import argparse
import sys

# -------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Randomization of time series")
parser.add_argument("--col", help = "time series column (default: 1)")
parser.add_argument("--numrealiz", help = "number of independent realizations (default: 1)")
parser.add_argument("--fourier", help = "Fourier surrogates [y / n] (default: n)")
parser.add_argument('filename', help="name of the output data file (default: poisson.dat")

args = parser.parse_args()

if args.col is None:
    column = 1
else:
    column = int(args.col)
if args.numrealiz is None:
    num_surr = 1
else:
    num_surr = int(args.numrealiz)
if args.fourier is None:
    four_flag = False
else:
    letter = int(args.fourier)
    if letter == 'y': four_flag = True
if args.filename is None:
    infile = "type_file_name_here.dat"
else:
    infile = args.filename

# -------------------------------------------------------------- #

signal = []

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal)

surrogates = np.zeros((num_surr,ts_lngth))

for j in range(num_surr):
    if four_flag:
        fft = np.fft.rfft(signal)
        fft_ampl = np.abs(fft)
        random_phases = np.exp(1j * (np.angle(fft) + 2 * np.pi * np.random.random(len(fft))))
        surrogates[j] = np.fft.irfft(fft_ampl * random_phases,n=ts_lngth)
        outfile = infile[:-4] + f'_foursurr{j+1:02}.dat'
    else:
        surrogates[j] = np.random.permutation(signal)
        outfile = infile[:-4] + f'_surr{j+1:02}.dat'

    with open(outfile, 'w') as file:
        for i in range(ts_lngth):
            file.write(f"{surrogates[j][i]}\n")

