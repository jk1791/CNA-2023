import numpy as np

# -------------------------------------------------------------- #

#    randomize time series by shuffling: data points or FFT phases (21.07.2023)
#
#    INPUT PARAMETERS:

# input file
infile = "pareto_signed_b4.0_T100k.dat"
# data column
column = 1
# number of realizations
num_surr = 1
# randomization method [f - Fourier / s - shuffling]
letter = 'f'

# ============================================================ 

flag = False
if letter == 'f': flag = True

signal = []

with open(infile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split()
        signal.append(float(items[column-1]))

ts_lngth = len(signal)

surrogates = np.zeros((num_surr,ts_lngth))

for j in range(num_surr):
    if flag:
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

