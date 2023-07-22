import numpy as np
import math
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
from scipy import special
import matplotlib.pyplot as plt

# -------------------------------------------------------------- #

#    calculate Pearson cross-correlation coefficient (20.07.2023)
#
#    INPUT PARAMETERS:

infiles = []
column = [1,1]
# input files
infiles.append("pareto_signed_b4.0_T100k.dat")
infiles.append("pareto_signed_b6.0_T100k.dat")
# time series columns:
column[0] = 1
column[1] = 1
# perform stationarity test? [y / n]
letter = 'y'
flag = False
if letter == 'y': flag = True

# ============================================================ 

signal = [[],[]]

for k in range(2):
    with open(infiles[k], 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split()
            signal[k].append(float(items[column[k]-1]))

if len(signal[0]) != len(signal[1]):
    print ("unequal signal lengths - exiting...\n")
    quit()

ts_lngth = len(signal[0]) + 1

crosscoeff = pearsonr(signal[0],signal[1])

print ('\n Pearson coefficient: %.4f with p-value: %f\n' % (crosscoeff[0], crosscoeff[1]))

if flag :
    ADF_test = [0.0,0.0]
    ADF_test[0] = adfuller(signal[0])
    ADF_test[1] = adfuller(signal[1])
    print (f' ADF test for {infiles[0]}: %f with p-value: %f' % (ADF_test[0][0], ADF_test[0][1]))
    print (f' ADF test for {infiles[1]}: %f with p-value: %f' % (ADF_test[1][0], ADF_test[1][1]))
    print('\n Critical values:')
    for key,value in ADF_test[0][4].items():
        print('    \t%s: %.3f\n' % (key, value))

