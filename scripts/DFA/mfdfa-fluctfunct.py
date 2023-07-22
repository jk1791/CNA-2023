import numpy as np
import math

# -------------------------------------------------------------- #

def fluctuation_function ():

    global fluctfunct,logwin_step
    profile = np.zeros(ts_lngth)
    average = np.mean(signal)
    profile[0] = signal[0] - average
    for i in range(1, ts_lngth):
        profile[i] = profile[i - 1] + signal[i] - average

    logwin_step = (np.log(max_scale) - np.log(min_scale)) / (num_scales - 1)
    for m in range(num_scales):
        act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
        if m == num_scales - 1:
            act_scale = max_scale

        win_num = ts_lngth // act_scale
        segment_var = np.zeros(2 * win_num)
        for w in range(win_num):
            win_pos = w * act_scale
            window = profile[win_pos:win_pos + act_scale]
            polynomial = np.polyfit(np.arange(1, act_scale + 1), window, polyorder)
            segment_var[w] = np.mean((window - np.polyval(polynomial, np.arange(1, act_scale + 1)))**2)

        for w in range(win_num):
            win_pos = (ts_lngth % act_scale) + w * act_scale
            window = profile[win_pos:win_pos + act_scale]
            polynomial = np.polyfit(np.arange(1, act_scale + 1), window, polyorder)
            segment_var[win_num + w] = np.mean((window - np.polyval(polynomial, np.arange(1, act_scale + 1)))**2)

        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            fluctfunct[m, j] = np.mean(segment_var**(q_act / 2))
            fluctfunct[m, j] = fluctfunct[m, j]**(1.0 / q_act)

    return fluctfunct

# -------------------------------------------------------------- #

def output1():
    outfile = infiles[act_file][:-4] + '_fluctfunct.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {fluctfunct[m,j]}\n')
            file.write('\n')

# -------------------------------------------------------------- #

#    MFDFA ver. 1.0 converted to python (18.07.2023)
#
#    infiles = []
infiles = []
#    file_num = int(input("number of input files: "))
#    if file_num == 1:
infiles.append('')
#        infiles.append(input("input file: "))
#    else:
#        infiles.append(input("Name of a file with file names: "))
#    with open(infiles[0], 'r') as f:
#        infiles.extend(f.read().splitlines())
#    column = int(input('signal column number : '))
#    min_scale = int(input('minimum scale : '))
#    num_scales = int(input('number of different scales : '))
#    q_min = float(input('minimum q : '))
#    q_max = float(input('maximum q : '))
#    q_step = float(input('q step : '))
#    polyorder = int(input('fitting polynomial order : '))
#    ts_lngth = int(input('the number of analysed points (0 = all): '))
#    num_zeros = 0
#    check_remove = input('remove constant-value intervals ? [ p / r ] : ') == 'r'
#    if check_remove:
#        num_zeros = int(input('max length of a constant-value interval : '))
#    flag = False
#    letter = input('signed variant? [y/n] : ')
#    if letter == 'y':
#        flag = True

# ====================================================================================== #

#    INPUT PARAMETERS:

# input file
infiles.append("arfima_d0.4_T65k_arfima_d-0.2_T65k_segmentwise_mixed6%.dat")
# data column
column = 1
# minimum scale & number of different scales
min_scale = 20
num_scales = 50
# range of the Renyi parameter q
q_min = -4
q_max = 4
q_step = 0.2
# detrending polynomial order
polyorder = 2
# number of data points to be considered (0 - all)
ts_lngth = 0
# remove sequences of zeros?  (True / False)
check_remove = True
# if so, how long is the maximum allowed sequence of zeros
if check_remove: num_zeros = 10
# use sign-preserving method (MFCCA)?  (True / False)
flag = True

# ====================================================================================== #

file_num = 1
q_num = int((q_max - q_min) / q_step) + 1
logwin_step = 0.0
num_points = [0] * file_num
if not check_remove: num_zeros = 0

signal = []
for k in range(1,len(infiles)):
    act_file = k
    with open(infiles[k], 'r') as f:
        lines = f.readlines() 
        num_points = len(lines)
        for line in lines:
            items = line.split()
            signal.append(float(items[column - 1]))

    if ts_lngth > 0 and ts_lngth < num_points:
        del signal[:ts_lngth]
    else:
        ts_lngth = num_points
       
    error = []
    num_err = 0
    eqval_lngth = 1
    for i in range(2,ts_lngth):
        if signal[i] == signal[i-1]:
            eqval_lngth += 1
            check_ok = False
        else:
            check_ok = True
            if eqval_lngth > num_zeros:
                error.append([i - eqval_lngth,eqval_lngth])
                num_err += 1
            eqval_lngth = 1
        if i == ts_lngth and eqval_lngth > num_zeros:
            error.append([i - eqval_lngth + 1,eqval_lngth])
            num_err += 1

    num_err -= 1
    for err in range(num_err,-1,-1):
        del signal[error[err][0]:error[err][0]+error[err][1]]

    ts_lngth = len(signal)
    max_scale = int(ts_lngth / 4)

    fluctfunct = np.zeros((num_scales, q_num))
    fluctfunct = fluctuation_function()
    output1()

print("")
