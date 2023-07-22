import numpy as np
import math

# -------------------------------------------------------------- #

def power_fit(vct_lngth, num_scales, vector):

    coeff = [0.0] * 2
    sigma = [0.0] * 2
    sum_x = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    sum_y = 0.0

    for i in range(vct_lngth):
        if vector[i][0] > 0.0:
            vector[i][0] = math.log(vector[i][0])
        if vector[i][1] > 0.0:
            vector[i][1] = math.log(vector[i][1])

    for i in range(vct_lngth):
        sum_x += vector[i][0]
        sum_xx += vector[i][0] ** 2.0
        sum_y += vector[i][1]
        sum_xy += vector[i][0] * vector[i][1]
    
    delta = float(vct_lngth) * sum_xx - sum_x ** 2.0
    coeff[0] = (sum_xx * sum_y - sum_x * sum_xy) / delta
    coeff[1] = (float(vct_lngth) * sum_xy - sum_x * sum_y) / delta
    sigma[0] = sum_xx / delta
    sigma[1] = float(vct_lngth) / delta
    covariance = -sum_x / delta
    power_fit = coeff[1]
    
    return power_fit

# -------------------------------------------------------------- #

def derivative(q_num, hurst_est):

    triple = [[0.0] * 2 for _ in range(3)]
    coeff = [0.0] * 3
    expression = [0.0] * 2
    hurst_deriv = np.zeros(q_num)
    for j in range(q_num - 2):
        for i in range(-1, 2):
            triple[i + 1][0] = q_min + float(j + i + 1) * q_step
            if triple[i + 1][0] == 0.0:
                triple[i + 1][0] = triple[i + 1][0] + q_step / 4.0
            triple[i + 1][1] = hurst_est[j + i + 1]
        
        expression[0] = (triple[2][1] - triple[1][1]) * ((triple[0][0]) ** 2.0 - (triple[1][0]) ** 2.0) + (triple[1][1] - triple[0][1]) * ((triple[2][0]) ** 2.0 - (triple[1][0]) ** 2.0)
        expression[1] = (triple[1][0] - triple[0][0]) * ((triple[2][0]) ** 2.0 - (triple[1][0]) ** 2.0) + (triple[2][0] - triple[1][0]) * ((triple[0][0]) ** 2.0 - (triple[1][0]) ** 2.0)
        coeff[1] = expression[0] / expression[1]
        coeff[0] = (coeff[1] * (triple[1][0] - triple[0][0]) + triple[0][1] - triple[1][1]) / ((triple[0][0]) ** 2.0 - (triple[1][0]) ** 2.0)
        coeff[2] = triple[0][1] - coeff[0] * (triple[0][0]) ** 2.0 - coeff[1] * triple[0][0]
        q_act = q_min + float(j + 1) * q_step
        if q_act == 0.0:
            q_act = q_act + q_step / 4.0
        hurst_deriv[j + 1] = 2.0 * coeff[0] * q_act + coeff[1]
        if j == 0:
            hurst_deriv[j] = 2.0 * coeff[0] * q_min + coeff[1]
        if j == q_num - 3:
            hurst_deriv[j + 2] = 2.0 * coeff[0] * (q_act + q_step) + coeff[1]

    return hurst_deriv

# -------------------------------------------------------------- #

def output2(alpha, f_alpha, tau, hurst_est):

    outfile = infiles[act_file][:-4] + '_tau.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {tau[j]}\n")
    
    outfile = infiles[act_file][:-4] + '_falpha.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{alpha[j]} {f_alpha[j]}\n")

    outfile = infiles[act_file][:-4] + '_hurst.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {hurst_est[j]}\n")

# -------------------------------------------------------------- #

def partition_function ():

    global partfunct,logwin_step
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

            partfunct[m, j] = np.mean(segment_var**(q_act / 2))
            partfunct[m, j] = partfunct[m, j]**(1.0 / q_act)

    return partfunct

# -------------------------------------------------------------- #

def output1():
    outfile = infiles[act_file][:-4] + '_partfunct.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {partfunct[m,j]}\n')
            file.write('\n')

# -------------------------------------------------------------- #

def legendre_transform():

    hurst_est = np.zeros(q_num)
    vector = np.zeros((num_scales,2))

    for j in range(q_num):
        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[m,j]
                v += 1

        vct_lngth = v - 1
        hurst_est[j] = power_fit(vct_lngth,num_scales,vector)

    hurst_deriv = np.zeros(q_num)
    alpha = np.zeros(q_num)
    f_alpha = np.zeros(q_num)
    tau = np.zeros(q_num)

    hurst_deriv = derivative(q_num,hurst_est)

    for j in range(q_num):
        q_act = q_min + j * q_step
        if q_act == 0.0:
            q_act += q_step / 4.0
        alpha[j] = hurst_est[j] + q_act * hurst_deriv[j]
        f_alpha[j] = q_act * (alpha[j] - hurst_est[j]) + 1.0
        tau[j] = q_act * hurst_est[j] - 1.0

    output2(alpha,f_alpha,tau,hurst_est)

# -------------------------------------------------------------- #

print("\n     MFDFA ver. 1.0 converted to python (18.07.2023)\n\n")

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
infiles.append("gauss_ts1_partfunct.dat")
min_range = 10
max_range = 1000
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
if check_remove: num_zeros = 3
# use sign-preserving method (MFCCA)?  (True / False)
flag = False

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

    partfunct = np.zeros((num_scales, q_num))
    partfunct = partition_function()
    output1()

    min_range = 10
    max_range = max_scale

    legendre_transform()

print("")

