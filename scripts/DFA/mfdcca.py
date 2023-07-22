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

def output2(alpha, f_alpha, tau, lambda_est, hx_est, hy_est, hxy_est):

    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_tau.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {tau[j]}\n")
    
    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_falpha.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{alpha[j]} {f_alpha[j]}\n")

    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_lambdaq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {lambda_est[j]}\n")

    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_hxyq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {hxy_est[j]}\n")

# -------------------------------------------------------------- #

def partition_function (flag):

    global partfunct,logwin_step

    average = [0] * 2
    profile = np.zeros((file_num,ts_lngth))
    for k in range(file_num):
        average[k] = np.mean(signal[k])
        profile[k][0] = signal[k][0] - average[k]
        for i in range(1, ts_lngth):
            profile[k][i] = profile[k][i - 1] + signal[k][i] - average[k]

    logwin_step = (np.log(max_scale) - np.log(min_scale)) / (num_scales - 1)
    for m in range(num_scales):
        window_lngth = round(np.exp(np.log(min_scale) + m * logwin_step))
        if m == num_scales - 1:
            window_lngth = max_scale

        win_num = ts_lngth // window_lngth
        segment_var = np.zeros((3,2 * win_num))
        window = [[] for _ in range(file_num)]
        polynomial = [[] for _ in range(file_num)]
        for w in range(win_num):
            win_pos = w * window_lngth
            for k in range(file_num):
                window[k] = profile[k][win_pos:win_pos + window_lngth]
                polynomial[k] = np.polyfit(np.arange(1, window_lngth + 1), window[k], polyorder)
                segment_var[k,w] = np.mean((window[k] - np.polyval(polynomial[k], np.arange(1, window_lngth + 1)))**2)
            segment_var[2,w] = np.mean((window[0] - np.polyval(polynomial[0], np.arange(1, window_lngth + 1))) * (window[1] - np.polyval(polynomial[1], np.arange(1, window_lngth + 1))))

        for w in range(win_num):
            win_pos = (ts_lngth % window_lngth) + w * window_lngth
            for k in range(file_num):
                window[k] = profile[k][win_pos:win_pos + window_lngth]
                polynomial[k] = np.polyfit(np.arange(1, window_lngth + 1), window[k], polyorder)
                segment_var[k,win_num + w] = np.mean((window[k] - np.polyval(polynomial[k], np.arange(1, window_lngth + 1)))**2)
            segment_var[2,win_num + w] = np.mean((window[0] - np.polyval(polynomial[0], np.arange(1, window_lngth + 1))) * (window[1] - np.polyval(polynomial[1], np.arange(1, window_lngth + 1))))

        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0
            if flag == False :
                partfunct[2,m,j] = np.mean(abs(segment_var[2])**(q_act / 2))
            else:
                partfunct[2,m,j] = np.mean(np.sign(segment_var[2]) * (abs(segment_var[2]))**(q_act / 2))

#            if j == 0 and m == num_scales - 1:
#                print (segment_var[2])
#                print (abs(segment_var[2])**(q_act / 2))

            partfunct[2,m,j] = np.sign(partfunct[2,m,j]) * abs(partfunct[2,m,j])**(1.0 / q_act)

            partfunct[0,m,j] = np.mean(segment_var[0]**(q_act / 2))
            partfunct[0,m,j] = partfunct[0,m,j]**(1.0 / q_act)
            partfunct[1,m,j] = np.mean(segment_var[1]**(q_act / 2))
            partfunct[1,m,j] = partfunct[1,m,j]**(1.0 / q_act)

    return partfunct

# -------------------------------------------------------------- #

def output1():
    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_partfunct.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                window_lngth = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{window_lngth} {partfunct[2,m,j]}\n')
            file.write('\n')

    outfile = infiles[0][:-4] + '_partfunct0.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                window_lngth = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{window_lngth} {partfunct[0,m,j]}\n')
            file.write('\n')

    outfile = infiles[1][:-4] + '_partfunct1.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                window_lngth = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{window_lngth} {partfunct[1,m,j]}\n')
            file.write('\n')


# -------------------------------------------------------------- #

def legendre_transform():

    lambda_est = np.zeros(q_num)
    hx_est = np.zeros(q_num)
    hy_est = np.zeros(q_num)
    hxy_est = np.zeros(q_num)
    vector = np.zeros((num_scales,2))

    for j in range(q_num):
        v = 0
        for m in range(num_scales):
            window_lngth = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_lngth <= window_lngth <= max_lngth:
                vector[v,0] = float(window_lngth)
                vector[v,1] = partfunct[2,m,j]
                v += 1

        vct_lngth = v
        lambda_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            window_lngth = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_lngth <= window_lngth <= max_lngth:
                vector[v,0] = float(window_lngth)
                vector[v,1] = partfunct[0,m,j]
                v += 1

        hx_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            window_lngth = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_lngth <= window_lngth <= max_lngth:
                vector[v,0] = float(window_lngth)
                vector[v,1] = partfunct[1,m,j]
                v += 1

        hy_est[j] = power_fit(vct_lngth,num_scales,vector)

        hxy_est[j] = (hx_est[j] + hy_est[j]) / 2

    lambda_deriv = np.zeros(q_num)
    alpha = np.zeros(q_num)
    f_alpha = np.zeros(q_num)
    tau = np.zeros(q_num)

    lambda_deriv = derivative(q_num,lambda_est)

    for j in range(q_num):
        q_act = q_min + j * q_step
        if q_act == 0.0:
            q_act += q_step / 4.0
        alpha[j] = lambda_est[j] + q_act * lambda_deriv[j]
        f_alpha[j] = q_act * (alpha[j] - lambda_est[j]) + 1.0
        tau[j] = q_act * lambda_est[j] - 1.0

    output2(alpha,f_alpha,tau,lambda_est,hx_est,hy_est,hxy_est)

# -------------------------------------------------------------- #

print("\n     MFDCCA ver. 1.0 converted to python (18.07.2023)\n\n")

#    infiles = []
#    infiles.append(input("Time series 1: "))
#    infiles.append(input("Time series 2: "))
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

infiles = []

# input files
infiles.append("gauss_ts1.dat")
infiles.append("gauss_ts3.dat")
# time series column
column = 1
# minimum scale & number of different scales
min_scale = 10
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

file_num = 2
q_num = int((q_max - q_min) / q_step) + 1
logwin_step = 0.0
num_points = [0] * file_num
if not check_remove: num_zeros = 0



signal = [[] for _ in range(file_num)]
for k in range(file_num):
    with open(infiles[k], 'r') as f:
        lines = f.readlines()
        num_points[k] = len(lines)
        for line in lines:
            items = line.split()
            signal[k].append(float(items[column - 1]))

if num_points[0] != num_points[1]:
    print ("error: unequal lengths",num_points)
    quit()

for k in range(file_num):
    if ts_lngth > 0 and ts_lngth < num_points[k]:
        del signal[k][:ts_lngth]
    else:
        ts_lngth = num_points[k]

error = []
num_err = 0
for k in range(file_num):
    eqval_lngth = 1
    for i in range(1,ts_lngth):
        if signal[k][i] == signal[k][i-1] == 0.0:
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

error.sort()
dummy_list = np.ones(ts_lngth,dtype=int)
for err in range(num_err):
    dummy_list[error[err][0]:error[err][0]+error[err][1]] = 0

error = []
num_err = 0
eqval_lngth = 1
for i in range(2,ts_lngth):
    if dummy_list[i] == dummy_list[i-1] == 0:
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
for k in range(file_num):
    for err in range(num_err,-1,-1):
        del signal[k][error[err][0]:error[err][0]+error[err][1]]

if len(signal[0]) != len(signal[1]):
    print ("unequal lengths",len(signal[0]),len(signal[1]))
    quit()

ts_lngth = len(signal[0])
max_scale = int(ts_lngth / 4)

partfunct = np.zeros(((3,num_scales,q_num)))
partfunct = partition_function(flag)
output1()

#    min_lngth = int(input("Fit - minimum segment length: "))
min_lngth = 10
#    max_lngth = int(input("Fit - maximum segment length: "))
max_lngth = max_scale

legendre_transform()

print("")

