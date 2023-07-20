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

def output(alpha, f_alpha, tau, lambda_est, hx_est, hy_est, hxy_est):

    outfile = infiles[2][:-14] + '_tauq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {tau[j]}\n")
    
    outfile = infiles[2][:-14] + '_falpha.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{alpha[j]} {f_alpha[j]}\n")

    outfile = infiles[2][:-14] + '_lambdaq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {lambda_est[j]}\n")

    outfile = infiles[2][:-14] + '_hxyq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            if q_act == 0.0:
                q_act = q_act + q_step / 4.0
            file.write(f"{q_act} {hxy_est[j]}\n")

    outfile = infiles[2][:-14] + '_rhoq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                rhoq = partfunct[2,m,j] / math.sqrt(partfunct[0,m,j] * partfunct[1,m,j])
                act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {rhoq}\n')
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
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[2,m,j]
                v += 1

        vct_lngth = v
        lambda_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[0,m,j]
                v += 1

        hx_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
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

    output(alpha,f_alpha,tau,lambda_est,hx_est,hy_est,hxy_est)

# -------------------------------------------------------------- #

print("\n     h(q), tau(q), f(alpha), rho(q) from partition functions (20.07.2023)\n\n")

infiles = []
#    infiles.append(input("univariate partition function no. 2: "))
#    infiles.append(input("univariate partition function no. 1: "))
#    infiles.append(input("bivariate partition function: "))
#    min_range = int(input('minimum scale : '))
#    max_range = int(input('maximum scale : '))
#    q_min = float(input('minimum q : '))
#    q_max = float(input('maximum q : '))
#    q_step = float(input('q step : '))

# ====================================================================================== #

#    INPUT PARAMETERS:

# partition function files
infiles.append("gauss_ts1_partfunct0.dat")
infiles.append("gauss_ts3_partfunct1.dat")
infiles.append("gauss_ts1_gauss_ts3_partfunct.dat")
# number of scales
num_scales = 50
# fitting scale range
min_range = 10
max_range = 1000
# range and step of the Renyi parameter q
q_min = -4
q_max = 4
q_step = 0.2

# ====================================================================================== #

file_num = 3
q_num = int((q_max - q_min) / q_step) + 1

scales = []
partfunct = np.zeros((file_num,num_scales,q_num))
items = np.zeros(3)

for k in range(file_num):
    with open(infiles[k], 'r') as f:
        lines = f.readlines()
        j = m = 0
        for line in lines:
            if line == '\n':
                j += 1
                m = 0
            else:
                items = line.split()
                partfunct[k,m,j] = items[2]
                m += 1
                if k == 0 and j == 0:
                    scales.append(items[1])

min_scale = scales[0]
max_scale = scales[num_scales - 1]
logwin_step = (np.log(float(max_scale)) - np.log(float(min_scale))) / (num_scales - 1)

legendre_transform()

print("")

