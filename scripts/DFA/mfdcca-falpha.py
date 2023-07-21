import numpy as np
import math
from scipy import special
import matplotlib.pyplot as plt

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

def plot_rhoq(rhoq):

    renyi = np.zeros(q_num)
    for j in range(q_num):
        renyi[j] = q_min + j * q_step
    min_xaxis = scales[0]
    max_xaxis = scales[-1]
    min_yaxis = 0.0
    max_yaxis = 1.05

    fig = plt.figure(figsize=(10, 6))
    plt.title('Detrended cross-correlation coefficient $\\rho_q(s)$')
    plt.xlabel('scale $s$',fontsize=15)
    plt.ylabel('$\\rho_q(s)$',fontsize=15)

    for j in range(q_num):
        q_act = q_min + j * q_step
        if q_act == 1.0 or q_act == 2.0 or q_act == 3.0 or q_act == 4.0:
            plt.semilogx(scales,rhoq[j],label=f'$q=${q_act}')

    plt.xlim(float(min_xaxis),float(max_xaxis))
    plt.ylim(min_yaxis,max_yaxis)
    plt.legend(fontsize=15)
    plt.grid(linestyle='--',linewidth=0.5,alpha=0.5)
        
    plt.show()

# -------------------------------------------------------------- #

def output(alpha, f_alpha, tau, lambda_est, hx_est, hy_est, hxy_est, rhoq):

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
                rhoq[j,m] = partfunct[2,j,m] / math.sqrt(partfunct[0,j,m] * partfunct[1,j,m])
                act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {rhoq[j,m]}\n')
            file.write('\n')

    if funct == 'rhoq': plot_rhoq(rhoq)

# -------------------------------------------------------------- #

def plot_deriv(hoelder,falpha,tau,lambda_est,hxy_est,qzero_pos):

    renyi = np.zeros(q_num)
    for j in range(q_num):
        renyi[j] = q_min + j * q_step
    lambda_width = lambda_est[0] - lambda_est[-1]
    hxy_width = hxy_est[0] - hxy_est[-1]
    alpha_width = abs(hoelder[0] - hoelder[-1])
    left_width = hoelder[qzero_pos] - hoelder[-1]
    right_width = hoelder[0] - hoelder[qzero_pos]

    asymmetry = (left_width - right_width) / (left_width + right_width)

    if funct == 'hurstq':
        min_xaxis = renyi[0]
        max_xaxis = renyi[-1]
        min_yaxis = lambda_est[-1]
        if hxy_est[-1] < lambda_est[-1]: min_yaxis = hxy_est[-1]
        max_yaxis = lambda_est[0]
        if hxy_est[0] > lambda_est[0]: min_yaxis = hxy_est[0]
    if funct == 'tauq':
        min_xaxis = renyi[0]
        max_xaxis = renyi[-1]
        min_yaxis = tau[0]
        max_yaxis = tau[-1]
    if funct == 'falpha':
        min_xaxis = hoelder[0]
        if min_xaxis > 0: min_xaxis = 0
        max_xaxis = hoelder[-1]
        if max_xaxis < 1: max_xaxis = 1
        min_yaxis = 0.0
        max_yaxis = 1.2

    fig = plt.figure(figsize=(10, 6))
    if funct == 'hurstq':
        plt.title('Generalized Hurst exponents: bivariate $\lambda(q)$ & average univariate $h_{xy}(q)$')
        plt.xlabel('$q$',fontsize=15)
        plt.ylabel('$\lambda(q)$, $h_{xy}(q)$',fontsize=15)
    if funct == 'tauq':
        plt.title('Multifractal spectrum $\\tau(q)$')
        plt.xlabel('$q$',fontsize=15)
        plt.ylabel('$\\tau(q)$',fontsize=15)
    if funct == 'falpha':
        plt.title('Singularity spectrum $f(\\alpha)$')
        plt.xlabel('$\\alpha$',fontsize=15)
        plt.ylabel('$f(\\alpha)$',fontsize=15)

    if funct == 'hurstq':
        plt.plot(renyi,lambda_est,label='$\lambda(q)$')
        plt.plot(renyi,hxy_est,label='$h_{xy}(q)$')
    if funct == 'tauq': plt.plot(renyi,tau)
    if funct == 'falpha': plt.plot(hoelder,falpha)

    plt.xlim(float(min_xaxis),float(max_xaxis))
    plt.ylim(min_yaxis,max_yaxis)
    plt.legend(fontsize=15)
    if funct == 'hurstq':
        plt.xticks(np.arange(q_min,q_max))
        plt.text(0.2,0.15, f'$\Delta lambda$={lambda_width:.2f}', fontsize=15)
        plt.text(0.2,0.15, f'$\Delta h_xy$={hxy_width:.2f}', fontsize=15)
    if funct == 'tauq':
        plt.xticks(np.arange(q_min,q_max))
    if funct == 'falpha':
        plt.xticks(np.arange(0,1,0.1))
        alpha_subscript = '\\alpha'
        plt.text(0.75,0.25, f'$\Delta {alpha_subscript}$={alpha_width:.2f}', fontsize=15)
        plt.text(0.75,0.15, f'$A_{alpha_subscript}$={asymmetry:.2f}', fontsize=15)
    plt.grid(linestyle='--',linewidth=0.5,alpha=0.5)
        
    plt.show()

# -------------------------------------------------------------- #

def legendre_transform():

    lambda_est = np.zeros(q_num)
    hx_est = np.zeros(q_num)
    hy_est = np.zeros(q_num)
    hxy_est = np.zeros(q_num)
    rhoq = np.zeros((q_num,num_scales))
    vector = np.zeros((num_scales,2))

    for j in range(q_num):
        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[2,j,m]
                v += 1

        vct_lngth = v
        lambda_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[0,j,m]
                v += 1

        hx_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = partfunct[1,j,m]
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
            qzero_pos = j
        alpha[j] = lambda_est[j] + q_act * lambda_deriv[j]
        f_alpha[j] = q_act * (alpha[j] - lambda_est[j]) + 1.0
        tau[j] = q_act * lambda_est[j] - 1.0

    if funct != 'rhoq':
        plot_deriv (alpha,f_alpha,tau,lambda_est,hxy_est,qzero_pos)

    output(alpha,f_alpha,tau,lambda_est,hx_est,hy_est,hxy_est,rhoq)

# -------------------------------------------------------------- #

def onclick(event):

    global click, min_range, max_range
    if click > 1: click = 0
    if click == 0: min_range = event.xdata
    if click == 1:
        max_range = event.xdata
    click += 1

# -------------------------------------------------------------- #

def plot_partfunct():

    fig = plt.figure(figsize=(10, 6))
    plt.title('Bivariate partition function $F_q(s)$')
    plt.xlabel('scale $s$ [pts]')
    plt.ylabel('$F_q(s)$')
    for j in range(q_num):
        plt.loglog(scales,partfunct[2,j])
    plt.xlim(float(min_scale),float(max_scale))
    plt.ylim(partfunct[2,0,0],partfunct[2,q_num-1,num_scales-1])
    plt.legend()
    plt.grid()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

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
infiles.append("arfima_d0.1_T65k_partfunct0.dat")
infiles.append("arfima_d0.4_T65k_partfunct1.dat")
infiles.append("arfima_d0.1_T65k_arfima_d0.4_T65k_partfunct.dat")
# number of scales
num_scales = 50
# range and step of the Renyi parameter q
q_min = -4
q_max = 4
q_step = 0.2
# function to be displayed ('falpha' / 'tauq' / 'hurstq')
funct = 'rhoq'

# ====================================================================================== #

file_num = 3
q_num = int((q_max - q_min) / q_step) + 1

scales = []
partfunct = np.zeros((file_num,q_num,num_scales))
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
                partfunct[k,j,m] = items[2]
                m += 1
                if k == 0 and j == 0:
                    scales.append(float(items[1]))

min_scale = scales[0]
max_scale = scales[num_scales - 1]
logwin_step = (np.log(float(max_scale)) - np.log(float(min_scale))) / (num_scales - 1)

min_range = max_range = 0
click = 0

plot_partfunct()

legendre_transform()

print("")

