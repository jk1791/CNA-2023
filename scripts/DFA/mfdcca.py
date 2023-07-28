import numpy as np
import math
from scipy import special
import matplotlib.pyplot as plt
import argparse
import sys

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
    max_yaxis = 1.1
    fig = plt.figure(figsize=(10,6))
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

    fig.canvas.mpl_connect('key_press_event',close_on_key)
   
    plt.show()

# -------------------------------------------------------------- #

def output(alpha,f_alpha,tau,lambda_est,hx_est,hy_est,hxy_est,rhoq):

    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_tauq.dat'
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

    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_rhoq.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                rhoq[j,m] = fluctfunct[2,j,m] / math.sqrt(fluctfunct[0,j,m] * fluctfunct[1,j,m])
                act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {rhoq[j,m]}\n')
            file.write('\n')

    if funct == 'rhoq': plot_rhoq(rhoq)

# -------------------------------------------------------------- #

def plot_deriv(hoelder,falpha,tau,lambda_est,hxy_est,qzero_pos):

    global funct, q_min, q_max, q_step
    
    renyi = np.zeros(q_num)
    for j in range(q_num):
        renyi[j] = q_min + j * q_step
    if funct == 'hurstq':
        min_xaxis = renyi[0]
        max_xaxis = renyi[-1]
        min_yaxis = lambda_est[-1] * 0.8
        if hxy_est[-1] < lambda_est[-1]: min_yaxis = hxy_est[-1] * 0.8
        max_yaxis = lambda_est[0] * 1.2
        if hxy_est[0] > lambda_est[0]: min_yaxis = hxy_est[0] * 1.2
        lambda_width = lambda_est[0] - lambda_est[-1]
        hxy_width = hxy_est[0] - hxy_est[-1]
    if funct == 'tauq':
        min_xaxis = renyi[0]
        max_xaxis = renyi[-1]
        min_yaxis = tau[0]
        max_yaxis = tau[-1]
    if funct == 'falpha':
        min_xaxis = hoelder[-1]
        if min_xaxis > 0: min_xaxis = 0
        max_xaxis = hoelder[0]
        if max_xaxis < 1: max_xaxis = 1
        min_yaxis = 0.0
        max_yaxis = 1.2
        alpha_width = abs(hoelder[0] - hoelder[-1])
        if min_scale <= 0.0:
            left_width = hoelder[qzero_pos] - hoelder[-1]
            right_width = hoelder[0] - hoelder[qzero_pos]
            asymmetry = (left_width - right_width) / (left_width + right_width)
        else:
            asymmetry = 'undef'

    fig = plt.figure(figsize=(10,6))
    plt.xlim(float(min_xaxis),float(max_xaxis))
    plt.ylim(min_yaxis,max_yaxis)
    plt.grid(linestyle='--',linewidth=0.5,alpha=0.5)
    if funct == 'hurstq':
        plt.title('Generalized Hurst exponents: bivariate $\lambda(q)$ & average univariate $h_{xy}(q)$')
        plt.xlabel('$q$',fontsize=15)
        plt.ylabel('$\lambda(q)$, $h_{xy}(q)$',fontsize=15)
        plt.plot(renyi,lambda_est,label='$\lambda(q)$')
        plt.plot(renyi,hxy_est,label='$h_{xy}(q)$')
        plt.xticks(np.arange(q_min,q_max))
        plt.text(0.2,0.15,f'$\Delta lambda$={lambda_width:.2f}',transform=plt.gcf().transFigure,fontsize=15)
        plt.text(0.2,0.15, f'$\Delta h_xy$={hxy_width:.2f}',transform=plt.gcf().transFigure,fontsize=15)
        plt.legend(fontsize=15)
    if funct == 'tauq':
        plt.title('Multifractal spectrum $\\tau(q)$')
        plt.xlabel('$q$',fontsize=15)
        plt.ylabel('$\\tau(q)$',fontsize=15)
        plt.plot(renyi,tau)
        plt.xticks(np.arange(q_min,q_max))
    if funct == 'falpha':
        plt.title('Singularity spectrum $f(\\alpha)$')
        plt.xlabel('$\\alpha$',fontsize=15)
        plt.ylabel('$f(\\alpha)$',fontsize=15)
        plt.plot(hoelder,falpha)
        plt.xticks(np.arange(0,max_xaxis,0.1))
        alpha_subscript = '\\alpha'
        plt.text(0.75,0.25,f'$\Delta {alpha_subscript}$={alpha_width:.2f}', transform=plt.gcf().transFigure, fontsize=15)
        if asymmetry != 'undef':
            plt.text(0.75,0.2,f'$A_{alpha_subscript}$={asymmetry:.2f}', transform=plt.gcf().transFigure, fontsize=15)
        else:
            plt.text(0.75,0.15,f'$A_{alpha_subscript}$={asymmetry}', transform=plt.gcf().transFigure, fontsize=15)

    fig.canvas.mpl_connect('key_press_event',close_on_key)

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
                vector[v,1] = fluctfunct[2,j,m]
                v += 1

        vct_lngth = v
        lambda_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = fluctfunct[0,j,m]
                v += 1

        hx_est[j] = power_fit(vct_lngth,num_scales,vector)

        v = 0
        for m in range(num_scales):
            act_scale = round(np.exp(np.log(float(min_scale)) + m * logwin_step))
            if min_range <= act_scale <= max_range:
                vector[v,0] = float(act_scale)
                vector[v,1] = fluctfunct[1,j,m]
                v += 1

        hy_est[j] = power_fit(vct_lngth,num_scales,vector)

        hxy_est[j] = (hx_est[j] + hy_est[j]) / 2

    lambda_deriv = np.zeros(q_num)
    alpha = np.zeros(q_num)
    f_alpha = np.zeros(q_num)
    tau = np.zeros(q_num)

    lambda_deriv = derivative(q_num,lambda_est)

    qzero_pos = 0
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

    global click, min_range, max_range, can_close
    if click > 1: click = 0
    if click == 0:
        if event.xdata is not None:
            min_range = event.xdata
        else:
            min_range = min_scale
    if click == 1:
        if event.xdata is not None:
            max_range = event.xdata
        else:
            max_range = max_scale
    click += 1
    if click > 1:
        can_close = True
        if min_range > max_range:
            temp = min_range
            min_range = max_range
            max_range = temp
        if min_range == max_range: max_range += 1

# -------------------------------------------------------------- #

def close_on_key(event):

    global can_close
    if can_close:
        plt.close("all")

# -------------------------------------------------------------- #

def plot_fluctfunct():

    fig = plt.figure(figsize=(10,6))
    plt.title('Bivariate fluctuation function $F_q(s)$')
    plt.xlabel('scale $s$ [pts]')
    plt.ylabel('$F_q(s)$')
    for j in range(q_num):
        plt.loglog(scales,fluctfunct[2,j])
    plt.xlim(float(min_scale),float(max_scale))
    if fluctfunct[2,0,0] > 0.0:
        min_yaxis = fluctfunct[2,0,0] / 1.5
    else:
        for j in range(q_num):
            if fluctfunct[2,j,0] > 0: break
        min_yaxis = fluctfunct[2,j,0] / 1.5
    max_yaxis = fluctfunct[2,q_num-1,num_scales-1] * 1.5
    plt.ylim(min_yaxis,max_yaxis)
    plt.grid()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event',close_on_key)

    plt.show()

# -------------------------------------------------------------- #

def fluctuation_function ():

    global fluctfunct, logwin_step, scales

    average = [0] * 2
    profile = np.zeros((2,ts_lngth))
    for k in range(2):
        average[k] = np.mean(signal[k])
        profile[k,0] = signal[k][0] - average[k]
        for i in range(1, ts_lngth):
            profile[k,i] = profile[k,i - 1] + signal[k][i] - average[k]
    logwin_step = (np.log(max_scale) - np.log(min_scale)) / (num_scales - 1)
    for m in range(num_scales):
        act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
        scales.append(act_scale)
        if m == num_scales - 1:
            act_scale = max_scale

        win_num = ts_lngth // act_scale
        segment_var = np.zeros((3,2 * win_num))
        window = [[] for _ in range(2)]
        polynomial = [[] for _ in range(2)]
        for w in range(win_num):
            win_pos = w * act_scale
            for k in range(2):
                window[k] = profile[k,win_pos:win_pos + act_scale]
                polynomial[k] = np.polyfit(np.arange(1,act_scale + 1),window[k],polyorder)
                segment_var[k,w] = np.mean((window[k] - np.polyval(polynomial[k],np.arange(1,act_scale + 1)))**2)
            segment_var[2,w] = np.mean((window[0] - np.polyval(polynomial[0],np.arange(1,act_scale + 1))) * (window[1] - np.polyval(polynomial[1],np.arange(1,act_scale + 1))))

        for w in range(win_num):
            win_pos = (ts_lngth % act_scale) + w * act_scale
            for k in range(2):
                window[k] = profile[k,win_pos:win_pos + act_scale]
                polynomial[k] = np.polyfit(np.arange(1,act_scale + 1),window[k],polyorder)
                segment_var[k,win_num + w] = np.mean((window[k] - np.polyval(polynomial[k],np.arange(1,act_scale + 1)))**2)
            segment_var[2,win_num + w] = np.mean((window[0] - np.polyval(polynomial[0],np.arange(1,act_scale + 1))) * (window[1] - np.polyval(polynomial[1],np.arange(1,act_scale + 1))))

        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0
            fluctfunct[2,j,m] = np.mean(np.sign(segment_var[2]) * (abs(segment_var[2]))**(q_act / 2))
            fluctfunct[2,j,m] = np.sign(fluctfunct[2,j,m]) * abs(fluctfunct[2,j,m])**(1.0 / q_act)

            fluctfunct[0,j,m] = np.mean(segment_var[0]**(q_act / 2))
            fluctfunct[0,j,m] = fluctfunct[0,j,m]**(1.0 / q_act)
            fluctfunct[1,j,m] = np.mean(segment_var[1]**(q_act / 2))
            fluctfunct[1,j,m] = fluctfunct[1,j,m]**(1.0 / q_act)

    return fluctfunct

# -------------------------------------------------------------- #

def output1():
    outfile = infiles[0][:-4] + '_' + infiles[1][:-4] + '_fluctfunct.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {fluctfunct[2,j,m]}\n')
            file.write('\n')

    outfile = infiles[0][:-4] + '_fluctfunct0.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {fluctfunct[0,j,m]}\n')
            file.write('\n')

    outfile = infiles[1][:-4] + '_fluctfunct1.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            if q_act == 0:
                q_act += q_step / 4.0

            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {fluctfunct[1,j,m]}\n')
            file.write('\n')

# -------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Multifractal Detrended Cross-correlation Analysis of time series")
parser.add_argument("--col1", help = "data column 1 (default: 1)")
parser.add_argument("--col2", help = "data column 2 (default: 1)")
parser.add_argument("--minsc", help = "minimum scale (default: 10)")
parser.add_argument("--numsc", help = "number of scales (default: 50)")
parser.add_argument("--minq", help = "minimum q (default: -4.0)")
parser.add_argument("--maxq", help = "maximum q (default: 4.0)")
parser.add_argument("--stepq", help = "increment of q (default: 0.5)")
parser.add_argument("--polyord", help = "detrending polynomial order (default: 2)")
parser.add_argument("--length", help = "number of data points to be considered (default: all)")
parser.add_argument("--remove", help = "remove sequences of zeros [no/yes] (default: yes)")
parser.add_argument("--numz", help = "maximum allowed sequence of zeros (default: 10)")
parser.add_argument("--funct", help = "function to be plotted [hurstq/tauq/falpha] (default: falpha)")
parser.add_argument('filename1', help="name of the 1st data file")
parser.add_argument('filename2', help="name of the 2nd data file")

args = parser.parse_args()

if len(sys.argv) < 3:
    print ("\nData file name(s) not provided. Exiting...\n")
    quit()

# ====================================================================================== #

#    INPUT PARAMETERS:

infiles = []
column = [1,1]
#   input file
if args.filename1 is None:
    infiles.append("type_1st_file_name_here.dat")
else:
    infiles.append(args.filename1)
if args.filename2 is None:
    infiles.append("type_2nd_file_name_here.dat")
else:
    infiles.append(args.filename2)
#   data column in file 1
if args.col1 is not None: column[0] = int(args.col1)
#   data column in file 2
if args.col2 is not None: column[1] = int(args.col2)
#   minimum scale & number of different scales
if args.minsc is None:
    min_scale = 20
else:
    min_scale = int(args.minsc)
if args.numsc is None:
    num_scales = 50
else:
    num_scales = int(args.numsc)
#   range of the Renyi parameter q
if args.minq is None:
    q_min = -4
else:
    q_min = float(args.minq)
if args.numsc is None:
    q_max = 4
else:
    q_max = float(args.maxq)
if args.stepq is None:
    q_step = 0.5
else:
    q_step = float(args.stepq)
# detrending polynomial order
if args.polyord is None:
    polyorder = 2
else:
    polyorder = int(args.polyord)
# number of data points to be considered (0 - all)
if args.length is None:
    ts_lngth = 0
else:
    ts_lngth = int(args.length)
# remove sequences of zeros?  (True / False)
if args.remove is None:
    check_remove = True
else:
    check_remove = bool(args.remove)
# if so, how long is the maximum allowed sequence of zeros
if args.numz is None:
    num_zeros = 10
else:
    num_zeros = int(args.numz)
# function to be displayed ('falpha' / 'tauq' / 'hurstq')
if args.funct is None:
    funct = 'falpha'
else:
    funct = args.funct

# ====================================================================================== #

q_num = int((q_max - q_min) / q_step) + 1
logwin_step = 0.0
num_points = [0] * 2
if not check_remove: num_zeros = 0

signal = [[] for _ in range(2)]
items = np.zeros(3)

for k in range(2):
    with open(infiles[k], 'r') as f:
        lines = f.readlines()
        num_points[k] = len(lines)
        for line in lines:
            items = line.split()
            signal[k].append(float(items[column[k] - 1]))

if num_points[0] != num_points[1]:
    print ("error: unequal lengths",num_points)
    quit()

for k in range(2):
    if ts_lngth > 0 and ts_lngth < num_points[k]:
        del signal[k][:ts_lngth]
    else:
        ts_lngth = num_points[k]

error = []
num_err = 0
for k in range(2):
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
for k in range(2):
    for err in range(num_err,-1,-1):
        del signal[k][error[err][0]:error[err][0]+error[err][1]]

if len(signal[0]) != len(signal[1]):
    print ("unequal lengths",len(signal[0]),len(signal[1]))
    quit()

ts_lngth = len(signal[0])
max_scale = int(ts_lngth / 4)
scales = []

fluctfunct = np.zeros(((3,q_num,num_scales)))
fluctfunct = fluctuation_function()
output1()

logwin_step = (np.log(float(max_scale)) - np.log(float(min_scale))) / (num_scales - 1)
min_range = max_range = 0
click = 0
can_close = False

plot_fluctfunct()
legendre_transform()

print("")

