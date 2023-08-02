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
        hurst_deriv[j + 1] = 2.0 * coeff[0] * q_act + coeff[1]
        if j == 0:
            hurst_deriv[j] = 2.0 * coeff[0] * q_min + coeff[1]
        if j == q_num - 3:
            hurst_deriv[j + 2] = 2.0 * coeff[0] * (q_act + q_step) + coeff[1]

    return hurst_deriv

# -------------------------------------------------------------- #

def output(alpha, f_alpha, tau, hurst_est):

    outfile = infile[:-14] + '_tau.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            file.write(f"{q_act} {tau[j]}\n")
    
    outfile = infile[:-14] + '_falpha.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            file.write(f"{alpha[j]} {f_alpha[j]}\n")

    outfile = infile[:-14] + '_hurst.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + float(j) * q_step
            file.write(f"{q_act} {hurst_est[j]}\n")

# -------------------------------------------------------------- #

def plot_deriv(hoelder,falpha,tau,hurst_est,qzero_pos):

    global funct, q_min, q_max, q_step

    renyi = np.zeros(q_num)
    for j in range(q_num):
        renyi[j] = q_min + j * q_step
    if funct == 'hurstq':
        min_xaxis = renyi[0]
        max_xaxis = renyi[-1]
        min_yaxis = hurst_est[-1] * 0.8
        max_yaxis = hurst_est[0] * 1.2
        hurst_width = hurst_est[0] - hurst_est[-1]
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
        if -1 * q_min == q_max:
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
        plt.title('Generalized Hurst exponent $h(q)$')
        plt.xlabel('$q$',fontsize=15)
        plt.ylabel('$h(q)$',fontsize=15)
        plt.plot(renyi,hurst_est)
        plt.xticks(np.arange(q_min,q_max))
        plt.text(0.2,0.2,f'$\Delta h$={hurst_width:.2f}',transform=plt.gcf().transFigure,fontsize=15)
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
        plt.text(0.75,0.25, f'$\Delta {alpha_subscript}$={alpha_width:.2f}', transform=plt.gcf().transFigure, fontsize=15)
        if asymmetry != 'undef':
            plt.text(0.75,0.2, f'$A_{alpha_subscript}$={asymmetry:.2f}', transform=plt.gcf().transFigure, fontsize=15)
        else:
            plt.text(0.75,0.2,f'$A_{alpha_subscript}$={asymmetry}', transform=plt.gcf().transFigure, fontsize=15)

    fig.canvas.mpl_connect('key_press_event',close_on_key)

    plt.show()

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
                vector[v,1] = fluctfunct[j,m]
                v += 1

        vct_lngth = v - 1
        hurst_est[j] = power_fit(vct_lngth,num_scales,vector)

    hurst_deriv = np.zeros(q_num)
    alpha = np.zeros(q_num)
    f_alpha = np.zeros(q_num)
    tau = np.zeros(q_num)

    hurst_deriv = derivative(q_num,hurst_est)

    qzero_pos = 0
    for j in range(q_num):
        q_act = q_min + j * q_step
        if q_act == 0.0: qzero_pos = j
        alpha[j] = hurst_est[j] + q_act * hurst_deriv[j]
        f_alpha[j] = q_act * (alpha[j] - hurst_est[j]) + 1.0
        tau[j] = q_act * hurst_est[j] - 1.0

    plot_deriv (alpha,f_alpha,tau,hurst_est,qzero_pos)

    output(alpha,f_alpha,tau,hurst_est)

# -------------------------------------------------------------- #

def onclick(event):

    global click, min_range, max_range, can_close
    if click > 1:
        click = 0
        can_close = False
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
    plt.title('Fluctuation function $F_q(s)$')
    plt.xlabel('scale $s$ [pts]')
    plt.ylabel('$F_q(s)$')
    for j in range(q_num):
        plt.loglog(scales, fluctfunct[j])
    plt.xlim(float(min_scale),float(max_scale))
    plt.ylim(fluctfunct[0,0],fluctfunct[q_num-1,num_scales-1])
    plt.grid()
    
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    fig.canvas.mpl_connect('key_press_event',close_on_key)

    plt.show()

# -------------------------------------------------------------- #

def fluctuation_function():

    global fluctfunct, logwin_step, scales
    profile = np.zeros(ts_lngth)
    average = np.mean(signal)
    profile[0] = signal[0] - average
    for i in range(1, ts_lngth):
        profile[i] = profile[i - 1] + signal[i] - average
    logwin_step = (np.log(max_scale) - np.log(min_scale)) / (num_scales - 1)
    for m in range(num_scales):
        act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
        scales.append(act_scale)
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
                fluctfunct[j,m] = np.mean([math.log(entry) for entry in segment_var])
                fluctfunct[j,m] = math.exp(fluctfunct[j,m] / 2)
            else:
                fluctfunct[j,m] = np.mean(segment_var**(q_act / 2))
                fluctfunct[j,m] = fluctfunct[j,m]**(1.0 / q_act)

    return fluctfunct

# -------------------------------------------------------------- #

def output1():
    outfile = infile[:-4] + '_fluctfunct.dat'
    with open(outfile, 'w') as file:
        for j in range(q_num):
            q_act = q_min + j * q_step
            for m in range(num_scales):
                act_scale = round(np.exp(np.log(min_scale) + m * logwin_step))
                file.write(f'{q_act} {act_scale} {fluctfunct[j,m]}\n')
            file.write('\n')

# -------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Multifractal Detrended Fluctuation Analysis of time series")
parser.add_argument("--col", help = "data column (default: 1)")
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
parser.add_argument('filename', help="name of the data file")

args = parser.parse_args()

if len(sys.argv) == 1: print ("\nNo data file name provided.\n")

# ======================================================================= #

#    INPUT PARAMETERS (if not specified on the command line)

#   input file
if args.filename is None:
    infile = "type_file_name_here.dat"
else:
    infile = args.filename
#   data column
if args.col is None:
    column = 1
else:
    column = int(args.col)
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

# ======================================================================= #

q_num = int((q_max - q_min) / q_step) + 1
logwin_step = 0.0
if not check_remove: num_zeros = 0

signal = []
with open(infile, 'r') as f:
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
scales = []

fluctfunct = np.zeros((q_num,num_scales))
fluctfunct = fluctuation_function()
output1()

logwin_step = (np.log(float(max_scale)) - np.log(float(min_scale))) / (num_scales - 1)
min_range = max_range = 0
click = 0
can_close = False

plot_fluctfunct()

legendre_transform()

print("")

