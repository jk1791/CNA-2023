import numpy as np
import random

def decimal2binary(n,length):
    return format(n,'0'+str(length)+'b')

# -------------------------------------------------------------- #

#    generate a binomial cascade (deterministic) (21.07.2023) 

#    INPUT PARAMETERS

multiplier = [0,0]

# number of cascade levels
num_levels = 17
# multiplier
multiplier[0] = random.random()
# output file
outfile = f"binomial_cascade_n{num_levels}.dat"

# ====================================================================================== #

multiplier[1] = 1 - multiplier[0]
ts_lngth = 2**num_levels
cascade = np.zeros(ts_lngth)

for i in range(2**num_levels):
    binstring = decimal2binary(i,num_levels)
    value = 1.0
    for k in range(num_levels):
        if binstring[k] == '1':
            factor = multiplier[1]
        else:
            factor = multiplier[0]
        value = value * factor
    cascade[i] = value

with open(outfile, 'w') as file:
    for i in range(ts_lngth):
        file.write(f"{cascade[i]}\n")

