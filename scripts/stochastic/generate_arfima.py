import _arfima

timeseries = _arfima.arfima([],0.2,[],2**16,warmup=2**16)
outfile = 'arfima_d0.2_T65k.dat'
with open(outfile, 'w') as file:
   for j in range(len(timeseries)):
       file.write(f"{timeseries[j]}\n")
