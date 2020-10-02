#!/usr/bin/env python
#Grupo Bimbo Contest
#Creates statistics for each possible combination of columns vs. Demanda
import numpy as np
import glob
import pandas as pd
import sys
import math
import os
from datetime import datetime
import gc

cols=['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Demanda_uni_equil']
#these are the ones we want to explore in their relationship to Demand:
acols=['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID']
dbg_start_time=datetime.now()
def dbg(*string):
    #puts out debug message, with time from start prepended
    #all arguments are converted to string and printed.  E.g. dbg('hello', array)
    delta=datetime.now()-dbg_start_time
    hours=delta.seconds // 3600
    mins=(delta.seconds - 3600*hours) // 60
    secs=(delta.seconds - 3600*hours - 60*mins)
    hundredths=(delta.microseconds)// 10000
    s=' '.join([str(x) for x in string])
    sys.stdout.write("{:02d}:{:02d}:{:02d}.{:02d} {}\n".format(hours, mins, secs, hundredths, s))
    sys.stdout.flush()

types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,
     'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16, 'Demanda_uni_equil':np.float32}

dbg('starting')
#Use full=False to read partial file for faster debugging
full=True
if full:
    train=pd.read_csv('../input/train.csv', usecols=cols, dtype=types)
else:
    print('reading partial train file')
    train=pd.read_csv('../input/train.csv', usecols=cols, nrows=100000, dtype=types)
    
dbg('read train, len=%d' % (len(train)))    

train['logDemand']=np.log1p(train['Demanda_uni_equil']).astype(np.float16)
del train['Demanda_uni_equil']

#create the statistics columns.  std will be NaN for those with size of 1

def get_col_list(n):
    global acols
    cols=list()
    num=len(acols)
    #want to have bit 0 be the last item in the list, so num-i
    for i in range(num):
        if 2**i & n:
            cols.append(acols[num-i-1])
    return cols
import psutil

def mem_usage(s=''):
    #compute memory usage by process and print out, prepend with arg s
    process = psutil.Process(os.getpid())
    m=process.memory_info().rss
    if m > 1024**2:
        dbg('({}) Memory used:{:.1f}M'.format(s, m/1024.0**2))
    else:
        dbg('({}) Memory used:{:,}B'.format(s, m))

num=len(acols)
print("COLUMNS for results: iter, count, mean, std-deviation")
print("mean and std-deviation are on log1p.  mean is expm1 of mean()")
mem_usage()
for iter in range(2 ** num):
    # compute cols from the binary representation of x: 1=include 0=do not
    #set cols for this iteration
    dbg('==== iter {}'.format(iter))
    cols=get_col_list(iter)
    print( iter, cols)

    if iter==0:
        sz=len(train)
        mn=train['logDemand'].mean()
        std=train['logDemand'].std()
    else:
        gp=train.groupby(cols)
        #x=gp['logDemand'].agg([np.size, np.mean, np.std])
        x=gp['logDemand'].agg([np.size, np.std])
        #create a series of the mean values of the 3 columns:
        s=x.mean()
        sz=s['size']
        #mn=s['mean']
        std=s['std']
        del gp, x
    gc.collect()
    mem_usage()
    #print("results   {:6d} {:12.2f}  {:12.4f}  {:12.4f}".format(iter, sz, np.expm1(mn), std))
    print("results   {:6d} {:12.2f}  {:12.4f}".format(iter, sz, std))
    
    


