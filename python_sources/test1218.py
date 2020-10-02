# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import platform
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print('=======  wutianbo  =======')
a = platform.system()
b = platform.architecture()
c = platform.version()
print('system: ',a,' ',b,' ',c)
print(' ')


def memory_stat(): 
    mem = {} 
    f = open("/proc/meminfo") 
    lines = f.readlines() 
    f.close() 
    xx = 1
    for line in lines: 
        print('/proc/meminfo[%s]: %s'%(xx,line))
        if len(line) < 2: continue 
        name = line.split(':')[0] 
        var = line.split(':')[1].split()[0] 
        mem[name] = float(var) * 1024.0 
        xx +=1
    mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached'] 
    return mem 
    
mem = memory_stat()

# Any results you write to the current directory are saved as output.