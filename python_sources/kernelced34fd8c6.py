# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

a = True
while a:
    rand_num = np.random.randint(100000,200000)
    print(rand_num)
    if (rand_num%2!=0 and rand_num%3!=0 and rand_num%5!=0):
        a = False
print(rand_num)

val_index= {}
for num in range(rand_num-1):
    num= num+1
    val_mod = []
    for power in range(rand_num-1):
        power = power+1
        value = num**power
        mod = value%rand_num
        val_mod.append(mod)
    val_index[num] = val_mod
    print('Calculations Done',num)
    
    

