# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
x=15
y=5
print('x+y=',x+y)
print('x-y=',x-y)
print('x*y=',x*y)
print('x/y=',x/y)
lista=[1000,2000,3000,4000,5000,6000]
print('lista sa elementima',lista)
print('1000')
print('4000')
lista=[1000,2000,3000,4000,5000,6000,7000]
print('ubacen u listu element 7000',lista)
lista=[1000,2000,3000,4000,6000,7000]
print('izbacen iz liste element 5000',lista)
lista=[27,2000,3000,4000,6000,7000]
print('promenjena vrednost prvog elementa na 27',lista)
lista=[2000,4000,7000]
print('izbrisan je element sa 3 pozicije',lista)






