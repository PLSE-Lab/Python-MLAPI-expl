#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
 
    
  print('hello')


# # H1
# ## H2
# ### H3
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# Data - Array
x=np.array([2,4,3,5,6])
y=np.array([10,5,9,4,3])

E_x=np.mean(x)
E_y=np.mean(y)
cov_xy=np.mean(x*y)- E_x*E_y
y_0= E_y- cov_xy / np.var(x)* E_x
m= cov_xy/np.var(x)
y_pred=m*x+y_0
print("E[(y_pred-y_actual)^2]=", np.mean(np.square(y_pred-y)))
    
# Graph
plt.scatter(x,y, color='Black', s=100)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.plot(x, y_pred, color='Red')
plt.show()
   
    

