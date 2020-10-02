#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import sys 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


inventory=pd.read_csv('../input/inventory.dat',sep='|',header=None)
item=pd.read_csv('../input/item.dat',sep='|',header=None)
web_sales=pd.read_csv('../input/web_sales.dat',sep='|',header=None)


# In[ ]:


# sys.getsizeof(inventory,size_type=SIZE_UNIT.BYTES)


# Lossy counting approx algorithm

# In[ ]:


epsilon=0.1
w=1/epsilon
D={}
O={}
D1={}
j=0
i=0
delta=0.0001
DELTA=0
beta=1
attr1=inventory[1]
attr2=inventory[2]
for e in attr1:
    if e in D:
        D[e]+=1
        D1[e]=delta
        O[e]+=1
    else:
        D[e]=1
        D1[e]=i
        O[e]=1
    
    j+=1
    
    if j==w:
        for a in range(len(D)):
            if a in D:
                if D[a]+D1[a]<=i:
                    D.pop(a)
                    D1.pop(a)
                
        j=0
        i+=1
        
    


# In[ ]:


un=[]
un1=[]
for i in D:
    un.append(i)
for j in O:
    un1.append(j)
    
un=pd.DataFrame(un)
un1=pd.DataFrame(un1)
print(un.shape,un1.shape)
# np.shape(attr2)


# In[ ]:


# count=0
# for i in attr2:
#     if i==0:
#         count+=1
# count


# In[ ]:


# count1=0
# list=[]
# # D=D.sort
# for i in D:
#     if i==10000:
#         print("yes")
# #         count1+=D[i]
# # count1


# In[ ]:


np.shape(attr1)


# In[ ]:


key=[]
value=[]
for i in D:
    key.append(i)
    value.append(D[i])
#     print(i,D[i])
key=pd.DataFrame(key)
value=pd.DataFrame(value)
key['value']=value


# In[ ]:


O={}
for i in attr1:
    if i in O:
        O[i]+=1
    else:
        O[i]=1
    
#     print(i)
    


# In[ ]:


key1=[]
value1=[]
for i in O:
    key1.append(i)
    value1.append(O[i])
#     print(i,D[i])
key1=pd.DataFrame(key1)
value1=pd.DataFrame(value1)
key1['value']=value1


# In[ ]:


key1.shape


# In[ ]:


key.shape


# In[ ]:


x=inventory[1].unique()


# In[ ]:


np.shape(x)


# In[ ]:




