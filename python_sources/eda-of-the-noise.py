#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")
df.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Let's take a look on the Big Picture

# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(df.time, df.signal)
plt.plot(df.time, df.open_channels,alpha=0.7)
plt.show()


# It's very interesting because we see some kind of pattern between **signal** and **open_channels** till 30 sec of our timestamp

# Ok, so that let's split our data with 10 sec intervals (yeah from official description of dataset we know that each batch was recorded with 5 seconds, btw let's do it in that way) 

# In[ ]:


b = 0
plt.figure(figsize=(16,10), dpi=100)
for i in range(0,401, 100):
    plt.subplot(2,3,b+1)
    b+=1
    plt.plot(df.time[(df.time<i+100) & (df.time>=i)], df.signal[(df.time<i+100) & (df.time>=i)])
    plt.plot(df.time[(df.time<i+100) & (df.time>=i)], df.open_channels[(df.time<i+100) & (df.time>=i)], alpha=0.7)

plt.show()


# As we can see, after 30s timestamp we cannot see some pattern between **signal** and **open_channels** instead of this we can notice some periodic signals. So that I go through their original [paper](http://https://www.nature.com/articles/s42003-019-0729-3.pdf) and found interesting note about how they added noise 
# 
# ## "The degree of noise could be altered simply by moving the patch-clamp headstage closer to or further from the PC. In some cases, driftwas added as an additional challenge via a separate Matlab scrip"

# So that I can make an assumption, that this noise was added artificially by moving patch-clamp headstage.

# ## I hope you find this helpfull :)
