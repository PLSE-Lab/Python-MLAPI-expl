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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
plt.plot([10, 20, 30, 40])
plt.show()


# In[ ]:


plt.plot([1,2,3,4], [12,43,25,15])
plt.show()


# In[ ]:


plt.title('plotting')
plt.plot([10,20,30,40])
plt.show()


# In[ ]:


plt.title('legend')
plt.plot([10,20,30,40], label='asc')
plt.plot([40,30,20,10], label='desc')
plt.legend()
plt.show()


# In[ ]:


plt.title('legend')
plt.plot([10,20,30,40],color='skyblue', label='asc')
plt.plot([40,30,20,10],'pink', label='desc')
plt.legend()
plt.show()


# In[ ]:


plt.title('legend')
plt.plot([10,20,30,40],color='r', linestyle='--', label='asc')
plt.plot([40,30,20,10],'g', ls=':', label='desc')
plt.legend()
plt.show()


# In[ ]:


plt.title('legend')
plt.plot([10,20,30,40],'r.',  label='asc')
plt.plot([40,30,20,10],'g^', label='desc')
plt.legend()
plt.show()


# In[ ]:




