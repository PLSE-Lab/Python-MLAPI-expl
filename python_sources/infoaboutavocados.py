#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv') #reading data


# In[ ]:


data.info() # get info about data 


# In[ ]:


data.columns #show columns


# In[ ]:


data.corr() # it helps to understand relationship between features


# In[ ]:


#visualization of correalation (seaborn library)
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True , linewidths = .5 , fmt= '.1f', ax=ax )
plt.show()


# In[ ]:





# In[ ]:


data.head(10)


# In[ ]:


data.tail(10) 


# In[ ]:


#line plot
data.AveragePrice.plot(kind="line",color="g",label="AveragePrice",linewidth = 1,grid = True, alpha=0.5, linestyle=":")
data.year.plot(kind="line",color="r",label="year",linewidth = 1,grid = True, alpha=1,linestyle="-.")
plt.legend()
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.title("Line Plot")
plt.show()


# In[ ]:


#scatter plot
data.plot(kind="scatter" , x="AveragePrice" , y="XLarge Bags" , alpha=0.5 , color="red")
plt.title("Scatter Plot")
plt.show()


# In[ ]:


#histogram plot
data.AveragePrice.plot(kind="hist",bins=100,figsize=(10,10),grid=True)
plt.xlabel("AveragePrice")
plt.title("Histogram Plot")
plt.show()

