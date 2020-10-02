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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/data.csv')
data.info()


# In[ ]:


#we have to 18207 entries and 89 colums
data.corr()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.plot(kind='scatter',x='Balance',y='Finishing',alpha=0.5,color='red')
plt.xlabel('Positioning')
plt.ylabel('Finishing')
plt.title('Positioning & Finishing Scatter Plot')


# In[ ]:


print(data.Nationality.unique())


# In[ ]:


Turkey=data[data.Nationality == 'Turkey']


# In[ ]:


print(Turkey)


# In[ ]:


plt.plot(Turkey.GKHandling,Turkey.GKReflexes,color='red',label='Turkey')
plt.xlabel('GKHandling')
plt.ylabel('GKReflexes')
plt.show()


# In[ ]:


dictionary={'Club':'RealMadrid','Nationality':'Brazil'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['Club']='Barcelona' #We're changing value RealMadrid to Barcelona
print(dictionary.values())


# In[ ]:


dictionary['Position']='ST' #Add new key and value
print(dictionary.values())


# In[ ]:


del dictionary['Position'] #delete Position key and value
print(dictionary)


# In[ ]:


dictionary.clear() #clear dictionary keys and values
print(dictionary)


# In[ ]:


print(dictionary)


# In[ ]:


series=data['Value'] #create series
print(type(series))


# In[ ]:


data_frame=data[['Value']] #create data frame
print(type(data_frame))


# In[ ]:


x=data['Finishing']>90 #Filtering
print(x)
data[x]


# In[ ]:




