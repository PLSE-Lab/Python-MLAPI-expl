#!/usr/bin/env python
# coding: utf-8

# 
# # Solar Radiation Prediction - My first work on Python
# Simple first steps ...
# 
# Content: Import Data, Numpy, Matplotlib, Pandas, Seaborn, Dictionaries, Logic, Control flow and Filtering, Loop Data Structures.
# 
# An imported data in **Solar Radiation Prediction**, i used **Radiation**, **Temperature**, **Humidity** and **Speed** features.
# 
# Referance Kaan Can.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dt=pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')

dt.info()
dt.columns
dt.head(10)


# Matplotlib

# In[ ]:


dt.corr ()  


# In[ ]:


f,ax = plt.subplots(figsize=(14, 14)) 
sns.heatmap(dt.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)   


# In[ ]:


dt.Radiation.plot(kind = 'line', color = 'black',label = 'Radiation',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')
dt.Temperature.plot(kind = 'line', color = 'red',label = 'Temperature',linewidth=2,alpha = 0.5,grid = True,linestyle = ':')
plt.rcParams["figure.figsize"] = (13,11)
plt.legend(loc='upper right') 
plt.xlabel('x axis')                           
plt.ylabel('y axis')
plt.title('Line Plot')                         
plt.show()


# In[ ]:


dt.plot(kind='scatter', x='Pressure', y='Humidity',color = 'red',alpha = 0.5, )   
plt.scatter(dt.Pressure, dt.Humidity , color = 'blue', alpha = 0.5)     
plt.xlabel('Pressure')              
plt.ylabel('Humidity')
plt.title('Pressure - Humidity Scatter Plot')
plt.show()


# In[ ]:


dt.Temperature.plot(kind = 'hist',bins = 80,figsize = (12,12))    
plt.show()


# In[ ]:


dt.columns
df2 = pd.DataFrame(dt, columns = ['Data' , 'Radiation' , 'Temperature', 'Pressure' , 'Humidity']) 
df2


# In[ ]:


dt_dict = dt.to_dict()
dt_dict['Temperature']


# In[ ]:


'Humidity' in dt_dict


# In[ ]:


del dt_dict['UNIXTime']
dt_dict['UNIXTime']


# In[ ]:


series = dt['Radiation'] 
series


# **Logic, Control flow and Filtering**
# 
# 

# In[ ]:


dt[np.logical_and(dt['Temperature']>48, dt['Humidity']>80 )]


# In[ ]:


dt_dict 

dt_list=dt_dict.items()
dt_list

for index, value in enumerate(dt_list):
          print(index," : ",value)
print('')    



# In[ ]:


for index,value in dt[['Speed']][20:30].iterrows():    
    print(index," : ",value)

