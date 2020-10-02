#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# If we don't add **encoding='ISO-8859-1'** to the code, we get this error.
# 
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 18: unexpected end of data
# 

# In[ ]:


data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1') 
#The csv file was read, transferred to the data.


# In[ ]:


print(type(data)) #type of data


# In[ ]:


data.dtypes #content of data and data types


# In[ ]:


data.info() # Information about data


# In[ ]:


data.columns # Column names of data


# There are certain parameters that allow us to understand the relationship between features.
# 
# ** correlation map ** -> If the correlation between the two features is 1 (or very close to 1), they are directly proportional to each other. If it is close to -1 or -1, there is an inverse ratio between them. If it is close to 0 and 0, there is no relationship between the properties.
# 

# In[ ]:


data.corr() #return correlations between features


# ** heatmap -> ** Enables visualization of the data frame from data.corr ().
# 
# ** annot = True -> ** Specifies that the numbers above the frames appear in the visualized output.
# 
# ** linewidths = .5 -> ** The thickness of the line between the frames in the output is 0.5.
# 
# ** fmt = '.1f' -> ** For numbers above the frame, 1 digit is printed after 0.
# 
# ** ax = ax -> ** f, ax at ax
# 
# ** plt.subplots (figsize = (18, 18)) -> **The size of the figure, the output squares is 18.18.
# 
# ** dark colors -> ** correlation low (-1)
# 
# ** light colors -> ** correlation high (1)

# In[ ]:


f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.7,fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(15) #Indicates the first 10 values.


# ** data.Speed.plot (kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':') **
# 
# ** kind = 'line' -> ** which kind of plot we will use -> line, scatter, ...
# 
# ** color = 'g' -> ** the color of that line is green
# 
# ** label = 'Speed' -> ** the speed column in the dataframe
# 
# ** linewidth -> ** line plot thickness
# 
# ** alpha -> ** transparency
# 
# ** grid -> ** check the background of the chart
# 
# ** linestyle = ':' -> ** lines are ':'
# 

# Note: If we do not give a numeric value as a label,maybe,then we receive a bo numeric data to plot error.

# In[ ]:


data.latitude.plot(kind = 'line', color = 'g',label = 'latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.longitude.plot(kind = 'line',color = 'r',label = 'longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('latitude')              
plt.ylabel('longitude')
plt.title('Line Plot')           
plt.show()


# In[ ]:


#Relationship between latitude and longitude

data.plot(kind='scatter', x='latitude', y='longitude',alpha = 0.5,color = 'red')
plt.xlabel('latitude')              
plt.ylabel('longitude')
plt.title('latitude longitude Scatter Plot')   

plt.show()


# In[ ]:


# A different use of it scatter plot
plt.scatter(data.latitude,data.longitude,color="red",alpha=0.5)
plt.show()


# In[ ]:


# Histogram
data.longitude.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


data.longitude.plot(kind = 'hist',bins = 50)
plt.clf() #cleans it up again you can start a fresh

