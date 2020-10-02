#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# I read Data. But data gives format error when reading data. Therefore i used encoding parameters. After i used info fonksiyon for information about data.

# In[ ]:


"../input/globalterrorismdb_0718dist.csv"
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding="ISO-8859-1")
data.info()


# Because data set is too big, I applied corr function on some part 

# In[ ]:


sns.heatmap(data.loc[0:5,"eventid":"location"].corr(),annot=True,linewidths=.5, fmt= '.1f')
plt.show()


# Visualize our data with line plot

# Terrorist activities in the first quarter of 1970

# In[ ]:


data.iyear[0:166].plot(kind="line",color='r',label="year",linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.country[0:166].plot(color="r",label="country", linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.xlabel('First Quarter') 
plt.ylabel('Country Count')
plt.title('Line Plot') 
plt.show()


# In[ ]:


data.plot(kind="scatter", x="iyear",y="country",color ="r",alpha = 0.5 )
plt.xlabel('Years') 
plt.ylabel('Country Count')
plt.title('Line Plot') 
plt.show()


# Data analysis

# In[ ]:


#more than 1000 events list
data[data["country"]>1000]


# In[ ]:


#1999 summer events
data[(data["country"]>1000) & (data["iyear"]==1999) &((data["imonth"]>5) & (data["imonth"]<9))]


# In[ ]:





# In[ ]:




