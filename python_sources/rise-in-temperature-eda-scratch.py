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


# # Importing *Matplotlib* and *Seaborn*

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# # Loading the data and fetching informations

# In[ ]:


df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['Region'].value_counts()


# We have data for 7 different regions.

# In[ ]:


df.info()


# # Beginning with Visualization

# # To Learn the trend, we plot the ***Lineplot***

# In[ ]:


plt.figure(figsize=(15,8))
sns.set_style('darkgrid')

sns.lineplot(x = df.loc[df['Region']=='Asia','Year'], y = df.loc[df['Region']=='Asia','AvgTemperature'], lw=8, color = 'red')


# # To see the distributions, we plot the *histogram*, *kde* (kernel distribution evaluation) and *Jointplot*

# # 1. Histogram (DistPlot)

# In[ ]:


plt.figure(figsize=(15,8))

sns.distplot(df['AvgTemperature'],
            kde_kws = {'color':'orange','lw':6},
            hist_kws = {'color':'red','lw':6})


# # 2. KDE

# In[ ]:


plt.figure(figsize=(15,8))

sns.kdeplot(df['AvgTemperature'], shade=True)


# # 3. JointPlot

# In[ ]:


sns.jointplot(x = df.loc[df['Region']=='Asia','Year'], y = df.loc[df['Region']=='Asia','AvgTemperature'], kind = 'reg')


# # To check the relationships, we plot the *bargraphs*, *scatterplots*, *regplot*

# # 1. Bar Graph

# In[ ]:


plt.figure(figsize=(15,8))

sns.barplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'])


# # 2. ScatterPlot

# In[ ]:


plt.figure(figsize=(15,8))

sns.scatterplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'], hue=df.loc[df['Region']=='Asia','City'])


# # 3. RegPlot

# In[ ]:


plt.figure(figsize=(15,8))

sns.regplot(x=df.loc[df['Region']=='Asia','Year'], y=df.loc[df['Region']=='Asia','AvgTemperature'])


# # **There are other plots through which we can evaluate the relationships, they are; *Heatmaps*, *Swarmplots* and *Lmplots***

# # I think you've got a basic idea about how to begin with data visualization. I would suggest you to visit the documentations of these plots in seaborn website to further improve upon your skills. 

# # THANK YOU !
