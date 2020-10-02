#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Load data

# In[ ]:


df = pd.read_csv('../input/GHG-Mexico-1990-2010.csv')


# In[ ]:


df.head()


# ## Data Visualization

# In[ ]:


plt.figure(figsize=(12,7))
sns.barplot(data=df, x='Sector', y='Amount', hue='GHG')


# In[ ]:


px.line(df[df['GHG'] == 'CO2'], x='Year', y='Amount', color='Subsector')


# **What happened to Mexican transport CO2 emissions in 2001?**

# In[ ]:


px.line(df[df['Subsector'] == 'Transport'], x='Year', y='Amount', color_discrete_sequence=['LimeGreen', 'Blue', 'Red'], color='GHG')


# In[ ]:


transport_CO2 = df[df['GHG'].apply(lambda x: x == 'CO2')][df['Subsector'] == 'Transport']
transport_CO2.head()


# Is there a way to find out the cause of this sudden dip?
# 
# Excerpt from [here](https://reliefweb.int/report/mexico/mexico-hurricane-juliette-ocha-situation-report-no-5) regarding 2001's Hurricane Juliette.
# > Situation
# > 
# > 4. In the southernmost part of Baja California, power lines have been downed and roads have been cut off. Damage caused to housing.
# > 
# > National response
# > 
# > 5. The Governor of the State of Baja California Sur has declared the disaster affected areas a Disaster Zone.
# > 
# > 6. Police and rescue workers provided assistance to some 3,000 people cut off by heavy flooding, and evacuated some 800 people from their vulnerable housing
# 
# Could this have caused significant damage to infrastructure to explain the noticeable reduction?
