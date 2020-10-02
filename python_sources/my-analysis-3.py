#!/usr/bin/env python
# coding: utf-8

# # COVID-19 comparison using Pie charts
# 
# Created by (c) Shardav Bhatt on 17 June 2020

# # 1. Introduction
# 
# Jupyter Notebook Created by Shardav Bhatt
# 
# Data (as on 16 June 2020)
# References: 
# 1. Vadodara: https://vmc.gov.in/coronaRelated/covid19dashboard.aspx
# 2. Gujarat: https://gujcovid19.gujarat.gov.in/
# 3. India: https://www.mohfw.gov.in/
# 4. Other countries and World: https://www.worldometers.info/coronavirus/
# 
# In this notebook, I have considered data of COVID-19 cases at Local level and at Global level. The aim is to determine whether there is a significance difference between Global scenario and Local scenario of COVID-19 cases or not. The comparison is done using Pie charts for active cases, recovered cases and deaths.

# # 2. Importing necessary modules

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


# # 3. Extracting data from file

# In[ ]:


date = str(np.array(datetime.datetime.now()))

data = pd.read_csv('/kaggle/input/covid19-comparison-using-pie-charts/data_17June.csv')

d = data.values

row = np.zeros((d.shape[0],d.shape[1]-2))
for i in range(d.shape[0]):
    row[i] = d[i,1:-1]


# # 4. Creating a funtion to print % in Pie chart

# In[ ]:


def func(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}% ({:d})".format(pct, absolute)


# # 5. Plot pre-processing

# In[ ]:


plt.close('all')
date = str(np.array(datetime.datetime.now()))
labels = 'Infected', 'Recovered', 'Died'
fs = 20
C = ['lightskyblue','lightgreen','orange']

def my_plot(i):
    fig, axs = plt.subplots()
    axs.pie(row[i], autopct=lambda pct: func(pct, row[i]), explode=(0, 0.1, 0), textprops=dict(color="k", size=fs-2), colors = C, radius=1.5)
    axs.legend(labels, fontsize = fs-4, bbox_to_anchor=(1.1,1))
    figure_title = str(d[i,0])+': '+str(d[i,-1])+' cases on '+date
    plt.text(1, 1.2, figure_title, horizontalalignment='center', fontsize=fs, transform = axs.transAxes)
    plt.show()
    print('\n')


# # 6. Local scenario of COVID-19 cases

# In[ ]:


for i in range(4):
    my_plot(i)


# # My Observations:
# 
# 1. Death rate in Vadodara city is less compared to state and nation. Death rate of Gujarat is almost double compared to the nation. Death rate of India is less compared to the global death rate.
# 
# 2. Recovery rates of Vadodara and Gujarat are higher compared to national and global recovery rate. The recovery rate of India and of World are similar.
# 
# 3. Proportion of active cases in Vadodara and Gujarat is lower compared to national and global active cases. Proportion of active cases of India and world are similar. 

# # 7. Global scenario of COVID-19 cases

# In[ ]:


for i in range(4,d.shape[0]):
    my_plot(i)


# # Observations:
# 
# 1. Russia, Chile, Turkey, Peru have comparatively lower death rate i.e. below 3%. Mexico, Italy and France have comparatively higher death rate i.e. above 10%. 
# 
# 2. Germany, Chile, Turkey, Iran, Italy, Mexico have recovery rate above 75%. These countries are comming out of danger.
# 
# 3. Russia, India, Peru, Brazil, France, USA have have recovery rate below 53%. These countries needs to recover fast. 
# 
# 4. Proportion of active cases is least in Germany, Italy, Turkey is least.
