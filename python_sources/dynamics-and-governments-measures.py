#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 
# In this notebook I discuss how the dynamics of the infections can change in response to goverment measures. I am using data from two different sources and combine them into one graph.
# 
# # Rational
# 
# This work demonstrates the number of cases vs. average growth in recent past. I assume that like any dynamic system every derivation is an independent variable, but since difference is unstable, I introduce a leaky moving average to compensate this unstability. This can be interpreted as results of some tests might be delayed to the next days to smooth the average new cases.
# Please note that this work is influenced by https://aatishb.com/covidtrends/.

# We import the required libraries:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
plt.rcParams['figure.figsize'] = [15, 10]


# and read the data:

# In[ ]:


cases = pd.read_csv('/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
restrictions = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv')


# Some prepairation including definition of moving average and circle:

# In[ ]:


#print(cases)
m = 7
kernel = np.arange(0, 1, 1/m)
kernel = kernel / np.sum(kernel)
kernel = np.reshape(kernel, (-1, 1))
kernel = np.flipud(kernel)
kernel = kernel[:, 0]

t = np.arange(0, 2*np.pi+1, np.pi/20)
r1, r2 = 2000, 200
circle = (np.cos(t), np.sin(t))


# Compensating the naming convention differences with a dictionary:

# In[ ]:


country = 'countryName'
countryMap = defaultdict(lambda :country)
countryMap['US'] = 'United States of America'


# Different coloring schemes for different categories:

# In[ ]:


categoryMap = defaultdict(lambda: 'k')
categoryMap['Public health measures'] = 'g'
categoryMap['Social distancing'] = 'y'
categoryMap['Movement restrictions'] = 'b'
categoryMap['Governance and socio-economic measures'] = 'c'
categoryMap['Lockdown'] = 'r'


# Now based on maximum number of cases we sort the countries and plot the smoothed growth of cases vs. total cases.
# We also mark every government measure by a circle around that point.

# In[ ]:


countries = cases.groupby('country_region').max().sort_values('confirmed', ascending=False).index
legendMap = {}

for country in countries[0:10]:
    countryData = cases[cases['country_region']==country]
    
    confirmed = countryData['confirmed'].values
    delta_confirmed = countryData['delta_confirmed'].values
    deaths = countryData['deaths'].values
    dates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), countryData['last_update'].values))
    
    smooth_delta = np.convolve(delta_confirmed, kernel, mode='valid')
    smooth_delta = np.concatenate(([0]*(m-1), smooth_delta))
    
    legendMap[country] = plt.plot(confirmed, smooth_delta, 'x-')[0]

    
    thisRestrictions = restrictions[restrictions['country'] == countryMap[country]]
    lastTimeIndex = 0
    timeIndex = 0
    n = 1
    for i, row in thisRestrictions.iterrows():
        if type(row['date_implemented']) is str:            
            d = datetime.strptime(row['date_implemented'], '%Y-%m-%d')
            lastTimeIndex = timeIndex
            try:
                timeIndex = dates.index(d)
            except:
                pass
                #print(d)
        else:
            continue
        xp, yp = confirmed[timeIndex], smooth_delta[timeIndex]
        #print(xp, yp)
        if (lastTimeIndex != timeIndex):
            n = 1
        n = n+.1
        plt.plot(xp+circle[0]*(r1*n), yp+circle[1]*(r2*n), categoryMap[row['category']])


        
first_legend = plt.legend(legendMap.values(), legendMap.keys(), bbox_to_anchor=(0.2,.4))
plt.gca().add_artist(first_legend)
        
from matplotlib.lines import Line2D
custom_lines = []    
names = []
for (n, c) in categoryMap.items():
    custom_lines.append(Line2D([0], [0], color=c, lw=2))
    names.append(n)
plt.legend(custom_lines, names);
plt.xlabel('cases');
plt.ylabel('average growth');


# # Some analysis
# We can see that on this plot, germany is followin italy's path very closely and they are both in better conditions than spain.
# 
