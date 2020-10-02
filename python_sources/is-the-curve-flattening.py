#!/usr/bin/env python
# coding: utf-8

# # Is the curve flattening!
# 
# 
# As is with most of the other countries of the world, the number of Corvid-19 cases and the resulting deaths are increasing day by day in our country. Since the infection of the virus spreads through human contact, the growth of cummulative cases over time is expected to be exponential. As the number cases increase, the potential number of new infections come down as the virus can only infect a certain number of people, the worst case of which is the entire population being infected. So after a certain point the curve of infections would plateau. Correspondingly if we take a look at the trend of daily new cases, the graph would increase upto a point before starting to dip, emulating a bell shaped curve. The peak point of the bell curve corresponds to the starting point of the plateau of the cummulative curve. If the cummulative curve reachs it's plateau very rapidly, then the bell curve would reach a larger peaker which inturn would mean that a large number of people would be under treatment at that point in time, which would strain our health systems beyond it's capacity. This could have devestating effects on the death rate during this epidemic. So to improve our chances of surviving this pandemic without a huge loss, we need to flatten the bell curve, that is bring down it's peak by widening it. The government has declared a lockout in the country with the objective of reducing human contact and thus flatten the curve. The entire country is under lockdown for close to 8 weeks now and it is time ask the question, have we achieved what it was meant to achieve, flatten the curve?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import keras
from scipy.optimize import curve_fit
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Read data and clean-up.

# In[ ]:


India_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'], dayfirst=True)
India_data.drop(['Sno'], axis=1, inplace=True)
India_data.rename(columns={"State/UnionTerritory": "States"}, inplace=True)
India_data['Active Cases'] = India_data['Confirmed'] - India_data['Cured'] - India_data['Deaths']
India_daily= India_data.groupby(['Date'])['Active Cases'].sum().reset_index()


# In[ ]:


cases = India_data.groupby(['Date'])['Confirmed'].sum().values
daily = [cases[i] - cases[i-1] for i in range(1,cases.shape[0])]
plt.plot(cases)
plt.xlabel('No. of days')
plt.ylabel('No. of confirmed cases')
plt.title('Cummulative number of Corvid-19 cases in India')
fig = plt.gcf()
fig.set_size_inches(20, 10.5)


# Above is the plot of cummulative cases since 31st Jan 2020. As expected the curve is growing exponentially and it hasn't peaked yet.

# In[ ]:


plt.plot(daily, 'g')
fig = plt.gcf()
plt.xlabel('No. of days')
plt.ylabel('No. of new cases')
plt.title('Daily new cases of Corvid-19 in India')
fig.set_size_inches(20, 10.5)


# The new cases curve shows a clear upward trend. 

# In[ ]:


def sigmoid(x, a, b, c, d):
    x = [np.floor(-709*d + c) if -(z-c)/d > 709 else z for z in x] #to prevent math range error
    return [a + b/(1 + math.exp(-(z-c)/d)) for z in x]


# # A mathematical model
# The cummulative curve can be closely approximated using a sigmoid curve which has the shape shown in the below figure.
# 
# Reference: https://www.datasciencecentral.com/profiles/blogs/nonlinear-regression-of-covid19-infected-cases

# In[ ]:


plt.plot(sigmoid(np.linspace(-2000,2000,200), 0.01, 200000, 1000, 100));


# We will try to fit a sigmoid curve using 2 set of data, the current data and data from a week ago, and will try to figure out wether the condition of the infection in our country is worsening or improving.

# In[ ]:


# Current data
x = np.linspace(1, len(cases)+1, num=len(cases))
y = cases

# Last week's data
x_lw = x[:-7]
y_lw = y[:-7]

# Plot
plt.plot(x, y, label='Current')
plt.plot(x_lw, y_lw, label='Last week')
plt.xlabel('No. of days')
plt.ylabel('Confirmed cases')
plt.legend();


# # Predicting the curves

# In[ ]:


popt, pcov = curve_fit(sigmoid, x, y, bounds=((0, 0, 0, 0), (np.Inf, np.Inf, np.Inf, np.Inf))) # Fit current data to sigmoid
# Extent the curve to 210 days
length = 210
xt = np.linspace(1, length+1, num=length)
pred = sigmoid(xt, *popt)

popt_lw, pcov = curve_fit(sigmoid, x_lw, y_lw, bounds=((0, 0, 0, 0), (np.Inf, np.Inf, np.Inf, np.Inf))) # Fit last week's data to sigmoid
pred_lw = sigmoid(xt, *popt_lw) # Extent the curve to 210 days


# In[ ]:


print(popt)
print(popt_lw)


# In[ ]:


plt.plot(xt, pred, 'r', label='Fit based on current data')
plt.plot(xt, pred_lw, 'b', label="Fit based on last week's data")
plt.plot(cases, 'c', label='Actual data')
#plt.plot(xt, np.max(pred)*np.ones(xt.shape[0]), '^r')
plt.xlabel('No. of days')
plt.ylabel('No. of confirmed cases')
plt.title('Sigmoid fits for confirmed cases')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(20, 10.5)


# With current data the sigmoid would plateau at around 9 lakhs confirmed cases while last week's data suggest that confirmed cases would have peaked at just 7 lakhs. So definetly the situation has worsened. 

# Similarly we can have look at daily new cases curve.

# In[ ]:


diff = [pred[i]-pred[i-1] for i in range(1, len(pred))]
diff_lw = [pred_lw[i]-pred_lw[i-1] for i in range(1, len(pred_lw))]
daily = [cases[i] - cases[i-1] for i in range(1,cases.shape[0])]
plt.plot(diff, 'r', label='Fit based on current data')
plt.plot(diff_lw, 'b', label="Fit based on last week's data")
plt.plot(daily[1:], 'g', label='Actual data')
plt.xlabel('No. of days')
plt.ylabel('No. of new cases')
plt.title('Fits for new cases')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(20, 10.5)


# It is clear that the Corvid-19 situation in India is worsening and it is to continue longer than we would want it to. 
