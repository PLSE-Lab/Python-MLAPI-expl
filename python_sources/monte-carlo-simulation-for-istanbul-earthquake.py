#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv')
df


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df['Times'] = pd.to_datetime(df[' Time '], format='%Y%m%d', errors='ignore')


# In[ ]:


plt.figure(figsize=(12.5,4.5))
plt.plot(df.index, df[' Magnitude '], color='red')
plt.xlabel('Number of Earthquake',color='cyan')
plt.ylabel('Earthquake Intensity',color='cyan')
plt.grid(True)
plt.title("Earthquakes Near Istanbul", color='cyan')
plt.show()


# In[ ]:



plt.scatter(x=df[' Time '], y=df[' Magnitude '])
plt.xticks(rotation=120)
plt.xlabel('Date of Earthquakes',color='cyan')
plt.ylabel('Earthquake Intensity',color='cyan')

plt.title("Earthquakes Near Istanbul", color='cyan')
plt.show()


# In[ ]:


#Last 1 year eartquakes' average
df[' Magnitude '].mean()


# In[ ]:


#Monte Carlo Simulation for future Earthquakes
ticker = '/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv'
data = pd.DataFrame()
data[ticker] = pd.read_csv('/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv')[' Magnitude ']
log_returns= np.log(1 + data.pct_change())
log_returns.tail()


# In[ ]:


log_returns.plot(figsize=(10,6))
plt.show()


# In[ ]:


u = log_returns.mean()
u


# In[ ]:


var = log_returns.var()
var


# In[ ]:




drift = u- (0.5 * var)
drift


# In[ ]:


stdev = log_returns.std()
stdev


# In[ ]:


type(drift)


# In[ ]:


type(stdev)


# In[ ]:


np.array(drift)


# In[ ]:


drift.values


# In[ ]:


stdev.values


# In[ ]:




norm.ppf(0.95)


# In[ ]:


x = np.random.rand(10,2)
x


# In[ ]:




norm.ppf(x)


# In[ ]:


Z = norm.ppf(np.random.rand(10,2))
Z


# In[ ]:


t_intervals = 365
iterations = 5


# In[ ]:


daily_returns = np.exp(drift.values + stdev.values*norm.ppf(np.random.rand(t_intervals, iterations)))


# In[ ]:


daily_returns


# In[ ]:


S0 = data.iloc[-1]
S0


# In[ ]:




price_list = np.zeros_like(daily_returns)
price_list


# In[ ]:


price_list[0] = S0
price_list


# In[ ]:


for t in range(1, t_intervals):
  price_list[t] = price_list[t-1] * daily_returns[t]
price_list


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(price_list) 
plt.grid(True)
plt.xlabel('DAYS', color='orange')
plt.ylabel('Magnitude', color='orange')
plt.title('Monte Carlo Simulation for Istanbul Earthquake',color='cyan')
plt.show()


# # # **Last data is from 4th Feb. Today is 26th June. 143 Days. And around 143.day 5.5 magnitude earthquake hit in Mersin.**

# In[ ]:




