#!/usr/bin/env python
# coding: utf-8

# How to use exponential time decay with pandas dataframe?    
#      
# Some theory about this topic: https://en.wikipedia.org/wiki/Exponential_decay.    
# For preparing data sample I used [Time series maker](http://mbonvini.github.io/TimeSeriesMaker/), you can find random-time-series.csv data sample in [projects](https://github.com/pythonthebilly) data folder.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/random-time-series.csv")


# Lets explore imported dataset.

# In[ ]:


print(df.head())


# In[ ]:


df.plot(x='date', y='value');


# We will add a new column 'rank' which is row number in dataframe sorted by date in reverse order.

# In[ ]:


df['rank'] = (df['date'].rank(ascending=False) - 1).astype('int64')


# In[ ]:


print(df.head())


# We will add a new column 'time_decay_value' with transformed value using exponential time decay.

# In[ ]:


time_decay_const = 0.99

df['time_decay_value'] = df['value'] * pow(time_decay_const, df['rank'])


# In[ ]:


print(df.head())


# Lets see what happend with our dataset.

# In[ ]:


df.plot(x='date', y='time_decay_value');


# If you want to have faster decay you can change a value of time_decay_const.
