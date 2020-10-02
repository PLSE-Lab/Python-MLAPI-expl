#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
df = df[['date', 'month','year', 'state', 'number']]
df.head()


# In[ ]:


df1 = df.groupby(by = ['year']).sum()['number']
df1.head()


# #### The following plot shows the total number of forest fires occuring between 1998 and 2017

# In[ ]:


plt.figure(figsize=(14,6))
ax = plt.plot(df1)
plt.title('Total fires in Brazil')
plt.xlabel('Year', fontsize = 14)
plt.ylabel('Number of Fires', fontsize = 14)
# ax.set_xlim(1998,2017)
from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))


# #### Let's forecast the total fire count for 5 years ahead using time series forecast methods (ARIMA).

# In[ ]:


model_obj =ARIMA(df1, order=(1,2,1)).fit(disp = -1)
model_obj.summary()


# In[ ]:


fc, se, conf = model_obj.forecast(5, alpha=0.1)  # 90% conf

# Make as pandas series
fc_series = pd.Series(fc, index=np.arange(2018, 2018+5))
lower_series = pd.Series(conf[:, 0], index=fc_series.index)
upper_series = pd.Series(conf[:, 1], index=fc_series.index)

# Plot
plt.figure(figsize=(14, 6), dpi=100)
plt.plot(df1, label='training')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# ## The above forecast graph shows that there's a decline in forest fires. There's a 90% chance that the mean values will lie within the range as shown in the graph

#  

# ### The following graphs show the fire occurences month-wise

# In[ ]:


df2 = df.groupby(by = ['month']).sum()['number'].reset_index()
df2


# In[ ]:


m = df2.number.idxmax()
mi = df2.number.idxmin()


# In[ ]:


plt.figure(figsize = (14,6))
b = plt.bar(x = df2.month, height = df2.number, width = 0.4)
b[m].set_color('red') 
b[mi].set_color('green')
plt.show()


# ## From the above graph, one can infer that Brazil has maximum fires in July and minimum in the month of April shown in red and green, respectively.

# In[ ]:




