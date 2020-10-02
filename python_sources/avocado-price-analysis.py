#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


# Avocado trees do best at moderately warm temperatures 60 F to 85 F (16 C to 29 C, or in this graph *289 K to 303 K*)

# In[ ]:


city= 'Chicago'


# **Extract Data**

# In[ ]:


avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv', header=0, index_col=1)
temperature_df = pd.read_csv('../input/temperature/temperature.csv', header=0, index_col=0)


# **Filter Avocado Data**

# In[ ]:


avocado_df = avocado_df[avocado_df['region'] == city].groupby(['Date']).agg('mean')
avocado_df.index = pd.to_datetime(avocado_df.index, format='%Y-%m-%d')
avocado_df = avocado_df.loc['2015-01-04':'2017-12-30', :]


# **Filter Temperature Data**

# In[ ]:


temperature_df.index = pd.to_datetime(temperature_df.index, format='%d/%m/%Y %H:%M').strftime('%Y-%m-%d')
temperature_df.index = pd.to_datetime(temperature_df.index, format='%Y-%m-%d')
temperature_df = temperature_df[[city]]
temperature_df = temperature_df.loc['2015-01-04':'2017-12-30']
temperature_df = temperature_df.resample('D').mean()


# **Linear Interpolation**

# In[ ]:


avocado_df = avocado_df.reindex(temperature_df.index).interpolate(how="linear")


# **Pearson Correlation Coefficient**

# In[ ]:


corr, p_value = pearsonr(temperature_df[[city]], avocado_df[['AveragePrice']])
print(corr) # 0.39518515
print(p_value) # 5.06963491e-41


# **Avocado Price Graph**

# In[ ]:


avocado_df['AveragePrice'].plot(style='b-', x_compat=True)
plt.title('Avocado Average Price')
plt.ylabel('Average Price')
plt.xlabel('Date')
plt.show()


# **Chicago Temperature Graph**

# In[ ]:


temperature_df['Chicago'].plot(style='r.', x_compat=True)
title = city + 'Temperature'
plt.title(title)
plt.ylabel('Temperature (K)')
plt.xlabel('Date')
plt.show()


# Insight: the temperature seems to have a positive correlation with the price of avocado

# In[ ]:




