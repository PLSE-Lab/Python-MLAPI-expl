#!/usr/bin/env python
# coding: utf-8

# From Rachel Tatman: https://www.kaggle.com/rtatman/the-5-day-data-challenge

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# In[ ]:


# Read data
dataframe = pd.read_csv('../input/DigiDB_movelist.csv')


# In[ ]:


# See part of the data
dataframe.head()


# In[ ]:


# Create array for SP Cost and power data
cost_power_array = pd.crosstab(dataframe['SP Cost'], dataframe['Power'])
cost_power_array.sample(10)


# In[ ]:


# Chi-Squared Test between Cost and Power 
stats.chi2_contingency(cost_power_array)


# * Chi-Squared: 1145
# * P-value: 8.5 * 10^-43 --> difference between power due to SP cost is statistically significant
# * Degrees of Freedom: 558 

# In[ ]:


# Scatterplot
sns.scatterplot(x = 'SP Cost', y = 'Power', data = dataframe).set_title('Memory vs. Lv 50 HP')


# In[ ]:




