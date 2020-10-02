#!/usr/bin/env python
# coding: utf-8

# # GDP per Capita & Number of Suicides

# ## prepare the weapon

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## prepare the ammo

# In[ ]:


raw_data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
raw_data.info()


# In[ ]:


raw_data.head()


# ## Analyze

# #### selecting needed variabels and drop unimportant variabels

# In[ ]:


suicides_gdp = (raw_data.groupby(['country', 'year'],as_index = False)
.agg({'suicides_no':'sum','gdp_per_capita ($)': 'mean'}))


# #### How many countries?

# In[ ]:


country = suicides_gdp['country'].unique()


# In[ ]:


len(country)


# #### data info and correlation

# In[ ]:


suicides_gdp.head()


# In[ ]:


suicides_gdp.info()


# In[ ]:


x = suicides_gdp['gdp_per_capita ($)']
y = suicides_gdp['suicides_no']
plt.xlabel("gdp per capita ($)")
plt.ylabel("number of suicides")
plt.scatter(x, y)
plt.show()


# In[ ]:


x.corr(y)

