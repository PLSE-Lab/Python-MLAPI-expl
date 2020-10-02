#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import scipy.stats as stats
import pylab

print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/NBA_player_of_the_week.csv')


# In[ ]:


df = df['Weight']


# In[ ]:


df.unique()


# In[ ]:


df = df.apply(lambda s: float(s[:-2]) if 'kg' in s else int(s) * 0.453592) # unify kg and pounds


# In[ ]:


np.sort(df.unique())


# In[ ]:


df.describe()


# In[ ]:


df.median()


# In[ ]:


stats.variation(df)


# In[ ]:


stats.kurtosis(df)


# In[ ]:


stats.skew(df)


# In[ ]:


df.plot.box()


# In[ ]:


df.hist(bins=5)


# In[ ]:


df.hist(bins=10)


# In[ ]:


df.plot.density()


# In[ ]:


probplot = sm.ProbPlot(df.values, stats.t, fit=True)


# In[ ]:


_ = probplot.ppplot(line='45')


# In[ ]:


_ = probplot.qqplot(line='45')


# In[ ]:




