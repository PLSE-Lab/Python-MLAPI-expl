#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

crime = pd.read_csv("../input/crime.csv")
crime


# In[ ]:


sns.lmplot('Population', 'Murder rate', data=crime, fit_reg=False)


# In[ ]:


sns.distplot(crime[['Population']])


# In[ ]:


sns.distplot(crime[['Murder rate']])


# In[ ]:


crime.mean()


# In[ ]:


crime.median()

