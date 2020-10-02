#!/usr/bin/env python
# coding: utf-8

# ## I am gonna look through the data ##
# 
# I'll build violin plots for some cathegorical variables

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
style.use('ggplot')


# In[ ]:


data = pd.read_csv('../input/foreveralone.csv')
data.head(3)


# In[ ]:


data.info()


# In[ ]:


sns.violinplot(data.age, data.pay_for_sex, hue='virgin', data=data, split=True)


# In[ ]:


data.pivot_table(['friends', 'age'], ['virgin'], aggfunc='mean')


# In[ ]:


sns.violinplot(data.age, data.bodyweight, hue='virgin', data=data, split=True)


# In[ ]:


sns.violinplot(data.age, data.gender, hue='virgin', data=data, split=True)


# In[ ]:


sns.violinplot(data.age, data.attempt_suicide, hue='virgin', data=data, split=True)


# In[ ]:


sns.violinplot(data.age, data.social_fear, hue='depressed', data=data, split=True)


# In[ ]:


sns.jointplot(data.age, data.friends)


# In[ ]:


data.groupby(['gender']).income.value_counts(normalize=True)


# In[ ]:


data.groupby(['gender']).employment.value_counts(normalize=True)


# In[ ]:


data.groupby(['gender']).edu_level.value_counts(normalize=True)

