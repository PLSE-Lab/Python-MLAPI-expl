#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
data=pd.read_csv('../input/primary_results.csv')
data.head()


# In[ ]:


aggregate=data.groupby('candidate').sum()['votes']
aggregate.name='sum_votes'
data=data.join(aggregate,on='candidate')
data.tail()


# In[ ]:


data_sort=data.sort('sum_votes',ascending=False)
data_sort.head()


# In[ ]:


data_t_c=data_sort[(data_sort.candidate=='Donald Trump') | (data_sort.candidate=='Hillary Clinton')]

data_t_c.head()


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(25,20))
sns.barplot(x='state_abbreviation',y='votes',data=data_t_c,hue='candidate')


# In[ ]:


data_sort_state=data_sort[data_sort['state']=='kentucky']
sns.pairplot(data_sort,hue='party',palette="Set2", diag_kind="kde", size=4.5)

