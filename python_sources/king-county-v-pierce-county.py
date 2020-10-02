#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file read/write
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions


# In[ ]:


df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

#I droped FIPS column. 
##not relevant for this analysis.
US=df.drop(['fips'], axis = 1) 
US


# **US COVID19 cases as of April 1, 2020**

# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
US.groupby("state")['cases'].max().plot(kind='bar', color='darkgreen')


# **Washington state**

# In[ ]:


WA=US.loc[US['state']== 'Washington']
WA


# **Across Washington counties**

# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="cases", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="deaths", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()

