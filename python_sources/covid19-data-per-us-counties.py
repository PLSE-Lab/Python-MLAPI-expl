#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df.corr().plot.bar()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=df, marker='o', color='purple') 
plt.title('Cases per day in the US') # Title
plt.xticks(df.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:




