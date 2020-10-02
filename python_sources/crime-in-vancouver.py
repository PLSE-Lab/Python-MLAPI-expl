#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/crime.csv')
df['counter'] = 1


# In[ ]:


df.head()


# # What are the various types of crime

# In[ ]:


plt.figure(figsize=(12,6))
plt.title('Types of Crime',fontdict={'fontsize':'30'},pad=20)
ax = sns.countplot(x='TYPE',data=df,palette='Blues_d', order = df['TYPE'].value_counts().index)
ax.set(xlabel='Types of Crime')
ax.set(ylabel='Counts')
plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='right')
plt.tight_layout()


# # Trends over the years of the crime in Vancouver

# In[ ]:


plt.figure(figsize=(12,6))
plt.title('Years Trend',fontdict={'fontsize':'30'},pad=20)
ax = sns.countplot(x='YEAR',data=df,palette='Blues_d')
ax.set(xlabel='Years', ylabel='Counts')
plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='right')
plt.tight_layout()


# # 5 safest area in Vancouver

# In[ ]:


df.groupby(['NEIGHBOURHOOD','TYPE']).count()['counter'].sort_values(ascending=True).head(5)


# # 5 dangerous area in Vancouver and major crime type

# In[ ]:


danger_region = df.groupby(['NEIGHBOURHOOD','TYPE']).count()['counter'].sort_values(ascending=False).head(10)
danger_region


# # Which month has the maximum crime rate over the given peroid of time?

# In[ ]:


df.groupby(['MONTH']).count()['counter'].sort_values(ascending=False).head(5)


# # Which day has the maximum crime rate over the given period of time?

# In[ ]:


df.groupby(['DAY']).count()['counter'].sort_values(ascending=False).head(5)


# # Which hour has the maximum crime rate over the given period of time?

# In[ ]:


df.groupby(['HOUR']).count()['counter'].sort_values(ascending=False).head(5)

