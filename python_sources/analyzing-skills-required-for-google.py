#!/usr/bin/env python
# coding: utf-8

# ## Skills that are valued at Google

# ![](http://lh3.googleusercontent.com/jN9tX6dCJ6_XL9E4K1KCO2Tuwe9_rYUbwv723eu6XGI0PWGLcPs0259VscOu249PPKKcU5AOXrq6JnleEaoK6K_JvZ2PY9lw3pMApzOpTQ=s660)

# ## Importing library 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir('../input/'))


# ## Reading the data

# In[ ]:


data=pd.read_csv('../input/google-job-skills/job_skills.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## Data Vizulization

# In[ ]:


sns.set(style="darkgrid")
sns.countplot(data['Company'])
plt.title('')

print(data['Company'].value_counts())


# In[ ]:


plt.title('Top 10 Job Titles')
top_title=data['Title'].value_counts().head(10)
top_title.plot(kind='bar')


# In[ ]:


plt.title('Top 10 Location')
top_location=data['Location'].value_counts().sort_values(ascending=False).head(10)
top_location.plot(kind='bar')


# In[ ]:


plt.figure(figsize=(10,6))
data['Category'].value_counts().plot(kind='bar')


# ## Missing Values

# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.isnull().any()


# ## Word Counts 

# In[ ]:


from collections import Counter
cnt = Counter()
for text in data['Minimum Qualifications'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[ ]:


for text in data['Preferred Qualifications'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# *If you have any suggestions please let me know in the comment section below. Thanks for reading.*
