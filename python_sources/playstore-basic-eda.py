#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')


# In[ ]:


apps = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


apps.info()


# In[ ]:


apps.head()


# In[ ]:


apps['App'].nunique()


# Top 10 Apps by Category

# In[ ]:


plt.figure(figsize=(15,10))
category = apps.groupby('Category')['App'].count()
top_10 = category.sort_values(ascending=False).head(10)
top_10.plot(kind = 'bar')
plt.title('Top 10 Apps by category')
plt.show()


# In[ ]:


apps.head(2)


# In[ ]:


apps[apps['Rating'] == apps['Rating'].max()]


# In[ ]:


apps[pd.isnull(apps['Android Ver'])]


# 'Life Made WI-Fi Touchscreen Photo Frame' has Rating = 19.0 is an outlier in the dataset

# In[ ]:


apps.drop(index=10472, inplace=True)


# List of top rated apps on play store

# In[ ]:


apps['Reviews'] = apps['Reviews'].apply(lambda x : pd.to_numeric(x))


# In[ ]:


apps[apps['Rating'] == apps['Rating'].max()][apps['Reviews']>=100]


# In[ ]:


plt.figure(figsize=(15,12))
sns.scatterplot(x='Rating', y='Reviews', data=apps, hue='Category', palette='Set1')
plt.title('Review vs Rating distribution')
plt.show()


# In[ ]:


#apps[apps['Rating'] == apps['Rating'].max()]['App']


# In[ ]:


apps[(apps['Rating'] == apps['Rating'].max()) & (apps['Reviews']>=100)]['App'].count()


# List of Top rated games

# In[ ]:


apps[apps['Category']=='GAME'][(apps['Rating'] == apps['Rating'].max())][['App','Reviews']]


# In[ ]:


apps[apps['Category']=='GAME'][apps['Rating'] == apps['Rating'].max()][['App']].count()


# Top paid games

# In[ ]:


apps[apps['Category']=='GAME'][apps['Rating'] == apps['Rating'].max()][apps['Type']=='Paid']['App']


# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(apps['Content Rating'], palette='rainbow')
plt.title('Apps by content rating')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(apps[apps['Installs'] == apps['Installs'].max()]['Category'], palette='rainbow')
plt.title('Apps category with most installs')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




