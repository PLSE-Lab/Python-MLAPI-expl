#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


us_videos=pd.read_csv('../input/USvideos.csv')
us_category=pd.read_json('../input/US_category_id.json')


# In[ ]:


us_videos.head()


# In[ ]:


us_category.head()


# In[ ]:


cdata={}
for index, row in us_category.iterrows():
    cid=row['items'].get('id')
    title=row['items'].get('snippet').get('title')
    cdata.update({cid:title})


# In[ ]:


def compute_title(i):
    if str(i) in cdata:
        return cdata[str(i)]
    else:
        return np.nan


# In[ ]:


us_videos['category_title']=us_videos['category_id'].apply(compute_title)


# In[ ]:


us_videos.head(5)


# In[ ]:


us_videos.info()


# In[ ]:


us_videos.describe()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(us_videos.isnull(),cbar=False,yticklabels=False,cmap="viridis")


# In[ ]:


us_videos[us_videos['description'].isnull()==True].count()


# In[ ]:


plt.figure(figsize=(14,8))
sns.set_style('whitegrid')
sns.countplot(y='category_title',data=us_videos)
plt.ylabel('Category Title')


# In[ ]:


plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="likes",data=us_videos,alpha=0.5)


# In[ ]:


us_videos.drop('description',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="comment_count",data=us_videos,alpha=0.5)


# In[ ]:


corr_df=us_videos.corr()


# In[ ]:


corr_df


# In[ ]:


plt.figure(figsize=(14,6))
sns.heatmap(corr_df,annot=True)


# In[ ]:


plt.figure(figsize=(14,6))
sns.jointplot(x='likes',y='comment_count',data=us_videos,color="green",kind='reg')


# In[ ]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='likes',data=us_videos)
plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='dislikes',data=us_videos)
plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='comment_count',data=us_videos)
plt.xticks(rotation=30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




