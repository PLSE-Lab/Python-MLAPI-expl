#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


plt.style.available


# In[93]:


plt.style.use('seaborn-dark')


# In[10]:


data = pd.read_csv('../input/Mall_Customers.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data['Gender'].value_counts().plot(kind='bar', title='Plotting records by borough', figsize=(10, 4),align='center')


# In[29]:


fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(data['Annual Income (k$)'],
             hist=False,
             kde_kws={'shade':True},
            ax = ax)
ax.axvline(x= 66, color='m', linestyle='--', linewidth=2)


# In[18]:


fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"], c=data.index)
ax.set_xlabel("Annual Income")
ax.set_ylabel('Spedning Score')


# In[30]:


gender = pd.get_dummies(data['Gender'])


# In[32]:


gender = gender['Male']


# In[33]:


data = pd.concat((data,gender),axis=1)


# In[36]:


data.drop(['Gender'],axis=1,inplace=True)


# In[37]:


data.head()


# In[66]:


y = data['Spending Score (1-100)'].values
X = data[['Age','Annual Income (k$)','Male']].values


# In[67]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# In[68]:


kmeans = KMeans(n_clusters=3)


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[70]:


kmeans.fit(X_train,y_train)


# In[71]:


y_pred = kmeans.predict(X_test)


# In[72]:


print(kmeans.labels_)


# In[77]:


clusters = kmeans.fit_predict(X)


# In[96]:


fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X[clusters == 0, 0], X[clusters == 0, 1], s = 80, c = 'Red')
ax.scatter(X[clusters == 1, 0], X[clusters == 1, 1], s = 80, c = 'Blue')
ax.scatter(X[clusters == 2, 0], X[clusters == 2, 1], s = 80, c = 'Green')
ax.legend(['Cluster 1','Cluster 2','Cluster 3'])


# In[ ]:




