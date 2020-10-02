#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The goal of this exercise is to create customer segmentation and analysis for a retail store within a mall, further create an unsupervised machine learning model such as KMeans Clustering to provide insights to the marketing team.

# ### Steps to follow
# <ul>
#     <li>Import the libraries</li>
#     <li>Clean the data</li>
#     <li>Analyze the data</li>
#     <li>Develop Model</li>
#     <li>Evaluate the Model</li>
# </ul>

# In[27]:


import numpy as np
import pandas as pd


# ## Clean the Data

# In[28]:


df=pd.read_csv('../input/Mall_Customers.csv')


# In[29]:


df.head(5)


# In[30]:


missing_data=df.isnull()


# In[31]:


missing_data.head(5)


# In[32]:


df.shape


# In[33]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')


# #### There are no missing data

# In[34]:


df.head(5)


# ## Analyze the Data

# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


df.dtypes


# #### Data types are correct and doesnt require to be updated

# In[37]:


df.corr()


# In[38]:


df[['Age', 'Annual Income (k$)']].corr()


# In[39]:


sns.regplot(x='Age', y='Annual Income (k$)', data=df)
plt.ylim(0,)


# #### There is a negative weak correlation between Age and Annual Income

# In[40]:


df[['Age', 'Spending Score (1-100)']].corr()


# In[41]:


sns.regplot(x='Age', y='Spending Score (1-100)', data=df)
plt.ylim(0,)


# #### There is a decent negative correlation between Age and Spending Score

# In[42]:


df[['Spending Score (1-100)','Annual Income (k$)']].corr()


# In[43]:


sns.regplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.ylim(0,)


# #### There is almost no correlation between Annual Income and Spending Score

# In[45]:


sns.boxplot(x='Genre', y='Annual Income (k$)', data=df)


# In[47]:


sns.boxplot(x='Genre', y='Spending Score (1-100)', data=df)


# In[46]:


plt.figure(1, figsize=(15,5))
sns.countplot(y='Genre', data=df)
plt.show()


# ### Conclusion: Important Variables are Age and Spending Score 

# ## Develop the Model

# ### Clustering using K-means

# In[48]:


from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


# In[49]:


X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)


# In[50]:


plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# In[51]:


algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_


# In[52]:


h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 


# In[53]:


plt.figure(1 , figsize = (10 , 5) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()


# In[ ]:




