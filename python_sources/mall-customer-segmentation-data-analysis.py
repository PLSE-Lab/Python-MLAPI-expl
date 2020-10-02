#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Mall Customer Segmentation Data
get_ipython().run_line_magic('reset', '-f')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[ ]:


cseg= pd.read_csv("../input/mall-customer-segmentation-data/Mall_Customers.csv")


# In[ ]:


cseg.head()


# In[ ]:


cseg.shape


# In[ ]:


cseg.dtypes


# In[ ]:


cseg['Gender'].unique()


# In[ ]:


cseg= cseg.rename(columns= {"Annual Income (k$)": "annualincome","Spending Score (1-100)":"spendingscore"})


# In[ ]:


cseg.head()


# In[ ]:


cseg.describe()


# In[ ]:


cseg.isnull().sum()        # no null values found in the given columns


# In[ ]:


cseg.groupby('Gender').spendingscore.mean()   #Average spending score of Female is higher than Men


# In[ ]:


cseg.groupby('Gender').annualincome.mean()


# In[ ]:


cseg.groupby('Gender').spendingscore.mean().plot(kind= 'bar')


# In[ ]:


cseg.groupby('Gender').spendingscore.mean().plot(kind= 'line')


# In[ ]:


sns.jointplot(cseg.annualincome, cseg.spendingscore, kind = 'reg')


# In[ ]:


cseg.info()


# In[ ]:


sns.pairplot(data=cseg, hue='Gender')


# In[ ]:


sns.pairplot(cseg, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True))


# In[ ]:


sns.pairplot(cseg, kind="reg")


# In[ ]:


sns.heatmap(cseg.corr(), cmap='coolwarm')


# In[ ]:


cseg.drop(columns=['CustomerID','Gender'],inplace= True)  #Dropping columns not needed


# In[ ]:


cseg.head(10)


# In[ ]:


# Copy 'Age' column to another variable and then drop it
#     We will not use it in clustering
y = cseg['Age'].values
cseg.drop(columns = ['Age'], inplace = True)


# In[ ]:


X_train, X_test, _, y_test = train_test_split(cseg,
                                               y,
                                               test_size = 0.25
                                               )


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


# Develop model
# Create an instance of modeling class
clf = KMeans(n_clusters = 2)
# Train the class over data
clf.fit(X_train)


# In[ ]:


# So what are our clusters?
clf.cluster_centers_


# In[ ]:


clf.cluster_centers_.shape                      


# In[ ]:


clf.labels_


# In[ ]:


clf.labels_.size


# In[ ]:


clf.inertia_


# In[ ]:


# 6 Make prediction over our test data and check accuracy
y_pred = clf.predict(X_test)
y_pred


# In[ ]:


np.sum(y_pred == y_test)/y_test.size


# In[ ]:


# 7.0 Are clusters distiguisable?
sns.scatterplot('annualincome','spendingscore', hue = y_pred, data = X_test)


# In[ ]:


# 7.1 Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # How many clusters?
    n_clusters = i+1
    # Create an instance of class
    clf = KMeans(n_clusters = n_clusters)
    # Train the kmeans object over data
    clf.fit(X_train)
    # Store the value of inertia in sse
    sse.append(clf.inertia_ )


# In[ ]:


sns.lineplot(range(1, 11), sse)


# In[ ]:


clf1 = KMeans(n_clusters = 5)


# In[ ]:


clf1.fit(X_train)


# In[ ]:


clf1.cluster_centers_


# In[ ]:


clf1.cluster_centers_.shape


# In[ ]:


clf1.labels_


# In[ ]:


clf1.labels_.size


# In[ ]:


clf1.inertia_


# In[ ]:


y_pred = clf1.predict(X_test)
y_pred


# In[ ]:


np.sum(y_pred == y_test)/y_test.size


# In[ ]:


sns.scatterplot('annualincome','spendingscore', hue = y_pred, data = X_test)


# In[ ]:




