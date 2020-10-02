#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/CC GENERAL.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# we can say that Variable BALANCE, ONEOFF_PURCHASES, INSTALLMENT_PURCHASES,CASH_ADVANCE, CASH_ADVANCE_TRX, PURCHASE_TRX, CREDIT_LIMIT, PAYMENTS, and MINIMUM_PAYMENTS have outlier.

# In[ ]:


df.isnull().sum()


# In[ ]:


# fill null value by using its mean
df = df.fillna(df.mean())


# In[ ]:


df.isnull().sum()


# In[ ]:


#Remove Unneccasary column
df.drop('CUST_ID', axis = 1, inplace = True)


# In[ ]:


df.info()


# In[ ]:


# find the unique value of the int types

df[['CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'TENURE']].nunique()


# In[ ]:


# find the correlation 

sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)


# In[ ]:


sns.pairplot(df)


# # Feature generation

# here we can use scale function of sklearn.preprocessing. this function will put all variable at the same scale, with mean zero and standard deviation equal to one.

# In[ ]:


#creat copy of data
features = df.copy()

cols = ['BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES',
       'CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT',
       'PAYMENTS','MINIMUM_PAYMENTS']

# NOTE: Adding 1 for each value to avoid inf values
features[cols] = np.log(1 + features[cols])
features.head()


# In[ ]:


cols = list(features)
irq_score = {}

for c in cols:
    q1 = features[c].quantile(0.25)
    q3 = features[c].quantile(0.75)
    score = q3 - q1
    outliers = features[(features[c] < q1 - 1.5 * score) | (features[c] > q3 + 1.5 * score)][c]
    values = features[(features[c] >= q1 - 1.5 * score) | (features[c] <= q3 + 1.5 * score)][c]
    
    irq_score[c] = {
        "Q1": q1,
        "Q3": q3,
        "IRQ": score,
        "n_outliers": outliers.count(),
        "outliers_avg": outliers.mean(),
        "outliers_stdev": outliers.std(),
        "outliers_median": outliers.median(),
        "values_avg:": values.mean(),
        "values_stdev": values.std(),
        "values_median": values.median(),
    }
    
irq_score = pd.DataFrame.from_dict(irq_score, orient='index')

irq_score


# In[ ]:


from sklearn import preprocessing as pp

# scale all feature
cols = list(features)
for col in cols:
    features[col] = pp.scale(np.array(features[col]))


# # Clustering using K-Means

# now we are ready to apply the clustering algorithm, using KMeans
# from sklearn.cluster

# firstly use **'ELBOW'** method to find number of cluster

# In[ ]:


X = np.array(features)
sum_of_squared_distances = []
K = range(1,30)


# In[ ]:


from sklearn.cluster import KMeans
for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)


# In[ ]:


plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


#by observing above graph, k = 10


# In[ ]:


n_clusters = 10

clustering = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = clustering.fit_predict(X)


# In[ ]:


# plot cluster size

plt.hist(cluster_labels, bins=range(n_clusters+1))
plt.title('Customers per Customer')
plt.xlabel('Cluster')
plt.ylabel('Customers')
plt.show()


#assign cluster number to features and original dataframe

features['cluster_index'] = cluster_labels
df['cluster_index'] = cluster_labels


# In[ ]:


sns.pairplot(features, hue='cluster_index')


# In[ ]:


features


# In[ ]:


df #Results


# https://www.kaggle.com/mbranbilla/credit-card-clustering

# In[ ]:




