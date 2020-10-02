#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 16,9
from tqdm import tqdm


# In[ ]:


K = 1


# # Read Test Data

# In[ ]:


df_test = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime'], index_col='key')
df_test.info()


# # Find k-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KDTree, NearestNeighbors


# In[ ]:


df_test['t'] = df_test['pickup_datetime'].astype('int64') // 10**9


# In[ ]:


features = ['t', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']


# In[ ]:


features_mean, features_std = df_test[features].describe().loc[['mean', 'std']].values


# In[ ]:


X_test = (df_test[features].values - features_mean) / features_std


# In[ ]:


test_neighbors = [[] for _ in range(len(df_test))]


# In[ ]:


for df_chunk in tqdm(pd.read_csv('../input/train.csv', index_col='key', parse_dates=['pickup_datetime'], chunksize=50000)):
    df_chunk['t'] = df_chunk['pickup_datetime'].astype('int64') // 10**9
    
    X_chunk = (df_chunk[features].values - features_mean) / features_std
    
    kd = KDTree(X_chunk, leaf_size=50)
    
    D, I = kd.query(X_test, k=K)
    Y = df_chunk['fare_amount'].values[I]

    for i in range(len(test_neighbors)):
        test_neighbors[i] += list(zip(D[i], Y[i]))
        test_neighbors[i] = sorted(test_neighbors[i], key=lambda x: x[0])[:K]


# 
# # Predict and Submit

# In[ ]:


df_sub = pd.DataFrame({
    'key': df_test.index,
    'fare_amount': np.array([[x[1] for x in neighbors] for neighbors in test_neighbors]).mean(axis=1)
}).set_index('key')


# In[ ]:


df_sub.head()


# In[ ]:


sns.distplot(df_sub['fare_amount'])


# In[ ]:


df_sub.to_csv('submission.csv')

