#!/usr/bin/env python
# coding: utf-8

# ### Site 1 vs Site 2, Unsupervised Learning

# In this notebook I will use unsupervised learning to try to deterine which entries in both test and train are site 2. It has been shown in other kernels there is signficant difference between the test and train sets. 

# "Models are expected to generalize on data from a different scanner/site (site 2). All subjects from site 2 were assigned to the test set, so their scores are not available. While there are fewer site 2 subjects than site 1 subjects in the test set, the total number of subjects from site 2 will not be revealed until after the end of the competition. To make it more interesting, the IDs of some site 2 subjects have been revealed below. Use this to inform your models about site effects. Site effects are a form of bias. To generalize well, models should learn features that are not related to or driven by site effects."

# Train Data -> Sit
# Test Data -> We know 510 Site 2 entries, rest unknown
# 

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter


# In[ ]:


fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv')
sites = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')


# In[ ]:


df = loading


# In[ ]:


# fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
# df = fnc.merge(loading, on="Id")


# In[ ]:


sites = np.array(sites).reshape(sites.shape[0])


# We know that all of site 2 is in the test set. So there should be 0 site 1 in our train data. It remains a mystery how many site is in the test set but we know that site 2 < site 1. 

# In[ ]:


def get_test_train(df):
    labels = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
    labels["is_train"] = True
    df = df.merge(labels, on="Id", how="left")

    
    test_df = df[df["is_train"] != True].copy()
    df = df[df["is_train"] == True].copy()
    df = df.drop(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2', 'is_train'], axis=1)
    test_df = test_df.drop(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2', 'is_train'], axis=1)
    return df, test_df


# In[ ]:


train_df, test_df = get_test_train(df)


# In[ ]:


site_unknown = test_df[~test_df['Id'].isin(set(sites))]
site2 = test_df[test_df['Id'].isin(set(sites))]


# Concat-ing site unknown and site 2 will give us back our test set. Also we know that the train is all site 1. 

# In[ ]:


site_unknown.shape


# In[ ]:


site2.shape


# We need to make a dataset where we know which ones are site 1 and site 2

# In[ ]:


site_1_2 = pd.concat([train_df, site2], axis=0)
site_1_2.head()


# ### KMeans, setting two clusters

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0).fit(site_1_2)
kmeans.labels_


# In[ ]:


from collections import Counter
Counter(kmeans.labels_)


# In[ ]:


site2_preds = kmeans.predict(site2)


# In[ ]:


Counter(site2_preds)


# This split shows about the same predctions in one cluster. Ideally the closer the number of cluster predictions are closer to the site 2 entries, the better we can deem our unsupervised learning techniques for determining the sites. 

# In[ ]:


site_unknown_preds = kmeans.predict(site_unknown)


# In[ ]:


Counter(site_unknown_preds)


# ### Isolation Forest

# The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples. <-- From Sklearn

# In[ ]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=(510/(5877+510)),random_state=0).fit(site_1_2)


# In[ ]:


site_unknown_preds = clf.predict(site_unknown)
Counter(site_unknown_preds)


# The competition states there are fewer site 2 than site 1. And all site 2 is in the test set. Here we see that if we assume -1 (outliers) from isolation forest was site 2,then we can see that  there might e 428 site 2 in the unknown data. 

# In[ ]:


site_unknown_preds = clf.predict(test_df)
Counter(site_unknown_preds)


# Here we se that for the test set, there are 476 outliers. We know that atleast 510 of the test set should be site 2. 

# In[ ]:




