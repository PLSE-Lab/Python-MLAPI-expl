#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")


# In[ ]:


pd.set_option('display.max_columns', 500)


# In[ ]:


cols_to_drop = ['max_glu_serum', 'A1Cresult', 'weight', 'medical_specialty', 'payer_code']
train = train.drop(cols_to_drop, axis=1)


# In[ ]:


temps = {}


# In[ ]:


for col in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'diabetesMed', 'race', 'gender', 'age', 'diabetesMed']:
    temps[col] = pd.DataFrame({
        'data': train[col].unique(), 
        'data_new':range(len(train[col].unique()))
    })

    for index, row in temps[col].iterrows():
        train = train.replace(row['data'], row['data_new'])


# In[ ]:


train = train.replace('?', 0)


# In[ ]:


for cols in train:
    print(train[col].dtype)


# In[ ]:


train.head()


# In[ ]:


train = train.drop(["diag_3", "diag_1", "diag_2"], axis=1)


# In[ ]:


X = np.array(train.drop(['readmitted_NO'], 1))


# In[ ]:


y = np.array(train['readmitted_NO'])


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


X


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


test = test.drop(cols_to_drop, axis=1)


# In[ ]:


for col in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'diabetesMed', 'race', 'gender', 'age', 'diabetesMed']:

    for index, row in temps[col].iterrows():
        test = test.replace(row['data'], row['data_new'])


# In[ ]:


test = test.drop(["diag_3", "diag_1", "diag_2", "index"], axis=1)


# In[ ]:


test = test.replace('?', 0)


# In[ ]:


test = test.replace('Yes', 0)


# In[ ]:


X_ = np.array(test)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled_ = scaler.fit_transform(X_)


# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


clustering = DBSCAN(eps=3, min_samples=2).fit(X, y)


# In[ ]:


u = clustering.fit_predict(X_)


# In[ ]:


u


# In[ ]:


np.unique(u)


# In[ ]:


u = clustering.predict(X_)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering


# In[ ]:


clustering = AgglomerativeClustering(linkage="complete", affinity="l2").fit(X_scaled, y)


# In[ ]:


u = clustering.predict(X_scaled_)


# In[ ]:


new_df = pd.DataFrame({"index": [i for i in range(len(u))], "target": u})


# In[ ]:


new_df.to_csv("solution2.csv", index=False)

