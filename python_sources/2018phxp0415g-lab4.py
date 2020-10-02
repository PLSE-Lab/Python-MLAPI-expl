#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load train and Test set
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")


# In[ ]:


#train.fillna(train.mean(), inplace=True)


# In[ ]:


#test.fillna(test.mean(), inplace=True)


# In[ ]:


#print("*****In the train set*****")
#print(train.isna().sum())
#print("\n")
#print("*****In the test set*****")
#print(test.isna().sum())


# In[ ]:


#train.info


# In[ ]:


#train = train.drop(['diag_1','diag_2', 'diag_3'], axis=1)
#test = test.drop(['diag_1','diag_2', 'diag_3'], axis=1)

train = train.drop(['medical_specialty'], axis=1)
test = test.drop(['medical_specialty'], axis=1)

train = train.drop(['max_glu_serum' ,'citoglipton','glimepiride','examide'], axis=1)
test = test.drop(['max_glu_serum','citoglipton','glimepiride','examide'], axis=1)

train = train.drop(['metformin-rosiglitazone' ,'acarbose','tolbutamide','miglitol','chlorpropamide'], axis=1)
test = test.drop(['metformin-rosiglitazone','acarbose','tolbutamide','miglitol','chlorpropamide'], axis=1)

train = train.drop(['nateglinide' ,'acetohexamide','tolazamide','troglitazone','metformin-pioglitazone'], axis=1)
test = test.drop(['nateglinide','acetohexamide','tolazamide','troglitazone','metformin-pioglitazone'], axis=1)

train = train.drop(['number_outpatient' ,'number_emergency','number_inpatient','payer_code','glipizide-metformin'], axis=1)
test = test.drop(['number_outpatient','number_emergency','number_inpatient','payer_code','glipizide-metformin'], axis=1)


# In[ ]:





# In[ ]:


from sklearn.preprocessing import LabelEncoder
# LabelEncoder
le = LabelEncoder()
train_encoded = train.apply(le.fit_transform)
test_encoded = test.apply(le.fit_transform)


# In[ ]:


train_input = np.array(train_encoded.drop(['readmitted_NO'], 1).astype(float))
train_output = np.array(train_encoded['readmitted_NO'])

test_input = np.array(test_encoded.drop(['index'], 1).astype(float))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size = 0.1, random_state=0)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train_input)
X1_scaled = scaler.fit_transform(test_input)


# In[ ]:


kmeans = KMeans(n_clusters=2, max_iter=400)
kmeans.fit(train_input)


# In[ ]:


correct = 0
for i in range(len(train_input)):
    predict_me = np.array(train_input[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == train_output[i]:
        correct += 1

print(correct/len(train_input))


# In[ ]:





# In[ ]:


predictions = kmeans.predict(X1_scaled)

test_output = pd.DataFrame({ 'index' : test['index'], 'target': predictions })
test_output.to_csv('lab4.csv', index = False)
test_output.head()


# In[ ]:


predictions = kmeans.predict(test_input)

test_output = pd.DataFrame({ 'index' : test['index'], 'target': predictions })
test_output.to_csv('lab4.csv', index = False)
test_output.head()


# In[ ]:


from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=2,init_params='kmeans')
model.fit(train_input)


# In[ ]:


correct = 0
for i in range(len(train_input)):
    predict_me = np.array(train_input[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = model.predict(predict_me)
    if prediction[0] == train_output[i]:
        correct += 1

print(correct/len(train_input))


# In[ ]:


from sklearn.cluster import DBSCAN 

db = DBSCAN(eps=11, min_samples=2)
db.fit(train_input)


# In[ ]:


correct = 0
for i in range(len(train_input)):
    predict_me = np.array(train_input[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = db.fit_predict(predict_me)
    if prediction[0] == train_output[i]:
        correct += 1

print(correct/len(train_input))


# In[ ]:


model = AffinityPropagation(damping = 0.5, max_iter = 40, affinity = 'euclidean')
model.fit(train_input)


# In[ ]:


from sklearn.cluster import SpectralClustering
sc = SpectralClustering(3, affinity='precomputed', n_init=100,assign_labels='discretize')
sc.fit(train_input)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(train_input)

