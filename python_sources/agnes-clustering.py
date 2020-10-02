#!/usr/bin/env python
# coding: utf-8

# Load Libraries

# In[ ]:


from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# Load Data

# In[ ]:


iris = datasets.load_iris()
features = iris.data


# Standardize features

# In[ ]:


scaler = StandardScaler()
features_std = scaler.fit_transform(features)


# Create meanshift object

# In[ ]:


cluster = AgglomerativeClustering(n_clusters = 3)


# Train model

# In[ ]:


model = cluster.fit(features_std)


# Show cluster membership

# In[ ]:


model.labels_


# In[ ]:


pred = 0

for i in range(150):
        if i<50:
            if model.labels_[i]==1:
                pred +=1
        elif i<100:
            if model.labels_[i]==2:
                pred +=1
        elif i<150:
            if model.labels_[i]==0:
                pred +=1
        
accuracy = (pred/150)*100
print(accuracy)


# In[ ]:




