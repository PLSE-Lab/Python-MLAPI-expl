#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


# In[2]:


data = pd.read_csv("../input/heart.csv")


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


data['cp'].value_counts()


# In[10]:


data['thal'].value_counts()


# In[11]:


data = pd.get_dummies(data=data, columns=['cp', 'thal'])


# In[12]:


data.head()


# In[13]:


x = data.drop(['target'], axis=1)
y = data['target']


# In[14]:


scaler = MinMaxScaler(feature_range=[0,1])
x = scaler.fit_transform(x)


# In[17]:


sse = {}
data = pd.DataFrame()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)
    data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[19]:


from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split


# In[21]:


x_train, x_test = train_test_split(x, test_size=0.1)


# In[22]:


x_train.shape, x_test.shape


# In[44]:


input_data = Input(shape=(19,))
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

encoded = Dense(2, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(19, activation='sigmoid')(decoded)


# In[45]:


autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[46]:


encoder = Model(input_data, encoded)


# In[47]:


reduced_x_train = encoder.predict(x_train)
reduced_x_test = encoder.predict(x_test)


# In[48]:


reduced_x_train.shape, reduced_x_test.shape


# In[49]:


dummy = pd.DataFrame()

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(reduced_x_train)
    dummy["auto_clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[ ]:





# In[ ]:





# In[51]:


#Using softmax to form clusters

input_data = Input(shape=(19,))
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

encoded = Dense(2, activation='softmax')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(19, activation='sigmoid')(decoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[52]:


encoder = Model(input_data, encoded)
reduced_x_train = encoder.predict(x_train)
reduced_x_test = encoder.predict(x_test)


# In[56]:


predict_clusters = np.argmax(reduced_x_test, axis=1)


# In[57]:


predict_clusters


# In[58]:


reduced_x_test


# In[ ]:




