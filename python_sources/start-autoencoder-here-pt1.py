#!/usr/bin/env python
# coding: utf-8

# # AutoEncoders for Dimensionality Reduction

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.datasets import make_blobs


# In[ ]:


data = make_blobs(n_samples=300,
    n_features=2,
    centers=2,
    cluster_std=1.0,random_state=101)


# In[ ]:


X,y = data


# In[ ]:


np.random.seed(seed=101)
z_noise = np.random.normal(size=len(X))
z_noise = pd.Series(z_noise)


# In[ ]:


feat = pd.DataFrame(X)
feat = pd.concat([feat,z_noise],axis=1)
feat.columns = ['X1','X2','X3']


# In[ ]:


feat.head()


# In[ ]:


plt.scatter(feat['X1'],feat['X2'],c=y)


# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#scatter-plots

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


#%matplotlib notebook  --> Try this command for an interactive 3D Visualization


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feat['X1'],feat['X2'],feat['X3'],c=y)


# # Encoder and Decoder

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


# In[ ]:


# 3 --> 2
encoder = Sequential()
encoder.add(Dense(units=2,activation='relu',input_shape=[3]))


# In[ ]:


# 2 ---> 3
decoder = Sequential()
decoder.add(Dense(units=3,activation='relu',input_shape=[2]))


# In[ ]:


# ENCODER
# 3 ---> 2 ----> 3
autoencoder = Sequential([encoder,decoder])
autoencoder.compile(loss="mse" ,optimizer=SGD(lr=1.5))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Note how all the data is used! There is no "right" answer here
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(feat)


# In[ ]:


# scaled_data


# In[ ]:


autoencoder.fit(scaled_data,scaled_data,epochs=5)


# In[ ]:


encoded_2dim = encoder.predict(scaled_data)


# In[ ]:


encoded_2dim


# In[ ]:


plt.scatter(encoded_2dim[:,0],encoded_2dim[:,1],c=y)


# ### This was a simple example of reducing dimensions from 3 to 2. In part 2, I will use MNIST data with added noise to reduce both dimensions and noise.

# # Thanks
