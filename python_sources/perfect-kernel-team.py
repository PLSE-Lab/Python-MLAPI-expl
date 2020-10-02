#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ### Read data

# In[20]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[21]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# ### Select columns with sensor data

# In[22]:


columns_in = [f"sensor#{i}" for i in range(12)]
columns_out = "oil_extraction"

train_x = train_df[columns_in].values
train_y = train_df[columns_out].values
test_x = test_df[columns_in].values
print(train_x.shape[0])
print(test_x.shape[0])


# ### *Visualize distribution for train and test*

# In[23]:


plt.figure(figsize=(16, 6))
train_df.boxplot(columns_in)
plt.figure(figsize=(16, 6))
test_df.boxplot(columns_in)


# In[24]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16, 6))
corrs = train_df.loc[:,'timestamp':'sensor#11'].corr()
# plot the heatmap and annotation on it
sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, annot=True)

plt.figure(figsize=(16, 6))
corrs = test_df.loc[:,'timestamp':'sensor#11'].corr()
# plot the heatmap and annotation on it
sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, annot=True)


# Obvious to drop sensor#8 or sensor#4 data

# In[25]:


train_df.drop(columns=['sensor#8'], inplace=True)
test_df.drop(columns=['sensor#8'], inplace=True)


# Use PCA to get rid of correlations

# In[26]:


from mpl_toolkits.mplot3d import Axes3D
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from matplotlib import cm

train_df_copy = pd.DataFrame(train_df)
test_df_copy = pd.DataFrame(test_df)
print(train_df_copy.shape[0])
print(test_df_copy.shape[0])


# In[27]:


train_df_copy = train_df_copy.loc[:,'sensor#0':'sensor#11']
test_df_copy = test_df_copy.loc[:,'sensor#0':'sensor#11']

train_df_copy = pd.DataFrame(train_df_copy)
print(train_df_copy.shape[0])
test_df_copy = pd.DataFrame(test_df_copy)
print(test_df_copy.shape[0])
train_copy = pd.concat([train_df_copy, test_df_copy])
plt.figure(figsize=(16, 6))
train_copy.boxplot()
#train_copy = StandardScaler().fit_transform(train_copy)


# In[28]:


pca = PCA(n_components = 3, random_state=1)

X_pca = pca.fit_transform(train_copy)
principalDf = pd.DataFrame(data = X_pca, columns =['principal1', 'principal2', 'principal3'])
print('Varience preserved : ' + str(pca.explained_variance_ratio_.cumsum()[1]))


# In[29]:


principalDf.describe()


# In[30]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.graph_objs as go
from sklearn import preprocessing

principalDf = principalDf.replace([np.inf, -np.inf], np.nan)
principalDf = principalDf.fillna(0)

train_pca = pd.DataFrame(principalDf[0:86389])
test_pca = pd.DataFrame(principalDf[86389:95030])

train_log, test_log = np.log10(train_pca), np.log10(test_pca)
train_pca = pd.DataFrame(train_pca).round(3)
test_pca = pd.DataFrame(test_pca).round(3)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train_pca)
train_pca = pd.DataFrame(np_scaled)

np_scaled = min_max_scaler.fit_transform(test_pca)
test_pca = pd.DataFrame(np_scaled)

figsize=(12, 7)
plt.figure(figsize=figsize)
pyplot.plot(train_pca[0],label='Actuals')

figsize=(12, 7)
plt.figure(figsize=figsize)
pyplot.plot(test_pca[0],label='Actuals')


# In[31]:


get_ipython().system('pip install pyramid-arima')


# In[32]:


import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from pyramid.arima import auto_arima

my_order = (1, 1, 1)
my_seasonal_order = (1, 1, 1, 1)

X = train_pca.values
print(X)


# In[33]:


train_pca = np.array(train_pca)
test_pca = np.array(test_pca)
train_pca = train_pca.reshape(train_pca.shape[0], 1,-1)
test_pca = test_pca.reshape(test_pca.shape[0], 1, -1)


# In[34]:


from keras.layers import Input, SpatialDropout1D, CuDNNLSTM, Dropout, Dense, LSTM 
from keras.models import Model
from keras.models import Sequential

import tensorflow as tf
tf.set_random_seed(56)
print(train_pca.shape)

np.random.seed(56)
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(1, train_pca.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()
model_lstm.fit(train_pca, train_df['oil_extraction'], batch_size=32, epochs=10, verbose=True, validation_split=0.01, shuffle=True)


# In[36]:


def predictions_to_submission_file(predictions):
    submission_df = pd.DataFrame(columns=['Expected', 'Id'])
    submission_df['Expected'] = predictions
    submission_df['Id'] = range(len(predictions))
    submission_df.to_csv('submission.csv', index=False)

test_predictions = model_lstm.predict(test_pca)
predictions_to_submission_file(test_predictions[:,0])

