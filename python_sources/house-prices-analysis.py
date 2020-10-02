#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import LeakyReLU, Dense, Conv1D, MaxPooling1D, Dropout, Permute, Flatten, LSTM
plt.style.use('ggplot')


# ## Opening training and test files to DataFrames and having a quick look at it's structure

# In[ ]:


df_train  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
y_train = df_train['SalePrice']
df_train = df_train.drop(columns=['SalePrice'])
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
display(df_train.head(10))


# As we can see, there is columns marked as 'object' (mostly strings), so, in order to further analyze the data, we need to deal with this datatype. For this, we will be using Pandas 'get_dummies', which will transform string into dummy integers.
# 
# Also, there is a lot of missing values, as a first analysis, we will be filling those values with the mean value of the column

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(df_train)
one_hot_encoded_test_predictors = pd.get_dummies(df_test)
df_train, df_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())


# Before inputing the data to a model, we should normalize it, to do so, we will be using StandardScaler, which will transform the data to have zero mean and unit variance

# In[ ]:


scaler = StandardScaler()
ids = df_test['Id']
df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)
y_train = np.log(y_train)


# Also, as we will be using neural networks as our model, we need to reshape the data

# In[ ]:


time_steps = 1
num_features = df_train.shape[-1]
df_train = np.reshape(df_train,(-1, time_steps, num_features))
df_test = np.reshape(df_test,(-1, time_steps, num_features))


# ## Creating the model
# 
# For the model, we will use a neural network with convolutional layers to extract relevant features

# In[ ]:


model = Sequential()

model.add(Permute((2, 1), input_shape=(time_steps, num_features)))
model.add(Conv1D(32, 2))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 2))
model.add(MaxPooling1D(2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(LeakyReLU())
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


model.fit(df_train, y_train, epochs=1000, batch_size=64, verbose=2)


# ## Preparing submition

# In[ ]:


predictions = pd.Series(np.exp(model.predict(df_test)).reshape((-1)), dtype='float64')
submission = pd.DataFrame({"Id": ids, "SalePrice": predictions})
submission.to_csv("submission.csv", index=False)

