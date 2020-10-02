#!/usr/bin/env python
# coding: utf-8

# # Photometric redshift estimation

# In my kernel I've tried different models and implementations for giving the best estimate to redshift. 
# 
# I run everything using google colab, so there at the beginning of the notebook you can see the way how to load csv data from your Google Drive and so run your notebook from any computer.
# 
# Later on I looked at the data, and processed it according to the models and implementations. 
# 
# I spent most of my time trying to find the best neural network to estimate redshift. I used sklearn MLP and Keras to build various deep learning models. Later on I turned to other models like random forest, which gave my best estimate in the competition. You can see all of the code for these models in this notebook. (Every model is specified in the way I found the best during my trial and error method, so if you download the data you can get the csv-s I uploaded. Hopefully.)

# packages

# In[ ]:


# packages for data processing and manipulation

get_ipython().system('pip install PyDrive # for loading csv from Drive')

from google.colab import files
import io

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# for loading csv from Drive 1. according to https://medium.freecodecamp.org/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[ ]:


# for loading csv from Drive 2.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[ ]:


# packages for estimation

from sklearn.preprocessing import normalize

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

from sklearn.ensemble import RandomForestRegressor


# data processing

# In[ ]:


download = drive.CreateFile({'id': '1c_8x-EIqzAyBJz8e29D4Xnhesqsz7zFV'}) # ID is the name of the file you are using. in my case it was automatically renamed.
download.GetContentFile('train.csv')


# In[ ]:


download = drive.CreateFile({'id': '1bDQIR7nt9KjLrQAdHY7B3By17H8FxR0j'})
download.GetContentFile('test.csv')


# In[ ]:


#uploaded = files.upload()


# In[ ]:


# loading training set
data_train = pd.read_csv("train.csv")
data_train.set_index("id", inplace = True)


# In[ ]:


# loading test set
data_test = pd.read_csv("test.csv")
data_test.set_index("id", inplace = True)


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_train.info()


# In[ ]:


# looking for missing values
plt.figure(figsize=(15,8))
sns.heatmap(data_train.isnull(), cbar=False)


# In[ ]:


# Splitting datasets to independent and dependent variables

indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]
dep_label = ["redshift"]

data_train_X = data_train[indep_labels]
data_train_Y = data_train[dep_label]

data_test_X = data_test[indep_labels]
#data_test_Y = data_test[dep_label] # because there is no redshift label in the test dataset, there is no data_test_Y given, we have to give an estimate for it.


# In[ ]:





# Estimations

#     -- Neural network (multi-layer perceptron regressor - MLP) with sklearn

# In[ ]:


# Datasets
data_train_X = data_train[indep_labels]
data_train_Y = data_train[dep_label]

# Train/test split
train_X, valid_X, train_Y, valid_Y = train_test_split(data_train_X, data_train_Y, train_size=0.8, random_state=0)


# In[ ]:


import time
start = time.time()

# Multi-layer perceptron
MLP_reg = MLPRegressor(hidden_layer_sizes=(200,100),
                       activation="relu",
                       solver="adam",
                       alpha=0.0001, batch_size="auto",
                       learning_rate="constant", learning_rate_init=0.001,
                       power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=True,
                       warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=True, validation_fraction=0.1,
                       beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

MLP_reg_fit = MLP_reg.fit(data_train_X, data_train_Y)

valid_y_pred_MLP = MLP_reg_fit.predict(data_test_X)

MSE_valid_MLP = mean_squared_error(valid_y_pred_MLP, valid_Y)

end = time.time()
print(end - start)


# In[ ]:


MSE_valid_MLP


# In[ ]:


y_pred = MLP_reg_fit.predict(data_test_X)


# In[ ]:


y_pred


#     saving results

# In[ ]:


predtosave = pd.DataFrame(y_pred, columns=["redshift"])


# In[ ]:


predtosave.to_csv('katona_sandor_predtoredshift.csv', index_label="id", header=True )


# In[ ]:


files.download('katona_sandor_predtoredshift.csv')


# In[ ]:





#     -- Neural network (multi-layer models) with Keras

# In[ ]:


# Datasets

# Splitting datasets to independent and dependent variables

indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]
dep_label = ["redshift"]

data_train_X = data_train[indep_labels]
data_train_Y = data_train[dep_label]

data_test_X = data_test[indep_labels]


# In[ ]:


# Normalisation of the data_train_X

data_train_Xn = (data_train_X - data_train_X.mean()) / data_train_X.std() # not used in the model down


# In[ ]:


# Creating the model

model = Sequential()

# number of variables in training data
n_cols = data_train_Xn.shape[1]

# adding model layers
model.add(Dense(200, input_shape=(n_cols,)))
model.add(Activation('relu'))

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(1))


# In[ ]:


# Compiling the model

model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


# Training the model with the possibility of early stopping (MSE is shown here!)

import time
start = time.time()

early_stopping_monitor = EarlyStopping(patience=3)

model.fit(data_train_X, data_train_Y, validation_split=0.2, batch_size=200, epochs=15, callbacks=[early_stopping_monitor])


end = time.time()
print(end - start)


# In[ ]:


y_pred_keras = model.predict(data_test_X)


# In[ ]:


y_pred_keras


#     saving results

# In[ ]:


predtosave_keras = pd.DataFrame(y_pred_keras, columns=["redshift"])


# In[ ]:


predtosave_keras.to_csv('katona_sandor_predtoredshift_keras.csv', index_label="id", header=True )


# In[ ]:


files.download('katona_sandor_predtoredshift_keras.csv')


# In[ ]:





#     -- Random forest regressor

# In[ ]:


# Datasets

# Splitting datasets to independent and dependent variables

indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]
dep_label = ["redshift"]

data_train_X = data_train[indep_labels]
data_train_Y = data_train[dep_label]

# Train/test split
train_X, valid_X, train_Y, valid_Y = train_test_split(data_train_X, data_train_Y, train_size=0.8, random_state=0)


# In[ ]:


import time
start = time.time()

# Random forest regression
rf = RandomForestRegressor(n_estimators=190, criterion="mse",
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                           max_features="auto", max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           bootstrap=True, oob_score=False, n_jobs=-1,
                           random_state=None, verbose=0, warm_start=False)

rf_fit = rf.fit(train_X, train_Y)

valid_y_pred_rf = rf_fit.predict(valid_X)

MSE_valid_rf = mean_squared_error(valid_y_pred_rf, valid_Y)

end = time.time()
print(end - start)


# In[ ]:


MSE_valid_rf


# In[ ]:


y_pred_rf = rf_fit.predict(data_test_X)


# In[ ]:


y_pred_rf


#     saving results

# In[ ]:


predtosave_rf = pd.DataFrame(y_pred_rf, columns=["redshift"])


# In[ ]:


predtosave_rf.to_csv('katona_sandor_predtoredshift.csv', index_label="id", header=True )


# In[ ]:


files.download('katona_sandor_predtoredshift.csv')

