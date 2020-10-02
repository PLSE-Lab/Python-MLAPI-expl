#!/usr/bin/env python
# coding: utf-8

# In[48]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[49]:


from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import class_weight
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import pickle
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model
from keras import optimizers, regularizers, initializers
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[50]:


NCA1 = 100
NCA2 = 100
DROPRATE = 0.2
EP = 500
BATCH_SIZE = 128
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ## Loading FreeSolv dataset
# 
# **FreeSolv:** Experimental and calculated hydration free energy of small molecules in water.

# In[51]:


freesolv_df= pd.read_csv('../input/freesolv_descriptors/SAMPL_df_revised.csv')
print(freesolv_df.shape)
freesolv_df.head()


# In[52]:


freesolv_df.drop(['smiles','iupac','calc'], axis=1, inplace=True)
print(freesolv_df.shape)
freesolv_df.head()


# In[53]:


# Get indices of NaN
#inds = pd.isnull(tox21_df).any(1).nonzero()[0]


# In[54]:


# Drop NaN from the dataframe
#tox21_df.dropna(inplace=True)
#print(tox21_df.shape)
#tox21_df.head()


# In[55]:


freesolv_df = freesolv_df.fillna(0)
freesolv_df.head()


# The simplified molecular-input line-entry system (**SMILES**) is a specification in form of a line notation for describing the structure of chemical species using short ASCII strings. SMILES can be converted to molecular structure by using RDKIT module.
# 
# Example: 
# ```python
# from rdkit import Chem
# m = Chem.MolFromSmiles('Cc1ccccc1')
# ```
# 
# Further reading:
# * https://www.rdkit.org/docs/GettingStartedInPython.html

# ## Loading molecular descriptors
# 
# Descriptors dataframe contains 1625 molecular descriptors (including 3D descriptors) generated on the NCI database using Mordred python module.
# 
# Further Reading:
# * https://en.wikipedia.org/wiki/Molecular_descriptor
# * https://github.com/mordred-descriptor/mordred

# In[56]:


freesolv_descriptors_df= pd.read_csv('../input/freesolv_descriptors/SAMPL_descriptors_df.csv',low_memory=False)
print(freesolv_descriptors_df.shape)
freesolv_descriptors_df.head()


# In[57]:


# function to coerce all data types to numeric

def coerce_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')


# In[58]:


coerce_to_numeric(freesolv_descriptors_df, freesolv_descriptors_df.columns)
freesolv_descriptors_df.head()


# In[59]:


freesolv_descriptors_df = freesolv_descriptors_df.fillna(0)
freesolv_descriptors_df.head()


# In[60]:


#tox21_descriptors_df.drop(tox21_descriptors_df.index[inds],inplace=True)
#tox21_descriptors_df.shape


# ## Scaling and Principal component analysis (PCA) 

# In[61]:


freesolv_scaler1 = StandardScaler()
freesolv_scaler1.fit(freesolv_descriptors_df.values)
freesolv_scaler1 = pd.DataFrame(freesolv_scaler1.transform(freesolv_descriptors_df.values),
                                columns=freesolv_descriptors_df.columns)


# In[62]:


nca = NCA1
cn = ['col'+str(x) for x in range(nca)]


# In[63]:


freesolv_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
freesolv_transformer1.fit(freesolv_descriptors_df.values)
freesolv_descriptors_df = pd.DataFrame(freesolv_transformer1.transform(freesolv_descriptors_df.values),
                                       columns=cn)
print(freesolv_descriptors_df.shape)
freesolv_descriptors_df.head()


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(freesolv_descriptors_df.values,
                                                    freesolv_df.values, random_state=32,
                                                    test_size=TEST_RATIO)


# ## Sklearn SVR Model

# In[65]:


parameters = {'estimator__kernel':['rbf'], 
              'estimator__epsilon':[0.1,0.25,0.5],
              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}
freesolv_svr = GridSearchCV(MultiOutputRegressor(SVR()), 
                            parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[66]:


result = freesolv_svr.fit(X_train, y_train)


# In[67]:


pred = freesolv_svr.predict(X_test)
svr_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(svr_rmse)


# ## Keras Neural Network Model

# In[68]:


def huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)


# In[69]:


fresolv_model = Sequential()
fresolv_model.add(Dense(128, input_dim=freesolv_descriptors_df.shape[1], 
                        kernel_initializer='he_uniform'))
fresolv_model.add(BatchNormalization())
fresolv_model.add(Activation('tanh'))
fresolv_model.add(Dropout(rate=DROPRATE))
fresolv_model.add(Dense(64,kernel_initializer='he_uniform'))
fresolv_model.add(BatchNormalization())
fresolv_model.add(Activation('tanh'))
fresolv_model.add(Dropout(rate=DROPRATE))
fresolv_model.add(Dense(32,kernel_initializer='he_uniform'))
fresolv_model.add(BatchNormalization())
fresolv_model.add(Activation('tanh'))
fresolv_model.add(Dropout(rate=DROPRATE))
fresolv_model.add(Dense(freesolv_df.shape[1],kernel_initializer='he_uniform',activation=None))


# In[70]:


fresolv_model.compile(loss=huber_loss, optimizer='adam',metrics=['mse'])


# In[71]:


checkpoint = ModelCheckpoint('fresolv_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')


# In[72]:


hist = fresolv_model.fit(X_train, y_train, 
                         validation_split=VAL_RATIO,epochs=EP, batch_size=BATCH_SIZE, 
                         callbacks=[checkpoint])


# In[73]:


plt.ylim(1., 4.)
plt.plot(hist.epoch, hist.history["loss"], label="Train loss")
plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")


# In[74]:


fresolv_model.load_weights('fresolv_model.h5')


# In[75]:


pred = fresolv_model.predict(X_test)
nn_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(nn_rmse)


# ## Gradient Boosting of Keras Model with SVC

# In[76]:


inp = fresolv_model.input
out = fresolv_model.layers[-2].output
fresolv_model_gb = Model(inp, out)


# In[77]:


X_train = fresolv_model_gb.predict(X_train)
X_test = fresolv_model_gb.predict(X_test)


# In[78]:


data = np.concatenate((X_train,X_test),axis=0)


# In[79]:


freesolv_scaler2 = StandardScaler()
freesolv_scaler2.fit(data)
X_train = freesolv_scaler2.transform(X_train)
X_test = freesolv_scaler2.transform(X_test)


# In[80]:


data = np.concatenate((X_train,X_test),axis=0)


# In[81]:


nca = NCA2


# In[82]:


freesolv_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
freesolv_transformer2.fit(data)
X_train = freesolv_transformer2.transform(X_train)
X_test = freesolv_transformer2.transform(X_test)


# In[83]:


nca = X_train.shape[1]
parameters = {'estimator__kernel':['rbf'], 
              'estimator__epsilon':[0.1,0.25,0.5],
              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}
freesolv_svr_gb = GridSearchCV(MultiOutputRegressor(SVR()), 
                              parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[84]:


result = freesolv_svr_gb.fit(X_train, y_train)


# In[85]:


pred = freesolv_svr_gb.predict(X_test)
svr_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(svr_gb_rmse)


# ## Gradient Boosting of Keras Model with XGBoost

# In[86]:


parameters = {'estimator__learning_rate':[0.05,0.1,0.15],'estimator__n_estimators':[75,100,125], 'estimator__max_depth':[3,5,7],
              'estimator__booster':['gbtree','dart'],'estimator__reg_alpha':[0.1,0.05],'estimator__reg_lambda':[0.5,1.]}

freesolv_xgb_gb = GridSearchCV(MultiOutputRegressor(XGBRegressor(random_state=32)), 
                               parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[87]:


result = freesolv_xgb_gb.fit(X_train, y_train)


# In[88]:


pred = freesolv_xgb_gb.predict(X_test)
xgb_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(xgb_gb_rmse)


# ## Saving models, transformer and scaler

# In[89]:


with open('freesolv_svr.pkl', 'wb') as fid:
    pickle.dump(freesolv_svr, fid)
with open('freesolv_transformer1.pkl', 'wb') as fid:
    pickle.dump(freesolv_transformer1, fid)
with open('freesolv_transformer2.pkl', 'wb') as fid:
    pickle.dump(freesolv_transformer2, fid)
with open('freesolv_scaler1.pkl', 'wb') as fid:
    pickle.dump(freesolv_scaler1, fid)
with open('freesolv_scaler2.pkl', 'wb') as fid:
    pickle.dump(freesolv_scaler2, fid)
with open('freesolv_svr_gb.pkl', 'wb') as fid:
    pickle.dump(freesolv_svr_gb, fid)
with open('freesolv_xgb_gb.pkl', 'wb') as fid:
    pickle.dump(freesolv_xgb_gb, fid)


# ## For loading saved model
# 
# ```python
# with open('freesolv_svr.pkl', 'rb') as fid:
#     freesolv_svr = pickle.load(fid)
#  ```

# # Comparision of Results with MoleculeNet results
# 
# http://moleculenet.ai/full-results
# 
# The RMSE best score on the test data for the MoleculeNet models is ~1.5, the best score obtained in this kernel is ~4.0 on the test data. 
# 
# **Further Reading:**
# * https://arxiv.org/pdf/1703.00564.pdf

# ## RMSE

# In[90]:


sns.set(style="whitegrid")
ax = sns.barplot(x=[svr_rmse,nn_rmse,svr_gb_rmse,xgb_gb_rmse],
                 y=['SVR','NN','SVC_GB','XGB_GB'])
ax.set_xlim(1.0,5.0)

