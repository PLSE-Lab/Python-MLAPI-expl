#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


NCA1 = 100
NCA2 = 100
DROPRATE = 0.2
EP = 500
BATCH_SIZE = 256
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ## Loading Lipophilicity dataset
# 
# **Lipophilicity:** Experimental results of octanol/water distribution coefficient(logD at pH 7.4).

# In[ ]:


lipophilicity_df= pd.read_csv('../input/lipophilicity_descriptors/Lipophilicity_df_revised.csv')
print(lipophilicity_df.shape)
lipophilicity_df.head()


# In[ ]:


lipophilicity_df.drop(['smiles','CMPD_CHEMBLID'], axis=1, inplace=True)
print(lipophilicity_df.shape)
lipophilicity_df.head()


# In[ ]:


# Get indices of NaN
#inds = pd.isnull(tox21_df).any(1).nonzero()[0]


# In[ ]:


# Drop NaN from the dataframe
#tox21_df.dropna(inplace=True)
#print(tox21_df.shape)
#tox21_df.head()


# In[ ]:


lipophilicity_df = lipophilicity_df.fillna(0)
lipophilicity_df.head()


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

# In[ ]:


lipophilicity_descriptors_df= pd.read_csv('../input/lipophilicity_descriptors/Lipophilicity_descriptors_df.csv',low_memory=False)
print(lipophilicity_descriptors_df.shape)
lipophilicity_descriptors_df.head()


# In[ ]:


# function to coerce all data types to numeric

def coerce_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')


# In[ ]:


coerce_to_numeric(lipophilicity_descriptors_df, lipophilicity_descriptors_df.columns)
lipophilicity_descriptors_df.head()


# In[ ]:


lipophilicity_descriptors_df = lipophilicity_descriptors_df.fillna(0)
lipophilicity_descriptors_df.head()


# In[ ]:


#tox21_descriptors_df.drop(tox21_descriptors_df.index[inds],inplace=True)
#tox21_descriptors_df.shape


# ## Scaling and Principal component analysis (PCA) 

# In[ ]:


lipophilicity_scaler1 = StandardScaler()
lipophilicity_scaler1.fit(lipophilicity_descriptors_df.values)
lipophilicity_descriptors_df = pd.DataFrame(lipophilicity_scaler1.transform(lipophilicity_descriptors_df.values),
                                            columns=lipophilicity_descriptors_df.columns)


# In[ ]:


nca = NCA1
cn = ['col'+str(x) for x in range(nca)]


# In[ ]:


lipophilicity_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
lipophilicity_transformer1.fit(lipophilicity_descriptors_df.values)
lipophilicity_descriptors_df = pd.DataFrame(lipophilicity_transformer1.transform(lipophilicity_descriptors_df.values),
                                            columns=cn)
print(lipophilicity_descriptors_df.shape)
lipophilicity_descriptors_df.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(lipophilicity_descriptors_df.values,
                                                    lipophilicity_df.values, random_state=32,
                                                    test_size=TEST_RATIO)


# ## Sklearn SVR Model

# In[ ]:


parameters = {'estimator__kernel':['rbf'], 
              'estimator__epsilon':[0.1,0.25,0.5],
              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}
lipophilicity_svr = GridSearchCV(MultiOutputRegressor(SVR()), 
                        parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[ ]:


result = lipophilicity_svr.fit(X_train, y_train)


# In[ ]:


pred = lipophilicity_svr.predict(X_test)
svr_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(svr_rmse)


# ## Keras Neural Network Model

# In[ ]:


def huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)


# In[ ]:


lipophilicity_model = Sequential()
lipophilicity_model.add(Dense(128, input_dim=lipophilicity_descriptors_df.shape[1], 
                              kernel_initializer='he_uniform'))
lipophilicity_model.add(BatchNormalization())
lipophilicity_model.add(Activation('tanh'))
lipophilicity_model.add(Dropout(rate=DROPRATE))
lipophilicity_model.add(Dense(64,kernel_initializer='he_uniform'))
lipophilicity_model.add(BatchNormalization())
lipophilicity_model.add(Activation('tanh'))
lipophilicity_model.add(Dropout(rate=DROPRATE))
lipophilicity_model.add(Dense(32,kernel_initializer='he_uniform'))
lipophilicity_model.add(BatchNormalization())
lipophilicity_model.add(Activation('tanh'))
lipophilicity_model.add(Dropout(rate=DROPRATE))
lipophilicity_model.add(Dense(lipophilicity_df.shape[1],kernel_initializer='he_uniform',activation=None))


# In[ ]:


lipophilicity_model.compile(loss=huber_loss, optimizer='adam',metrics=['mse'])


# In[ ]:


checkpoint = ModelCheckpoint('lipophilicity_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')


# In[ ]:


hist = lipophilicity_model.fit(X_train, y_train, 
                               validation_split=VAL_RATIO,epochs=EP, batch_size=BATCH_SIZE, 
                               callbacks=[checkpoint])


# In[ ]:


plt.ylim(0., 1.)
plt.plot(hist.epoch, hist.history["loss"], label="Train loss")
plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")


# In[ ]:


lipophilicity_model.load_weights('lipophilicity_model.h5')


# In[ ]:


pred = lipophilicity_model.predict(X_test)
nn_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(nn_rmse)


# ## Gradient Boosting of Keras Model with SVC

# In[ ]:


inp = lipophilicity_model.input
out = lipophilicity_model.layers[-2].output
lipophilicity_model_gb = Model(inp, out)


# In[ ]:


X_train = lipophilicity_model_gb.predict(X_train)
X_test = lipophilicity_model_gb.predict(X_test)


# In[ ]:


data = np.concatenate((X_train,X_test),axis=0)


# In[ ]:


lipophilicity_scaler2 = StandardScaler()
lipophilicity_scaler2.fit(data)
X_train = lipophilicity_scaler2.transform(X_train)
X_test = lipophilicity_scaler2.transform(X_test)


# In[ ]:


data = np.concatenate((X_train,X_test),axis=0)


# In[ ]:


nca = NCA2


# In[ ]:


lipophilicity_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
lipophilicity_transformer2.fit(data)
X_train = lipophilicity_transformer2.transform(X_train)
X_test = lipophilicity_transformer2.transform(X_test)


# In[ ]:


nca = X_train.shape[1]
parameters = {'estimator__kernel':['rbf'], 
              'estimator__epsilon':[0.1,0.25,0.5],
              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}
lipophilicity_svr_gb = GridSearchCV(MultiOutputRegressor(SVR()), 
                                    parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[ ]:


result = lipophilicity_svr_gb.fit(X_train, y_train)


# In[ ]:


pred = lipophilicity_svr_gb.predict(X_test)
svr_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(svr_gb_rmse)


# ## Gradient Boosting of Keras Model with XGBoost

# In[ ]:


parameters = {'estimator__learning_rate':[0.05,0.1,0.15],'estimator__n_estimators':[75,100,125], 'estimator__max_depth':[3,5,7],
              'estimator__booster':['gbtree','dart'],'estimator__reg_alpha':[0.1,0.05],'estimator__reg_lambda':[0.5,1.]}

lipophilicity_xgb_gb = GridSearchCV(MultiOutputRegressor(XGBRegressor(random_state=32)), 
                                    parameters, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)


# In[ ]:


result = lipophilicity_xgb_gb.fit(X_train, y_train)


# In[ ]:


pred = lipophilicity_xgb_gb.predict(X_test)
xgb_gb_rmse = np.sqrt(mean_squared_error(y_test,pred))
print(xgb_gb_rmse)


# ## Saving models, transformer and scaler

# In[ ]:


with open('lipophilicity_svr.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_svr, fid)
with open('lipophilicity_transformer1.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_transformer1, fid)
with open('lipophilicity_transformer2.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_transformer2, fid)
with open('lipophilicity_scaler1.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_scaler1, fid)
with open('lipophilicity_scaler2.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_scaler2, fid)
with open('lipophilicity_svr_gb.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_svr_gb, fid)
with open('lipophilicity_xgb_gb.pkl', 'wb') as fid:
    pickle.dump(lipophilicity_xgb_gb, fid)


# ## For loading saved model
# 
# ```python
# with open('lipophilicity_svr.pkl', 'rb') as fid:
#     lipophilicity_svr = pickle.load(fid)
#  ```

# # Comparision of Results with MoleculeNet results
# 
# http://moleculenet.ai/full-results
# 
# The RMSE best score on the test data for the MoleculeNet models is ~0.7, the best score obtained in this kernel is ~0.72 on the test data. 
# 
# **Further Reading:**
# * https://arxiv.org/pdf/1703.00564.pdf

# ## RMSE

# In[ ]:


sns.set(style="whitegrid")
ax = sns.barplot(x=[svr_rmse,nn_rmse,svr_gb_rmse,xgb_gb_rmse],
                 y=['SVR','NN','SVC_GB','XGB_GB'])
ax.set_xlim(0.65,0.8)

