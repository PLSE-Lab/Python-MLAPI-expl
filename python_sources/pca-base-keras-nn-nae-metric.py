#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import scipy.io
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import nilearn as nl
import nilearn.plotting as nlplt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, BatchNormalization, PReLU
import tensorflow.keras.backend as K
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA, TruncatedSVD
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_scores = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
reveal_ID_site2 = pd.read_csv("/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv")
fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")
loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")
ICN_numbers = pd.read_csv("/kaggle/input/trends-assessment-prediction/ICN_numbers.csv")
sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")


# In[ ]:


train_scores.head()


# In[ ]:


fnc.describe()


# In[ ]:


train_scores.describe()


# In[ ]:


data = pd.concat([loading, fnc.drop(['Id'], axis=1)], axis=1)


# In[ ]:


train = pd.DataFrame(train_scores.Id).merge(data, on='Id')


# In[ ]:


test_id=[]
for i in range(0,len(sample_submission.Id),5):
    test_id.append(float(sample_submission.Id[i].split("_")[0]))
    


# In[ ]:


test = pd.DataFrame(test_id, columns=["Id"]).merge(data, on='Id')


# In[ ]:


X_train1, X_pretest1, y_train1, y_pretest1 = train_test_split(train.drop(['Id'], axis=1), pd.DataFrame(train_scores).drop('Id', axis=1), test_size=0.1, random_state=42)


# In[ ]:


impute = KNNImputer(n_neighbors=20)
y_train2 = impute.fit_transform(y_train1)
y_pretest2 = impute.transform(y_pretest1)


# In[ ]:


scaler = StandardScaler()
train2 = scaler.fit_transform(X_train1)
pretest2 = scaler.transform(X_pretest1)
test2 = scaler.transform( test.drop(['Id'], axis=1))


# In[ ]:


pca = PCA(n_components=445)# more than 0.95
train3 = pca.fit_transform(train2)
pretest3 = pca.transform(pretest2)
test3 = pca.transform(test2)


# In[ ]:


def weighted_NAE(yTrue,yPred):
    weights = K.constant([.3, .175, .175, .175, .175], dtype=tf.float32)
    

    return K.sum(weights*K.sum(K.abs(yTrue-yPred))/K.sum(yPred))


# In[ ]:


Model = Sequential()
Model.add(Dense(100, input_shape = (445,), activation ='elu'))
Model.add(Dropout(0.2))

Model.add(Dense(300, activation='elu'))
Model.add(Dropout(0.4))
Model.add(Dense(300, activation='elu'))
Model.add(Dropout(0.4))

Model.add(Dense(5, activation = 'elu'))

Model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=[ weighted_NAE])
Model.summary()


# In[ ]:


filepath = "best.hdf5"

checkpoint = ModelCheckpoint(filepath,
                            monitor='val_weighted_NAE',
                            verbose=1,
                            save_best_only=True, 
                            mode='min')
Model.fit(train3, y_train2, validation_data=(pretest3, y_pretest2), epochs=15, callbacks=[checkpoint])


# In[ ]:


Model.load_weights(filepath)


# In[ ]:


pred=pd.DataFrame()
pred["Id"]=sample_submission.Id
pred["Predicted"]=Model.predict(test3).flatten()
pred.to_csv('out2.csv', index=False)

