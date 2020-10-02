#!/usr/bin/env python
# coding: utf-8

# **Importing the required libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from tensorflow.python.framework import ops
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
import time
import tensorflow as tf
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


seed = 7
np.random.seed(seed)


# **Train Data**

# In[ ]:


train_file_name = '../input/train.csv'
train = pd.read_csv(train_file_name)


# In[ ]:


print(train.head(5))
X = shuffle(train.iloc[:,2:])[:100]
print(X)


# In[ ]:


#Examining correlation between columns 


# In[ ]:


#corrplot = train.corr(method='pearson')


# **Preview of train data**

# **Test Data**

# In[ ]:


test_file_name = '../input/test.csv'
test = pd.read_csv(test_file_name)


# **Preview of Test Data**

# In[ ]:


print(test.head(5))


# In[ ]:


X_tr = train


# In[ ]:


X_tr.iloc[:,1:].plot.bar(stacked=True)


# In[ ]:


y = train['target']


# In[ ]:


X_tr = X_tr.drop(columns=['target'])


# In[ ]:


scaler = StandardScaler()


# **Standard Scaler conversion for X_tr**

# In[ ]:


data = X_tr.iloc[:, 1:]
scaler.fit(data)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_tr.iloc[:, 1:] = scaler.transform(data)


# **Standard Scaler for X_test**

# In[ ]:


X_test = test
data = X_test.iloc[:, 1:]
scaler.fit(data)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_test.iloc[:, 1:] = scaler.transform(data)


# **Preview of train data after applying Standard Scaler**

# In[ ]:


print(X_tr.head(5))


# **Preview of Test data after applying standard scaler**

# In[ ]:


print(X_test.head(5))


# In[ ]:


print(X_tr.head(5))


# In[ ]:


X = X_tr.iloc[:, 1:]


# In[ ]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)


# In[ ]:


# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=200, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=500, batch_size=5, verbose=3)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
pipeline.fit(X,y)
#results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:


y_test = pipeline.predict(X_test.iloc[:, 1:])


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame()#(y_test, columns = ['Target'])
df['ID_code'] = X_test['ID_code']
df1 = pd.DataFrame(y_test, columns = ['Target'])
df['Target'] = df1['Target']
df


# In[ ]:


id_code = X_test['ID_code']


# In[ ]:


submission = df

submission.to_csv('Santander Prediction.csv', index=False)


# In[ ]:


submission


# In[ ]:




