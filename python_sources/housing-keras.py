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
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
train=train_data.loc[:,train_data.dtypes!=object]
train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
test=test_data.loc[:,train_data.dtypes!=object]
test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
mean=train.mean(axis=0)
train-=mean
std=train.std(axis=0)
train/=std

mean=test.mean(axis=0)
test-=mean
std=test.std(axis=0)
test/=std


print(train.shape,test.shape)
#print(test.isnull().sum())
from keras.layers.advanced_activations import LeakyReLU
from keras import models
from keras import layers
from sklearn.cross_validation import train_test_split
X=train.iloc[:,:-1].values
y=train['SalePrice'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from keras import backend as K
from keras import optimizers
def build_model():
        def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(K.log(y_pred- y_true)), axis=-1)) 
        model=models.Sequential()
        model.add(layers.Dense(32,activation=LeakyReLU(alpha=0.1),input_shape=(X.shape[1],)))
        model.add(layers.Dense(32,activation=LeakyReLU(alpha=0.1)))
        model.add(layers.Dense(1))
        model.compile(optimizer='Adam',loss=root_mean_squared_error,metrics=['accuracy'])
        return model
model=build_model()
model.fit(X_train,y_train,epochs=80,batch_size=128,verbose=0)
test_rmse_score,test_mae_score=model.evaluate(X_test,y_test)
print(test_rmse_score) #can some one please tell what is wrong , since i am getting nan values
print(train.head())
print(test.head())
model.predict(test)
# Any results you write to the current directory are saved as output.


# In[ ]:




