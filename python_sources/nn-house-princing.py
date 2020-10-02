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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing.imputation import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
traincorr = df_train.corr()['SalePrice']
traincorr


# In[ ]:


#Select the best correlations
df1 = traincorr.to_frame()
df1.sort_values(by='SalePrice', ascending=False)


# In[ ]:


cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd']
train = df_train[cols]


# In[ ]:


# Create dummy values
train = pd.get_dummies(df_train)
#filling NA's with the mean of the column:
train = train.fillna(train.mean())


# In[ ]:


from sklearn.preprocessing import StandardScaler

# Always standard scale the data before using NN
scale = StandardScaler()
X_train = train[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd']]
X_train = scale.fit_transform(X_train)
# Y is just the 'SalePrice' column
y = train['SalePrice'].values
seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33, random_state=seed)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor


# In[ ]:


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mae])
    return model


# In[ ]:


model = create_model()
model.summary()


# In[ ]:


history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=32)


# In[ ]:


# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


model.save("model.h5")


# In[ ]:


<a href="../input/model.h5">Baixar</a>

