#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df1 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df2 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df = pd.concat([df1, df2], ignore_index=True)
df.shape


# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
y_train = train.iloc[:,-1]
x_train = train.iloc[:, :-1]
x_train.head()
y_test = test.iloc[:,-1]
x_test = test.iloc[:,:-1]
x_train =x_train.drop(['Serial No.'],axis=1)
x_test=x_test.drop(['Serial No.'],axis=1)


# In[ ]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)
x = np.concatenate((x_train_minmax, x_test_minmax), axis=0)
y = np.concatenate((y_train.to_numpy(), y_test.to_numpy()), axis=0)


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def build_model():
    
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[7]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse','mae'])
    return model


# In[ ]:


model = build_model()
history = model.fit(
  x, y,validation_split = 0.2,
  epochs=1000, verbose=2)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
result = permutation_importance(model,x_test_minmax,y_test,scoring='neg_mean_squared_error')
importance = result.importances_mean
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[ ]:




