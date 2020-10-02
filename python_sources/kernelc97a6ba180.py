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


# The data set is regression it have numeric

# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')
data.head()


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')
data.columns 


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')
data.tail()


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')

data.describe()


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')

y = data['motor_speed']
y.head()


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/pmsm_temperature_data.csv')

y = data['motor_speed']
X = data.drop(['motor_speed'],axis = 1)
X.head()


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('../input/pmsm_temperature_data.csv')

y = data['motor_speed']
X = data.drop(['motor_speed'],axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(X_train, y_train)
predictions = model.predict(X_valid)

result = mean_absolute_error(y_true= y_valid , y_pred=predictions)
print('the result is :',result)


# #overfitting and underfitting

# import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# 
# data = pd.read_csv('../input/pmsm_temperature_data.csv')
# 
# y = data['motor_speed']
# X = data.drop(['motor_speed'],axis = 1)
# 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
# # Define model. Specify a number for random_state to ensure same results each run
# model = DecisionTreeRegressor(random_state=1)
# 
# # Fit model
# model.fit(X_train, y_train)
# predictions = model.predict(X_valid)
# 
# result = mean_absolute_error(y_true= y_valid , y_pred=predictions)
# 
# def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(X_train, y_train)
#     preds_val = model.predict(X_valid)
#     mae = mean_absolute_error(y_valid, preds_val)
#     return(mae)
# 
# for max_leaf_nodes in data['motor_speed'] :
#    my_mae = get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid)
#    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
#     
#     
# my_mae = get_mae(y[0], X_train, X_valid, y_train, y_valid)
# best_index = 0
# for i in range(1,len(y)):
#     mae = get_mae(data['motor_speed'][i], X_train, X_valid, y_train, y_valid)
#     if mae < my_mae:
#         my_mae=mae
#         best_index = i
# best_tree_size =data['motor_speed'][best_index]

# ##Plot the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

data = pd.read_csv('../input/pmsm_temperature_data.csv',index_col = "profile_id",parse_dates=True)
sns.lineplot(data)


# In[ ]:


get_ipython().system(' pip install tensorflow==2.0.0-beta1')


# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/pmsm_temperature_data.csv')
y = data['motor_speed']
X = data.drop(['motor_speed'],axis = 1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


#define model
model = tf.keras.models.Sequential()
#add layers
#input
model.add(tf.keras.layers.Flatten(input_shape=(228,228),name="layer",data_format="channels_last"))
#hidden
'''
num_nutrons = 128
hidden_layers = 3
for i in range(hidden_layers):
  model.add(tf.keras.layers.Dense(128, activation='relu'))
'''

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#output
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#define optimizer
#compel madel
model.compile(optimizer='adam' ,loss='mean_absolute_error',
              metrics=['accuracy'])
print(model.summary())

model.fit(x=X_train,y= y_train, epochs=3,batch_size=100)
result = model.evaluate(x=X_test,y= y_test)
print(result)

