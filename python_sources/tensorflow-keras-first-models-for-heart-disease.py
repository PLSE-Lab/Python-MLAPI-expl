#!/usr/bin/env python
# coding: utf-8

# After doing basic predictions with scikit-learn in the previous kernel,  [Heart Disease Playground (EDA and Predicitions)](https://www.kaggle.com/codeai/heart-disease-playground-eda-and-predictions), decided to do a small test for the same purpose using tensorflow with keras. Please, leave a comment if you find it interesting and if you have ideas for improvement.

# **Import libraries**

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# **Load the file - 'heart.csv'**

# In[ ]:


path = '../input'
heart_file = 'heart.csv'
file_path = os.listdir(path)
data_heart = pd.read_csv(os.path.join('../input',heart_file))


# **Explore data and handle missing values**

# In[ ]:


data_heart.shape


# In[ ]:


data_heart.columns


# In[ ]:


for i in data_heart.index:
    if (data_heart.loc[i].isnull().sum() != 0):
        print('Missing value at ', i)
print('Done!')


# **First model: all features used**

# **Separate features and target**

# In[ ]:


data_heart_features = data_heart.loc[:,data_heart.columns!='target']
data_heart_target = data_heart.iloc[:,-1]


# In[ ]:


X_train_all,X_test_all,y_train_all,y_test_all = train_test_split(data_heart_features,data_heart_target,test_size=0.20,random_state=42)


# In[ ]:


X_train_all.shape


# In[ ]:


X_test_all.shape


# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_all.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])


# In[ ]:


optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train_all,y_train_all,epochs=1000)


# In[ ]:


print(model.evaluate(X_test_all,y_test_all))


# **Second model: usingonly four features - the ones that we found more correlated to the target, from previous kernel**

# In[ ]:


data_features = data_heart.loc[:,['cp','slope','exang','thal']]
data_target = data_heart.iloc[:,-1]


# In[ ]:


X_train_four,X_test_four,y_train_four,y_test_four = train_test_split(data_features,data_target,test_size=0.20,random_state=42)


# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_four.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])


# In[ ]:


model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])


# In[ ]:


model.fit(X_train_four,y_train_four,epochs=1000)


# In[ ]:


print(model.evaluate(X_test_four,y_test_four))


# Performance a bit better than in [Heart Disease Playground (EDA and Predictions)](https://www.kaggle.com/codeai/heart-disease-playground-eda-and-predictions) !

# In[ ]:




