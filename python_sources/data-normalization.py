#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset
import keras
from keras.utils import to_categorical
from keras import models
from keras import layers


df = pd.read_csv('../input/covtype.csv')


# In[ ]:


#Select predictors
x = df[df.columns[:54]]
#Target variable 
y = df.Cover_Type


# In[ ]:


x.head()


# <h1> Normalization </h1>
#     <p> Normalization is a data preprocessing technique for machine learning with the goal of changing the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. It is not required to do normalization everytime, it is only required when numerical features have a significant difference in terms of ranges </p> 
# 
# <p> For example, take age and income. Age ranges from 1-100 usually and income usually ranges from 0-50,000 and higher. Income is significantly larger than age and when we do further analysis like multivariate linear regression, income influences the result more than the age feature due it its higher value. However, it doesn't necessarily mean that income is more important than age as a predictor. In order to see the effects of normalization, let's create 2 Neural networks, where we do normalization for one and not for the other. </p>

# <h1> Unnormalized </h1> 

# In[ ]:


#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)

model = keras.Sequential([
 keras.layers.Dense(64, activation=tf.nn.relu,                  
 input_shape=(x_train.shape[1],)),
 keras.layers.Dense(64, activation=tf.nn.relu),
 keras.layers.Dense(8, activation=  'softmax')
 ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))


# <p> As you can see, accuracy is low and is unchanging per epoch, now let's try using normalized data</p>

# In[ ]:


from sklearn import preprocessing


#Select numerical columns which needs to be normalized
train_norm = x_train[x_train.columns[0:10]]
test_norm = x_test[x_test.columns[0:10]]

# Normalize Training Data 
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)

#Converting numpy array to dataframe
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
print (x_train.head())

# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_test.head())


# In[ ]:


history2 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))


# <p> Accuracy has improved significantly and the model now learns per epoch, which is better compared to before </p>
#  <h1> Explanation </h1>
# <p>   The model with the unnormalized data didn't learn in 26 epochs because different features do not have similar ranges of values, as a result, gradients oscillate back and forth and take a long time before it can finally find its way to the global/local minimum. In order to remedy the model learning problem, data normalization was used, which make sures that the different features take on similar ranges of values so that gradient descents can converge more quickly </p>

# <h1> Footnote and reference </h1>
# I still am new to the field of machine learning and data science in general, feedback and additional resources would definitely help! The source I used for this kernel is https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029. I thought it would be nice to try it myself in this kernel as a way of taking notes and might as well publish it to share it with others. Thanks for reading!
