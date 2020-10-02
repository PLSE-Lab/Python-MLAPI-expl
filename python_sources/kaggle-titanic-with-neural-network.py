#!/usr/bin/env python
# coding: utf-8

# import all library

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# importing data of titanic dataset

# In[ ]:


gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# analyzing train and test dataset

# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# extracting Survived from the train dataset and making label

# In[ ]:


x_train=train.drop("Survived",axis=1)
y_train=train['Survived']


# removing all unnecessary features that are not impacting label.
# PassengerId,
# Name,
# Ticket, 
# Cabin

# In[ ]:


x_train=x_train.drop('PassengerId',axis=1)
x_train=x_train.drop('Name',axis=1)
x_train=x_train.drop('Ticket',axis=1)
x_train=x_train.drop('Cabin',axis=1)


# same with the test dataset

# In[ ]:


x_test=test
x_test=x_test.drop('Name',axis=1)
x_test=x_test.drop('PassengerId',axis=1)
x_test=x_test.drop('Ticket',axis=1)
x_test=x_test.drop('Cabin',axis=1)


# analyzing relationship between features and label.

# In[ ]:


train.plot(kind='scatter',x='Survived',y='Age')
plt.show()


# In[ ]:


train.plot(kind='scatter',x='Survived',y='Pclass')
plt.show()


# In[ ]:


train.plot(kind='scatter',x='Survived',y='Fare')
plt.show()


# using one hot encoding on both datasets

# In[ ]:


x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)


# checking for missing values in both dataset

# In[ ]:


x_train.info()


# In[ ]:


x_test.info()


# both dataset have missing values in age(missing value in Fare in test dataset)

# In[ ]:


x_train.Age.fillna(x_train.Age.mean(), inplace=True)
x_test.Age.fillna(x_test.Age.mean(), inplace=True)
x_test.Fare.fillna(x_test.Fare.mean(),inplace=True)


# here, i'm doing data normalization.(z-normalization)

# In[ ]:


train_stats=x_train.describe()
train_stats=train_stats.transpose()
train_stats


# In[ ]:


test_stats=x_test.describe()
test_stats=test_stats.transpose()
test_stats


# normalizing both datasets

# In[ ]:


def norm_train(x):
  return (x - train_stats['mean']) / train_stats['std']
x_train=norm_train(x_train)


# In[ ]:


def norm_test(x):
  return (x - test_stats['mean']) / test_stats['std']
x_test=norm_test(x_test)


# 4 layers neural network with 10,20,8,1 neurons in respective layers.

# In[ ]:


model=tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='relu',input_shape=[len(x_train.keys())]),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# training the model.

# In[ ]:


model.fit(x_train,y_train,epochs=15,validation_split=0.1)


# evaluating the model.

# In[ ]:


loss,acc=model.evaluate(x_train,y_train)


# In[ ]:


pred=model.predict(x_test)
pred=np.around(pred)
prediction = pd.DataFrame(pred, columns=['Survived']).to_csv('y_titanic.csv')

