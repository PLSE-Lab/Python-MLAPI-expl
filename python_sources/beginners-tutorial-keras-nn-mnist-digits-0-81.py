#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Let's read **data**:

# In[ ]:


test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# Now we can look in our data. Every pixel is encoded as int value in columns.

# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


Y_train_df = train_df['label']
#Y_train_df = pd.get_dummies(Y_train_df).values
#Y_train_df = Y_train_df.values.reshape(-1, 1)
#Y_train_df = standard_scaler.fit_transform(Y_train_df)
#Y_train_df = keras.utils.np_utils.to_categorical(Y_train_df)
#Y_train_df = Y_train_df.reshape(-1, 10, 1)
Y_train_df.shape


# In[ ]:


X_train_df = train_df.drop(columns="label")
#X_train_df = standard_scaler.fit_transform(X_train_df)
#X_train_df = keras.utils.np_utils.to_categorical(X_train_df)
X_train_df.shape


# In[ ]:


seed_num = 7
np.random.seed(seed_num)
# split into 67% for train and 33% for test
X_train, X_valid, y_train, y_valid = train_test_split(X_train_df, Y_train_df, test_size=0.15, random_state=seed_num)


# In[ ]:


print(X_train)
print(X_valid)
print(y_train)
print(y_valid)


# Here we go) Let's create our semple recognition model.
# In this kernel we use full dense custom model.
# You may trying to get more efficient parameters) Doing it by yourself!

# In[ ]:


def create_model(train_data):
    # create model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(800, input_dim=train_data.shape[1], activation='relu'))
    #model.add(keras.layers.Dense(800, activation='relu'))
    model.add(keras.layers.Dense(500, activation='relu'))
    #model.add(keras.layers.Dense(15, activation='relu'))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation("softmax"))
    # Compile model
    model.compile(optimizer ='sgd', loss = 'sparse_categorical_crossentropy') 
#               metrics =[keras.metrics.mae])
    return model


# In[ ]:


model = create_model(X_train_df)


# Starting train our model. Parameter **epochs** was picked up in trainings process.

# In[ ]:


history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid), epochs=15, batch_size=1000)


# Predict on **test** data

# In[ ]:


answer = model.predict(test_df)


# In[ ]:


answer = answer.astype(int)


# 

# In[ ]:


y = np.argmax(answer, axis=-1)
y


# There is an answer with ~ 0.81% rights!

# In[ ]:


frames_aswer = pd.DataFrame({"ImageId": pd.DataFrame(y).index.values+1,
                             "Label": y
                            })
frames_aswer


# In[ ]:




