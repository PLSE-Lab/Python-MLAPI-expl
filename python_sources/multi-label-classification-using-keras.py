#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read dataset
df_questions = pd.read_csv("../input/questions.csv", nrows=5000, usecols =['Id', 'Score', 'AnswerCount'],encoding='latin1')
# df_questions = pd.read_csv("questions.csv", parse_dates=["ClosedDate", "CreationDate", "DeletionDate"])
df_questions = df_questions.dropna()
df_questions.head(15)

X = df_questions.values.astype('float')
X[0].shape


# In[ ]:


# read dataset
df_question_tags = pd.read_csv("../input/question_tags.csv", nrows=5000,encoding='latin1')
df_question_tags = df_question_tags.dropna()
df_question_tags.head(15)

y = df_question_tags.values[:4967]


# In[ ]:


# model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y[:,1])
encoded_Y = encoder.transform(y[:,1])
encoded_Y[1]


# In[ ]:


np_y = np.array([])
for _, tag in enumerate(encoded_Y):
    np_y = np.append(np_y, tag)


# In[ ]:


Y = np.column_stack((y[:,0], np_y)).astype('float')


# In[ ]:


history = model.fit(X, Y, verbose=0,
                     epochs=100,
                     batch_size=128,
                     validation_split=0.4)


# In[ ]:


acc = history.history['acc']
loss = history.history['loss']


# In[ ]:


plt.plot(range(len(acc)), acc,'b', label="training accuracy", color='g')
plt.plot(range(len(loss)), loss,'b', label="training loss", color='r')
plt.show()


# In[ ]:




