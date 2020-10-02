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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers


# In[ ]:


train = pd.read_csv('../input/train.csv')
gender = np.array([0 if x == 'male' else 1 for x in train['Sex']])


# In[ ]:


X_train = np.array([train['Pclass'],gender,train['Age'],train['Fare']]).T
Y_train = np.array(train['Survived'])[:,np.newaxis]


# In[ ]:


model = Sequential()

model.add(Dense(15,activation='relu',input_dim=4))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, Y_train, validation_split=0.25, batch_size=50, epochs=15, shuffle=True, verbose=1)

print('val_loss',history.history['val_loss'][-1],'val_acc',history.history['val_acc'][-1])


# In[ ]:


import matplotlib.pyplot as plt

#score = model.evaluate(Xva,Yva,batch_size=100,verbose=1)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(model.predict(X_train))

