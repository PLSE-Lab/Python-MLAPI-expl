#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <h1> 1. EXPLORE THE DATA </h1>
# <h2> a) Load the data </h2>

# In[ ]:


train = pd.read_csv('../input/mockdatadec/MOCK_DATA_DEC.csv')
test = pd.read_csv('../input/mockedtest/MOCK_TEST.csv')


# In[ ]:


train[train['x1'] == 0.01] = None
train[train['x2'] == 0.01] = None
test[test['x1'] == 0.01] = None
test[test['x2'] == 0.01] = None
train[train['x1'] == 0.00] = None
train[train['x2'] == 0.00] = None
test[test['x1'] == 0.00] = None
test[test['x2'] == 0.00] = None
train[train['x1'] == -0.01] = None
train[train['x2'] == -0.01] = None
test[test['x1'] == -0.01] = None
test[test['x2'] == -0.01] = None


# In[ ]:


#train['id'] = train.id.astype('int')


# In[ ]:


train.loc[(train.x1 > 0) & (train.x2 < 0), 'target'] = 1
train.loc[(train.x1 < 0) & (train.x2 > 0), 'target'] = 1
train.loc[train.target != 1, 'target'] = 0


# <h2>b) Print first ten rows</h2>

# In[ ]:


train.head(10)


# <h2>c) Statistic details</h2>

# In[ ]:


train.describe()


# In[ ]:


x = train.x1
y = train.x2


# 

# In[ ]:


target = train.target


# In[ ]:


x1 = train.loc[train.target > 0].x1
y1 = train.loc[train.target > 0].x2
x2 = train.loc[train.target == 0].x1
y2 = train.loc[train.target == 0].x2


# In[ ]:


plt.plot(x1,y1,'o',color='orange')
plt.plot(x2,y2,'bo')
plt.show()


# <h2> d) Explore test dataset </h2>

# In[ ]:


test.head(10)


# In[ ]:


x_test = test.x1
y_test = test.x2


# In[ ]:


plt.plot(x_test,y_test, 'ro')
plt.show()


# <h1> 2. Clean it </h1>

# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


train.dropna(0, inplace=True)
test.dropna(0, inplace=True)
print('done')


# In[ ]:


test.isna().sum()


# In[ ]:


test['id'] = test.id.astype('int')


# <h1> 3. Select target </h1>

# In[ ]:


target = train.target


# In[ ]:


del train['target'], train['id']


# <h1> 4. Split the data on training set and validation set</h1>

# In[ ]:


x_train, x_validation, y_train, y_validation = train_test_split(train.values, target.values, test_size=0.2, random_state=42)


# In[ ]:


optimAdam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


# <h1> 5. Create a neural network model </h1>

# In[ ]:


model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimAdam, metrics=['accuracy'])


# In[ ]:


bst_model_path = 'model.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


# <h1> 6. Everything is ready - let's test it</h1>

# In[ ]:


history = model.fit(x_train, y_train, validation_data=([x_validation], y_validation), epochs=50, batch_size=50, callbacks=[model_checkpoint], verbose=2)


# In[ ]:


plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


model.load_weights('./model.h5')


# In[ ]:


test_ids = test.id


# In[ ]:


del test['id']


# In[ ]:


preds = model.predict_classes(test, batch_size=10, verbose=2)


# In[ ]:


test['target'] = preds


# In[ ]:


print(test.head(10))


# In[ ]:


x1_test = test.loc[test.target > 0].x1
y1_test = test.loc[test.target > 0].x2
x2_test = test.loc[test.target == 0].x1
y2_test = test.loc[test.target == 0].x2


# In[ ]:


old_x1 = x1_test
old_y1 = y1_test
old_x2 = x2_test
old_y2 = y2_test


# In[ ]:


plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.title('Predicted test')
plt.plot(x1_test,y1_test,'o',color='orange')
plt.plot(x2_test,y2_test,'bo')
plt.subplot(122)
plt.title('Train dataset')
plt.plot(x1,y1,'o',color='orange')
plt.plot(x2,y2,'bo')
plt.show()


# <h1>7. Simple feature engineering</h1>

# In[ ]:


train['x1x2'] = train['x1'] * train['x2']
test['x1x2'] = test['x1'] * test['x2']


# In[ ]:


train.head(10)


# In[ ]:


x_train, x_validation, y_train, y_validation = train_test_split(train.values, target.values, test_size=0.2, random_state=4)


# In[ ]:


model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimAdam, metrics=['accuracy'])
bst_model_path = 'model.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


# In[ ]:


history = model.fit(x_train, y_train, validation_data=([x_validation], y_validation), epochs=50, batch_size=50, callbacks=[model_checkpoint], verbose=2)


# In[ ]:


plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


del test['target']


# In[ ]:


model.load_weights('./model.h5')
preds = model.predict_classes(test, batch_size=10, verbose=2)
test['target'] = preds


# In[ ]:


x1_test = test.loc[test.target > 0].x1
y1_test = test.loc[test.target > 0].x2
x2_test = test.loc[test.target == 0].x1
y2_test = test.loc[test.target == 0].x2


# <h1>New and old results</h1>

# In[ ]:


plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.title('New Results')
plt.plot(x1_test,y1_test,'o',color='orange')
plt.plot(x2_test,y2_test,'bo')
plt.subplot(122)
plt.title('Old results')
plt.plot(old_x1,old_y1,'o',color='orange')
plt.plot(old_x2,old_y2,'bo')
plt.show()

