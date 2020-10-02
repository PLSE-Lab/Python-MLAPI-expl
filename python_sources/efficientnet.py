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


submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


submission


# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')


# In[ ]:


train


# In[ ]:


test


# In[ ]:


x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]


# In[ ]:


x_train.shape


# In[ ]:


x_train


# In[ ]:


x = train.iloc[:, 1:].values.astype('float32') / 255
y = train.iloc[:, 0] 


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=7) 


# In[ ]:


x_train = x_train.values.reshape(-1,28,28,1)
x_val = x_val.values.reshape(-1,28,28,1)


# In[ ]:


x_train.shape


# In[ ]:


x_val.shape


# In[ ]:


test_id = test['id']
test = test.drop("id",axis="columns")
test = test / 255
x_test = test.values.reshape(-1,28,28,1)
x_test.shape


# In[ ]:


x_train = np.concatenate((x_train,)*3, axis=-1)
x_val = np.concatenate((x_val,)*3, axis=-1)
stacked_img = np.concatenate((x_test,)*3, axis=-1)


# In[ ]:


stacked_img.shape


# In[ ]:


x_train_padded = np.pad(x_train, ((0,0),(2, 2),(2,2),(0,0)), 'constant')
x_val_padded = np.pad(x_val, ((0,0),(2, 2),(2,2),(0,0)), 'constant')
x_test_padded = np.pad(stacked_img, ((0,0),(2, 2),(2,2),(0,0)), 'constant')


# In[ ]:


x_train_padded.shape


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# In[ ]:


print(x_train_padded.shape)
print(x_val_padded.shape)
print(x_test_padded.shape)


# In[ ]:


get_ipython().system('pip install git+https://github.com/qubvel/efficientnet')


# In[ ]:


from efficientnet.keras import EfficientNetB3


# In[ ]:


model = EfficientNetB3(weights='imagenet', input_shape = (32,32,3), include_top=False)


# In[ ]:


model.trainable = False


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils


# In[ ]:


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(units = 10, activation="softmax")(x)
model = Model(input = model.input, output = predictions)
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


x_train_padded.shape


# In[ ]:


y_train.shape


# In[ ]:


history = model.fit(x_train_padded, y_train,
              epochs=10,
              batch_size = 128,
              validation_data=(x_val_padded, y_val),
              shuffle=True,
              verbose=1)


# In[ ]:


submit = model.predict(x_test_padded)
submit = np.argmax(submit,axis=1) 

submission['label'] = submit
submission.to_csv('submission.csv',index=False)


# In[ ]:




