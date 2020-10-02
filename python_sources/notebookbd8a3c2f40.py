#!/usr/bin/env python
# coding: utf-8

# Try out.

# In[ ]:


# Load Data 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
train = pd.concat([train['image_name'], train.tags.str.get_dummies(sep=' ')], axis=1)


# In[ ]:


labels = train.columns[1:]


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


# Data Preparation

import cv2
from tqdm import tqdm

nTrain = 20000
x_train = []
for f in train.image_name.values[:nTrain]:
    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
    x_train.append(cv2.resize(img, (32, 32)))
    
x_train = np.array(x_train, np.float16)
y_train = train.ix[:nTrain-1,1:].values.astype('int8')


# In[ ]:


print(x_train.shape)
print(y_train.shape)


# In[ ]:


# Keras Model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense , Dropout, Flatten


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

from keras.optimizers import Adam
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_fit = model.fit(x_train, y_train, validation_split = 0.05, epochs=3, batch_size=64)


# In[ ]:


# Display Loss and Accuracy

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

history = model_fit.history

loss_values = history['loss']
val_loss_values = history['val_loss']

epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[ ]:


plt.clf()   # clear figure
acc_values = history['acc']
val_acc_values = history['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# In[ ]:


# Test Data

test = pd.read_csv('../input/sample_submission.csv')
x_test = []
for f in test.image_name.values:
    img = cv2.imread('../input/test-jpg/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (32, 32)))
    
x_test = np.array(x_test, np.float16)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


preds = [' '.join(labels[index] for index, p in enumerate(pre) if p > 0.) for pre in pred]


# In[ ]:


subm = pd.DataFrame()
subm['image_name'] = test.image_name.values
subm['tags'] = preds
subm.to_csv('submission.csv', index=False)

