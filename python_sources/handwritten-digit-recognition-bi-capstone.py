#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

np.random.seed(64)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Loading Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = (pd.read_csv("../input/test.csv").values).astype('float32')


# In[ ]:


train.head()


# In[ ]:


test.shape


# **Converting training data to label and image**

# In[ ]:


train_label = train['label']
train_label = to_categorical(train_label)
classes = train_label.shape[1]
classes


# In[ ]:


train_image = (train.ix[:,1:].values).astype('float32')


# In[ ]:


train_image.shape


# In[ ]:


#train_image = train_image.reshape(train_image.shape[0], 28, 28)


# # Fully Connected Neural Network

# ## With PCA

# In[ ]:


train_image = train_image / 255
test = test / 255


# In[ ]:


from sklearn import decomposition
from sklearn import datasets

pca = decomposition.PCA(n_components = 784)
pca.fit(train_image)
plt.plot(pca.explained_variance_ratio_)


# In[ ]:


pca = decomposition.PCA(n_components = 100)
pca.fit(train_image)
plt.plot(pca.explained_variance_ratio_)


# In[ ]:


pca = decomposition.PCA(n_components = 40)
pca.fit(train_image)
train_pca = np.array(pca.transform(train_image))
test_pca = np.array(pca.transform(test))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense , Dropout

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(40)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


#model_fit = model.fit(train_pca, train_label, validation_split = 0.05, epochs=24, batch_size=64)


# In[ ]:


#predictions = model.predict_classes(test_pca, verbose=0)
#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
#result.to_csv("output.csv", index=False, header=True)


# ## Fully connected network without PCA

# In[ ]:


#train_image = train_image / 255
#test = test / 255


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(28 * 28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


train_image.shape


# In[ ]:


#model_fit = model.fit(train_image, train_label, validation_split = 0.05, 
                      epochs=24, batch_size=64)


# In[ ]:


#predictions = model.predict_classes(test, verbose=0)
#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
#result.to_csv("output.csv", index=False, header=True)


# # A convoluted neural network

# In[ ]:


from keras import backend

backend.image_data_format()


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = (pd.read_csv("../input/test.csv").values).astype('float32')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))


# In[ ]:


train_x = train.values[:,1:].reshape(train.shape[0], 28, 28, 1).astype('float32') / 255
train_y = to_categorical(train.values[:, 0], 10)

test_x = test.reshape(test.shape[0], 28, 28, 1).astype('float32') / 255


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model_fit = model.fit(train_x, train_y, batch_size=128, epochs=2)


# In[ ]:


predictions = model.predict_classes(test_x)
result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})


# In[ ]:


result


# In[ ]:


result.to_csv("output.csv", index=False, header=True)


# # Predictions using K-NN

# In[ ]:




