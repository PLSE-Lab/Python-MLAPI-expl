#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns        
from keras import layers
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
 
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train= pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


Y_train=train["label"]
X_train=train.drop(labels=["label"],axis=1)
g = sns.countplot(Y_train)
Y_train.value_counts()
X_test=test
del train


# normalize

# In[ ]:


X_train=X_train/255.00
X_test=X_test/255.00


# In[ ]:


X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)


# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.01, random_state=2)


# In[ ]:


# Example of a picture
index = 0
plt.imshow(X_train[index][:,:,0])
print ("y = " + str(np.squeeze(Y_train[index])))


# In[ ]:


m=X_train.shape[0] #no. of training examples
print(m)
print(Y_train.shape)
print(X_test.shape)


# In[ ]:


from keras.models import Sequential
model=Sequential()

model.add(Conv2D(32, (5,5), strides=(1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (5,5), strides=(1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(Dense(10, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))


# In[ ]:


model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(x = X_train, y = Y_train, epochs = 32, batch_size = 64 )


# In[ ]:


model.evaluate(X_train,Y_train)


# In[ ]:


model.evaluate(X_val,Y_val)


# In[ ]:


result= model.predict(X_test)
result=np.argmax(result,axis=1)
result = pd.Series(result,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:





# In[ ]:




