#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


# Import data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[4]:


#train_data.describe()


# In[5]:


#test_data.describe()


# In[6]:


X_train_orig = train_data.drop("label", axis = 1) /255
Y_train_orig = train_data.label
X_test = test_data / 255


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


# Split train and dev set
num_test = 0.3
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig , Y_train_orig, test_size = num_test, shuffle = False)


# In[9]:


X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape


# In[10]:


# See a training sample
index = 2000
sample = X_train.iloc[index].reshape(28, 28)
plt.imshow(sample)
print("y =" + str(Y_train[index]))


# In[ ]:





# In[11]:


from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


# In[12]:


def flat_to_square(X):
    m = X.shape[0]
    X_square = np.zeros((m, 28, 28, 1))
    for i in range(m):
        X_square[i] = X.values[i,:].reshape(28, 28, 1) 
    return X_square


# In[13]:


X_train = flat_to_square(X_train)
X_dev = flat_to_square(X_dev)
X_train.shape, X_dev.shape


# In[14]:


tf.reset_default_graph


# In[15]:


def get_one_hot(z):
    a = tf.placeholder(tf.int64, name = "y")
    
    one_hot = tf.one_hot(a, depth = 10)
    
    #Y = tf.transpose(one_hot)
    Y = one_hot
    with tf.Session() as sess:
        result = sess.run(Y, feed_dict = {a : z})
    return result


# In[16]:


Y_train = get_one_hot(Y_train)
Y_dev = get_one_hot(Y_dev)
Y_train.shape, Y_dev.shape


# In[17]:


# start of keras
def Digitmodel(input_shape):
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32, (10, 10), strides = (1, 1), name = "conv0")(X)
    X = BatchNormalization(axis = 3, name = "bn0")(X)
    X = Activation("relu")(X)
    
    X = MaxPooling2D((2, 2), name = "max_pool")(X)
    
    X = Flatten()(X)
    X = Dense(10, activation = 'softmax', name = 'fc')(X)
    
    model = Model(inputs = X_input, outputs = X, name = "DigitRecognizer")
    
    return model


# In[18]:


dgmodel = Digitmodel((28, 28, 1))


# In[19]:


dgmodel.compile("adam", "categorical_crossentropy", metrics = ["accuracy"])


# In[20]:


dgmodel.fit(X_train, Y_train, epochs = 10, batch_size = 16)


# In[21]:


loss, acc = dgmodel.evaluate(X_dev, Y_dev)
print("loss = " + str(loss))
print("dev accuracy = " + str(acc))


# In[22]:


X_test = flat_to_square(X_test)
X_test.shape


# In[23]:


Y_pred = dgmodel.predict(X_test)
Y_pred.shape


# In[24]:


Y_pred = np.argmax(Y_pred, axis = 1)
Y_pred.shape


# In[25]:


ImageId = np.arange(1,28001)


# In[26]:


Submission = pd.DataFrame({"ImageId" : ImageId, "Label" : Y_pred})
Submission.to_csv("Submission.csv", index = False)
Submission.head()


# In[27]:


dgmodel.summary()


# In[29]:


import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
plot_model(dgmodel, to_file = "Dgmodel.png")
SVG(model_to_dot(dgmodel).create(prog = "dot", format = "svg"))


# In[ ]:




