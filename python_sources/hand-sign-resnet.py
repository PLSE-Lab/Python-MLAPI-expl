#!/usr/bin/env python
# coding: utf-8

# # Hand Sign Image Recognition using Pytorch Deep Neural Network  

# In this notebook , I am building a Hand Sign Recognition system with Deep NN using Pytorch . Hand Sign means numeric signs (e.g. 0,1,2,3,4,5) using our hand . We have built a deep learning neural network to accomplish our target . We are using Pytorch deep learning framework . 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The Hand sign data set that we will use is available in .h5 format . So, we use h5py package we load the data . Then we convert the data in numpy array . Each image is a 3 channel image with 64 * 64  dimension. There are 1080 images in the train data and 120 in test data. 

# In[2]:


def load_dataset():
    train_dataset = h5py.File('../input/hand-sign/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../input/hand-sign-test/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    #classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig,test_set_x_orig,test_set_y_orig


# Load the data and have quick look at rhe shape of the data.

# In[3]:


train_x,train_y,test_x,test_y = load_dataset()
train_x.shape,train_y.shape,test_x.shape,test_y.shape


# In[4]:


train_y = train_y.reshape((1080,))
test_y = test_y.reshape((120,))
Y_train = np.zeros([1080,6])
count = 0
for i in train_y:
    Y_train[count,i] = 1
    count = count+1
Y_test = np.zeros([120,6])
count = 0
for  i in test_y:
    Y_test[count,i] = 1
    count = count+1


# Just to have a look at how the images look like we use matplotlib.pyplot.imshow to display 2 imges from training data and 2 from test data.

# In[5]:


import matplotlib.pyplot as plt
plt.subplots(2,2)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)

plt.subplot(2,2,1)
plt.title('train_x[5] label : 4')
plt.imshow(train_x[5])
plt.subplot(2,2,2)
plt.title('train_x[10] label : 2')
plt.imshow(train_x[10])

plt.subplot(2,2,3)
plt.title('test_x[5] label : 0')
plt.imshow(test_x[5])
plt.subplot(2,2,4)
plt.title('test_x[10] label : 5')
plt.imshow(test_x[10])


# In[6]:


from keras import layers
from keras.layers import Input,Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')


# In[7]:


def plain_layer(X,n_c):
    X_in = X
    X = Conv2D(n_c,kernel_size=(3,3),padding='same')(X_in)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    
    return X


# In[8]:


def identity_block(X,F):
    X_in = X
    
    F1,F2,F3 = F
    
    X = Conv2D(F1,kernel_size=(3,3),padding='same')(X_in)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F2,kernel_size=(3,3),padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F3,kernel_size=(3,3),padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    
    X_in = Conv2D(F3,kernel_size=(3,3),padding='same')(X_in)
    X_in = BatchNormalization(axis=3)(X_in)
    
    X = Add()([X,X_in])
    X = Activation('relu')(X)
    
    return X


# In[9]:


def Resnet(input_shape=(64,64,3),classes=6):
    X_in = Input(input_shape)
    
    X = plain_layer(X_in,32)
    
    F1 = [16,16,32]
    X = identity_block(X,F1)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    
    F2 = [16,16,32]
    X = identity_block(X,F2)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    
    F3 = [16,16,32]
    X = identity_block(X,F3)
    #X = MaxPooling2D(pool_size=(2,2))(X)
    
    #X = plain_layer(X,32)
    X = AveragePooling2D((2,2))(X)
    
    X = Flatten()(X)
    X = Dense(512,activation='relu')(X)
    X = Dense(128,activation='relu')(X)
    X = Dense(classes,activation='softmax')(X)
    
    model = Model(inputs=X_in,outputs=X,name='Resnet')
    return model


# In[10]:


X_train = train_x/255
X_test = test_x/255


# In[11]:


my_model = Resnet()


# In[12]:


my_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
my_model.fit(x=X_train,y=Y_train,epochs=20,batch_size=32)
time.sleep(5)


# In[ ]:


my_model.evaluate(X_train,Y_train,batch_size=32)


# In[ ]:


my_model.fit(x=X_train,y=Y_train,epochs=20,batch_size=32)
time.sleep(5)


# In[ ]:


my_model.evaluate(X_train,Y_train,batch_size=32)


# In[ ]:


my_model.fit(x=X_train,y=Y_train,epochs=2fjjf0,batch_size=32)
time.sleep(5)


# In[ ]:


my_model.evaluate(X_train,Y_train,batch_size=32)


# In[ ]:


my_model.evaluate(X_test,Y_test,batch_size=32)


# Classification report on training data 

# Classification report on test data . 

# In[ ]:


form sklearn.metrics import classification_report
pred = my_model.predict(X_test)
pred_label = np.argmax(pred,axis=1)
print(classification_report(Y_test,pred_label))


# In[ ]:




