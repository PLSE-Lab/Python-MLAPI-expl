#!/usr/bin/env python
# coding: utf-8

# Here I will present a model that is accurate by 99.69 percent on the validation set. I have a lot to learn, and this is more of a record of porogress than an ideal/ most accurate model. I appreciate comments and corrections, and I hope the beginners out there might benefit from this notebook just like I did from other people's contributions!
# *directly fork and run to see performance. You can then tweak the parameters while annealing, epochs, the model etc*

# **Convolutional neural networks** 
# * In a convolutional neural network, every node is not connected to every next node.
# * When we use a regular fully connected model, most images would result in a large model(because of the number of pixels) and thus cause over-fitting. Thats why a convolutional neural network is better than a fully connected network.
# * A convolutional network has 4 layers: Convolutional layer, pooling, ReLU,and fully connected.
# * Using the above layers in the right combination gives us a good model.
# * Dense layer is what we used of any other neural network where we had a large number of nodes and the input was a linear matrix. Flatten layers allow you to change the shape of the data from a vector of 2d matrixes into the correct format for a dense layer to interpret. So we first add the flatten layer and then the dense layer.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


#load data
dig_train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
dig_test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#separate target and features
Y_train= dig_train_data.label
X_train= dig_train_data.drop("label", axis=1)

del dig_train_data      


# In[ ]:


#normalizing the data
X_train = X_train/255.0
dig_test_data= dig_test_data/255.0

#reshaping the data
X_train = X_train.values.reshape(-1,28,28,1)
dig_test_data = dig_test_data.values.reshape(-1,28,28,1)

#one-hot
Y_train = to_categorical(Y_train,num_classes = 10)


# In[ ]:


# splitting training and validation data
random_seed= 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= 0.1, random_state = random_seed)


# In[ ]:


#visualize the data, Change the value of n to visualize different values
n= 16
g= plt.imshow(X_train[n][:,:,0])


# In[ ]:


#model
#784->[Conv2D->Conv2D->Conv2D(with kernel_size=5, stride=2)->dropout]x2 ->Flatten-> Dense(128)->Dropout ->10(out)
model= Sequential()
model.add(Conv2D(filters=32, kernel_size =3, activation='relu',input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32,kernel_size =5,strides=2, padding = 'Same',activation='relu' ))
model.add(BatchNormalization())
model.add(Dropout(0.4))
 
model.add(Conv2D(filters=64, kernel_size =3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64,kernel_size =5,strides=2, padding = 'Same',activation='relu' ))
model.add(BatchNormalization())
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))


# In[ ]:


#compiling the model
model.compile(optimizer = 'adam', loss="categorical_crossentropy", metrics= ["accuracy"])


# In[ ]:


#annealing
#the learning rate gets halved if accuracy is not improved after 3 epochs
learning_rate_reduction = ReduceLROnPlateau(moitor='val_acc',
                                            patience=3 ,
                                            verbose=1,
                                           factor =0.5,
                                           min_lr=0.00001)


# In[ ]:


epochs= 40
batch_size = 128


# In[ ]:


#Data agumentation. This changes the orientation of the images by either resizing or rotating thus creating more data
#to train the model on.
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1,
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False) 


datagen.fit(X_train)


# In[ ]:


#fitting after agumentation
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# Also read:
# * https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
# * https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# 

# # Prediction and submission

# In[ ]:


#predicting results
results = model.predict(dig_test_data)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

