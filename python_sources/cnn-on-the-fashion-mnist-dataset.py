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


# Importing the basic libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf


# In[ ]:


# checking the version of the tensorflow 

print(tf.__version__)


# ---
# 
# Loading the fashion_mnist dataset

# In[ ]:


(x_train,y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# In[ ]:


# Checking the shape of the dataset

x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# ---
# 
# ### Importing the importing functions which will be used in CNN

# In[ ]:


from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout


# In[ ]:


# Getting the Model as well

from tensorflow.keras.models import Model


# ---
# 
# In CNN shape required by N x H x W x C ...........C means color
# 
# So, In order to make the color channel we need to make that color channel as 1 because this image is a grey scale image

# In[ ]:


# kind of normalization

x_train, x_test =  x_train/255.0, x_test/255.0


# In[ ]:


print("Shape of x_train is:-",x_train.shape)


# In[ ]:


print("Shape of x_test is:-", x_test.shape)


# ---
# 
# So, as we have N x D images or data but in CNN we require N x D x C where C is color.
#  So, we require the color information superfluously
# 
#  We will expand the data

# In[ ]:


x_train = np.expand_dims(x_train, -1)


# In[ ]:


x_train.shape


# In[ ]:


# same for x_test

x_test = np.expand_dims(x_test, -1)


# In[ ]:


x_test.shape


# In[ ]:


# Getting the number of classes 

k_classes = len(set(y_train))


# In[ ]:


print("Total number of classes are:-",k_classes)


# ---
# 
# ### Bulding the Model using the functional API this time not with the sequential way

# In[ ]:


# Giving the shape of Input on the basis of first data of input data.

i = Input(shape = x_train[0].shape)


# ---
# 
# *  Conv2D is used beacause data iss 2D in real like Height and Weight,   there is Conv1D and Conv3D as well 
# *  1st parameter used is feature map, second is the filter size, then strides and then activation function
# *  This whole calculation is applied on i type data

# In[ ]:


x = Conv2D(32, (3,3), strides=2, activation='relu')(i)


# ---
# 
# * As we did functional programming in spark and pig, the resultant of this previous layer will be used in the next layer as in functional programming concepts.

# In[ ]:


x = Conv2D(64, (3,3), strides=2, activation='relu')(x)


# In[ ]:


x = Conv2D(128, (3,3), strides=2, activation='relu')(x)


# In[ ]:


# To convert the image into the feature vector

x = Flatten()(x)


# In[ ]:


# Dropout is for regularization

x = Dropout(0.2)(x)


# In[ ]:


# Applying the Dense layer
x = Dense(512, activation = 'relu',)(x)


# In[ ]:


x = Dropout(0.2)(x)


# In[ ]:


x = Dense(k_classes, activation = 'softmax')(x)


# In[ ]:


# passing inside the model constructor

# First parameter can be considered as input and second is considered as output
cnn_model_1 = Model(i, x)


# ---
# 
# ## Compiling the model

# In[ ]:


cnn_model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])


# ---
# 
# ## Fitting the model with the data or say training the model

# In[ ]:


my_result_1 = cnn_model_1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 20)


# ---
# 
# # Plotting the loss per iteration and it should decrease

# In[ ]:


plt.plot(my_result_1.history['loss'],label = 'loss line')
plt.plot(my_result_1.history['val_loss'],label = 'validation loss line')

plt.legend()


# ---
# 
# # Did not perform that great in case of validation loss, one of reason is also that fashion mnist dataset is more tought than mnist dataset. You can call it overfitting as well.

# In[ ]:


### Plotting the accuracy per Iteration

plt.plot(my_result_1.history['accuracy'], label = 'Accuracy line')
plt.plot(my_result_1.history['val_accuracy'], label = 'Validation Accuracy line')

plt.legend()


# This also shows that in later Iteration of the model epochs the accuracy start to decrease
# 
# ---

# ---
# 
# # Plotting the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


import itertools


# In[ ]:


def plot_confusion_matrix(cm, classes, normalize = False,
                         title = 'Confusion Matrix',
                         cmap = plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix. 
    Normalization can be applied by setting 'normalize=True'.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
        
    else:
        print("Confusion Matrix, without Normalization")
        
    print(cm)
    
    
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color="white" if cm[i, j] > thresh else 'black')
    
    
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    plt.show()
    


# In[ ]:



p_test = cnn_model_1.predict(x_test).argmax(axis =1) 
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))


# In[ ]:


# Now, performing the label mapping 

my_labels = '''T-shirt/Top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split()


# In[ ]:


# We will get a list of these dresses

my_labels


# ---
# 
# ## Checking some miss classification

# In[ ]:


misclassified_idx = np.where(p_test!=y_test)[0]


# In[ ]:


misclassified_idx


# In[ ]:


# randomly selecting one data from all those misclassified data

i = np.random.choice(misclassified_idx)


# In[ ]:


i


# In[ ]:


plt.imshow(x_test[i].reshape(28,28), cmap = 'gray')

plt.title("True Label: %s Predicted %s" %(my_labels[y_test[i]], my_labels[p_test[i]]))


# * Our classification model using Neural Network got confused in some of the images.
# * It predicted the image as 'Coat' while it is 'Sandal' in real.
# * It also proves that fashion_mnist dataset is more tough in classification than normal mnist dataset.
# * These days fashion_mnist dataset is the normal bench mark for judging the classification of the any image classification technique.
# 
# ---

# In[ ]:




