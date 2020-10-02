#!/usr/bin/env python
# coding: utf-8

# ## **Digit Recognition**
# * [Setting up](#1)
#  * [Import libraries](#1.1)
#  * [Utility functions](#1.2)
# * [Prepare the data](#2)
# * [Neural Network](#3)
#  * [Build a multi-layer, fully connected neural network](#3.1)
#  * [Configure loss function, cost function and metrics](#3.2)
#  * [Prepare validation and training data sets](#3.3)
#  * [Train the neural network](#3.4)
#  * [Evaluate performance](#3.5)
#  * [Enrich training data](#3.6)
#  * [Retrain the neural network](#3.7)
#  * [Reevaluate performance](#3.8)
# * [Submit predictions](#4)

# ## Setup <a class="anchor" id="1"></a>
# 
# 

# ### Import libraries <a class="anchor" id="1.1"></a>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #seaborn plotting lib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import model_selection
from skimage.transform import rotate,rescale, resize, downscale_local_mean,warp,SimilarityTransform,AffineTransform
from skimage.morphology import thin,skeletonize
from skimage.util import invert


import math


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Utility functions <a class="anchor" id="1.2"></a>
# 

# In[ ]:


# Utility functions
def extract_images_and_labels(data,is_test=False):
    n = data.shape[0]
    print("Number of images: %s"%n)
    data_array = data.to_numpy(dtype=np.int32)
    if is_test:
        X = data_array.reshape(-1,28,28)
        print("Dimensions of images: {0}".format(X.shape))
        return X,n
    else:
        X = data_array[:,1::].reshape(-1,28,28)
        y = data_array[:,0].reshape([data_array.shape[0],1])
        print("Dimensions of labels: %s and images: %s"%(y.shape,X.shape))
        return X,y,n


# ### Prepare the data <a class="anchor" id="2"></a>
# 

# In[ ]:


training_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
testing_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


print("Number of training images: %d"%training_data.shape[0])
print("Number of pixels per image: %d"%(training_data.shape[1]-1))
training_data.head()


# In[ ]:


# Reshape training data
X,y,n = extract_images_and_labels(training_data)


# In[ ]:


# Display the first 25 images from the data set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i],cmap=plt.cm.binary)
    plt.xlabel(y[i][0])
plt.show()


# In[ ]:


# Normalize pixel data for each image by --> (pixel_value - minimum_pixel_value)( maximum_pixel_value - minimum_pixel_value)
X_normalized = X/255.0
# Re-render our previous 25 digits to make sure all is well
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_normalized[i],cmap=plt.cm.binary)
    plt.xlabel(y[i][0])
plt.show()


# ## Neural Network <a class="anchor" id="3"></a>
# 

# ### Build a multi-layer, fully connected neural network   <a class="anchor" id="3.1"></a>

# In[ ]:


# Build a multi-layer neural network (NN) to classify the images
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # flatten images to single array of length 28x28 (784)
    keras.layers.Dense(128*4,activation='relu'), # Add a hidden layer with 128*4 nodes, using RELU activation function
    keras.layers.Dropout(0.5), # Dropout 0.5
    keras.layers.Dense(128*2,activation='relu'), # Add a hidden layer with 128*2 nodes, using RELU activation function
    keras.layers.Dense(10,activation='softmax') # # Add an output layer with 10 nodes, using Softmax activation function
])


# ### Define loss function, cost function and metrics <a class="anchor" id="3.2"></a>

# In[ ]:


# Optimizer
optimizer = 'adam'
# Loss function
loss = 'sparse_categorical_crossentropy'
# Metrics 
metrics = ['accuracy']
# Compile the model
model.compile(optimizer=optimizer,
             loss=loss,
             metrics=metrics)


# ### Prepare validation and training data sets <a class="anchor" id="3.3"></a>

# In[ ]:


# Split the monolithic training data into 'training' and 'validation' data set 
X_norm_train,X_norm_validation,y_train,y_validation = model_selection.train_test_split(X_normalized,y,test_size=0.33,random_state=42)


# ### Train the neural network <a class="anchor" id="3.4"></a>

# In[ ]:


# Train the model
epochs = 100
validation_split = 0.33
# Define an 'early stopping' callback
earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=0.00001,
  patience=3)
model.fit(X_norm_train,y_train,epochs=epochs,validation_split=validation_split, callbacks=[earlystop_callback])


# ### Evaluate performance <a class="anchor" id="3.5"></a>

# In[ ]:


# Evaluate the model on the validation data set
validation_loss,validation_accuracy = model.evaluate(X_norm_validation,y_validation,verbose=2)
print("Accuracy: %.4f, Loss: %.4f"%(validation_accuracy,validation_loss))


# ### Enrich the training data <a class="anchor" id="3.6"></a>

# In[ ]:


# Generate modified images
num_repeats = 3
df = training_data.copy()
df_repeated = pd.concat([df] * num_repeats, ignore_index=True) 
data_copy = df_repeated.copy().to_numpy(dtype=np.int32)
# Handy utility to warp, thin and rotate an image randomly (within reasonable limits)
def warp_image(image):
    image = (image/255.0).reshape(28,28)
    #image = warp(image,inverse_map=AffineTransform(shear=np.random.normal(0,0.2),translation=np.random.normal(-2,2,(1,2))))
    image = thin(image,max_iter=np.random.randint(0,3))
    image = rotate(image,angle=np.random.normal(-10,10))
    return image.flatten()

X_generated = np.apply_along_axis(lambda x: warp_image(x),1,data_copy[:,1::]).reshape(-1,28,28) 
y_generated = data_copy[:,0].reshape([data_copy.shape[0],1]) 
# Add generated images to training data
X_augmented = np.concatenate((X_normalized,X_generated),axis=0)
y_augmented = np.concatenate((y,y_generated),axis=0)


# In[ ]:


# Show some examples of original & generated images 
num_rows = 5
num_images_shown = num_repeats*num_rows
figure = plt.figure(
    figsize=(15,20)
)
for i in range(num_images_shown):
    start_index = 1+i*(num_repeats+1)
    for j in range(num_repeats+1):
        plt_index =  start_index + j
        plt.subplot(num_images_shown,num_repeats+1,plt_index)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_index = (j*n)+i
        #print('{0} {1} image index {2} plt index {3}'.format(i,j,image_index,plt_index))
        image_label = y_augmented[image_index][0]
        plt.imshow(X_augmented[image_index],cmap=plt.cm.binary)
        if j == 0: 
            plt.xlabel('Original {0} ({1},{2})'.format(image_label,i,j))
        else:
            plt.xlabel('Transformed {0} ({1},{2})'.format(image_label,i,j))
figure.suptitle('Each row shows the original digit and the {0} generated variations'.format(num_repeats),y=1.05,fontsize=16)
plt.tight_layout()        
plt.show()


# ### Retrain the neural network <a class="anchor" id="3.7"></a>

# In[ ]:



# Split the monolithic training data into 'training' and 'validation' data set 
X_augmented_train,X_augmented_validation,y_augmented_train,y_augmented_validation = model_selection.train_test_split(X_augmented,y_augmented,test_size=0.33,random_state=42)
# Train the model
epochs = 100
validation_split = 0.33
model.fit(X_augmented_train,y_augmented_train,epochs=epochs,validation_split=validation_split,callbacks=[earlystop_callback])


# ### Reevaluate performance <a class="anchor" id="3.8"></a>

# In[ ]:


# Evaluate the model on the validation data set
validation_loss,validation_accuracy = model.evaluate(X_augmented_validation,y_augmented_validation,verbose=2)
print("Accuracy: %.4f, Loss: %.4f"%(validation_accuracy,validation_loss))


# ## Submit predictions  <a class="anchor" id="5"></a>

# In[ ]:


# Submit predictions
X_test,n_test = extract_images_and_labels(testing_data,is_test=True)
X_test_normalized = X_test/255.0
result = model.predict(X_test_normalized)
predictions = np.apply_along_axis(lambda row: np.argmax(row),1,result)
predictions.reshape([n_test,1])
testing_data['Label'] = predictions
testing_data['ImageId'] = list(range(1,n_test+1))
submission = testing_data[['ImageId','Label']]
submission.to_csv("submission.csv", index=False)
submission.tail()

