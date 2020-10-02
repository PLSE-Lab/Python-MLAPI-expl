#!/usr/bin/env python
# coding: utf-8

# 1. [Introduction](#introduction)<br>
# 2. [Visualazing images](#image)<br>
# 3. [How does a Convolution Neural Network works ?](#cnn) <br>
# 4. [Training our model](#training)<br>
# 5. [Result and conclusion](#result)<br>

# <a id='introduction'>1. Introduction</a><br>
# From the phone industry to military and government, cameras with artificial intelligence have become in widespread use. Phone makers have integrated this technology so that users could unlock their phone or pay just by taking a glimpse at their device, governments are using it to build giants surveillance network. I have always wondered how a computer could recognize something on an image. In this article, we will build a simple algorithm capable of recognizing handwritten numbers. Our goal is to correctly identify digits from a dataset of tens of thousands of handwritten image. In order to do so, we are going to create a neural network called Convolution Neural Network (CNN) and train it on our Dataset. They have proved to be very efficient on image recognition problem. 

# We will use the Keras library in order to build our model.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.core import Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD, RMSprop
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id='image'>2. Visualazing images</a><br>
# The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine. There are 42000 entries in our dataset. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

# In[ ]:


df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# We save the images in a numpy array 'predictors' and the corresponding digit in the array 'target'.

# In[ ]:


predictors=np.array(df_train.drop(columns=['label']))
target=np.array(df_train["label"])


# In[ ]:


test=np.array(df_test)


# We have to reshape our predictors array in order to visualize the 28x28 px images. 

# In[ ]:


images=predictors.reshape(-1,28,28)


# In[ ]:


images[0].shape


# In[ ]:


f, axs = plt.subplots(1,10,figsize=(15,15))  
axs.ravel()
for k in range(0,10):
    axs[k].imshow(images[k],cmap="gray_r")


# In order to visualize a set of handwritten digit 3, we will filter our training dataset and save the result in the 'image_three' array.

# In[ ]:


image_three=df_train[df_train["label"]==3]
image_three=np.array(image_three.drop(columns=['label']))
image_three=image_three.reshape(-1,28,28)


# In[ ]:


f, axs = plt.subplots(5,5,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,5):
    for j in range(0,5):
        axs[j,k].imshow(image_three[i],cmap="gray_r")
        i+=1


# <a id="cnn" >3. How does a Convolution Neural Network works ?</a>
# 
# What's a convolution Neural Network ? 
# Convolution Neural Networks are a category of neural networks that are very effective in image recognition and classification. 
# A Convolutional Neural Network is typically composed by four operations : 
# 1. Convolution
# 1. Non linearity 
# 1. Pooling 
# 1. Classification 

# **Convolution** <br>
# The purpose of this operation is to extract features from the input images. 
# Let's consider a 5x5 image whose pixel values are 0 or 1. Now, let's consider a 3x3 matrix. The convolution consists in sliding the matrix (called **kernel** or **filter**) over the input image and multiply the matrixes which are overlapping. The number of pixels by which we slide our filter on the input matrix is called the **stride**.
# By using different types of filter, we can identify different features in the image ( curves, edges ...). 
# During this step, the input image will be filtered by different kernels. The output will be the result of the input image's filtering. The number of filters applied during the convolution is called the **Depth** and the output matrixes are called **Convolved features** or **Feature map** .
# As convolution is a linear operation ( multiplication of matrixes ) and our model will be trained on non-linear data, we should apply a non linear operation to this layer's output. 
# 
# **Non linearity** 
# The function that is applied to Convolved Features  is called 'RELU' ( stands for Rectified Linear Unit ). It consists in replacing all the negatives pixels by 0 :
# $$ Relu=max(0,input)$$. 
# Relu is a non linear operation
# 
# Let's see what does a feature map look like ! We will apply this operation on an image and display the output. 
# 

# In[ ]:


model_test = Sequential()
model_test.add(Conv2D(50, kernel_size=5, padding="same",input_shape=(28, 28, 1),activation='relu'))


# In[ ]:


image=image_three[0]
plt.imshow(image,cmap="gray_r")
image=image.reshape(-1,28,28,1)


# In[ ]:


image.shape


# In[ ]:


conv_image=model_test.predict(image)
conv_image=np.squeeze(conv_image,axis=0)
conv_image.shape


# In[ ]:


f, axs = plt.subplots(5,10,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,10):
    for j in range(0,5):
        axs[j,k].imshow(conv_image[:,:,i],cmap="gray_r")
        i+=1


# **Pooling** <br>
# We don't want our model to overfit our training data, indeed it must be capable of correctly identifying numbers from another dataset. 
# In order to avoid overfitting, we will try to decrease the level of details of our Feature map. This operation is called **Pooling** or **subsampling**. This step is quite similar to the convolution step. It consists in sliding a filter over the input image and apply a function on the window which is flown over by the filter. The most common pooling function are max or average. 
# 
# Let's visualize the output of this operation !

# In[ ]:


model_test.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
conv_image=model_test.predict(image)
conv_image=np.squeeze(conv_image,axis=0)
conv_image.shape


# In[ ]:


f, axs = plt.subplots(5,10,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,10):
    for j in range(0,5):
        axs[j,k].imshow(conv_image[:,:,i],cmap="gray_r")
        i+=1


# **Classification** <br>
# Eventually the output of the pooling feature is given to a fully connected layer. The purpose of this operation is to use those features to classify the input image.
# Here, this task can have ten possible outputs. In the output layer of the neural network, we use an activation function called **softmax**.  This function transforms a vector of arbitrary real values into a vector of real values where each entry is in the range (0, 1], and all the entries add up to 1. 

# <a id='training'>4. Training our model</a><br>

# As Images with high pixel values would have a bigger impact on our model's training then low pixels value images, we have to normalize the data set. Hence, each image will contribute equally to our model loss. 
# We also tranform our target value ( digit ) into a vector to match the softmax function output. 
# $$ 6=[0,0,0,0,0,0,1,0,0,0]$$. 

# In[ ]:


predictors=predictors/255
test=test/255


# In[ ]:


target=to_categorical(target)
print(target)


# If we train our model on the training set and evaluate its accuracy on the same data, our model would likely perform better than it actually does on unseen data. In order to avoid overfitting, we will split our train set in two subsets : 80% of the training set will be used for training and 20% will be used for testing our model. We use the train_test_split function to generate those subsets.

# In[ ]:


predictors=predictors.reshape(-1,28,28,1)
test=test.reshape(-1,28,28,1)
X_train, X_test, y_train, y_test = train_test_split(predictors,target,test_size=0.2, random_state=42)


# The dropout layer consists in randomly dropping nodes, the aim is to avoid overfitting.<br> 
# The flatten layer consists in transforming the input array to an output array of size (*,500)

# In[ ]:


model = Sequential()
model.add(Conv2D(50, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(80, kernel_size=5, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs= 15 , batch_size=200, validation_split = 0.2)


# <a id='result'>5. Result and conclusion</a><br>

# Our model can predict handwritten's digit value with 99% accuracy. Let's test our model !

# In[ ]:


result_test=model.predict(predictors)


# In[ ]:


# select the index with the maximum probability
result_test = np.argmax(result_test,axis=1)


# Our model has predicted that the handwritten digit of the 456th image corresponds to a 5 !

# In[ ]:


result_test[456]


# As we can see below, our model has correctly predicted the outcome :)

# In[ ]:


plt.imshow(predictors[456].reshape(28,28),cmap="gray_r")


# In[ ]:


results=model.predict(test)


# In[ ]:


results=np.argmax(results,axis=1)


# In[ ]:


results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("result.csv",index=False)

