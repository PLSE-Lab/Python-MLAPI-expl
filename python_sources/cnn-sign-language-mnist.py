#!/usr/bin/env python
# coding: utf-8

# # **                     Convolutional Neural Network (CNN)**
# ##  **What is CNN?** 
#    * CNN, is a well-known method in computer vision applications.
#    * CNN are the most representative supervised Deep Learning model.
#    
# ### CNN Architecture
# * A typical CNN architecture can be summarized in the picture below.
# <br>
# <br>
#  
# <a href="https://ibb.co/xfMnpRK"><img src="https://i.ibb.co/X3znhvf/cnn.jpg" alt="cnn" border="0"></a>
# 
# 
# 
# ### ** Convolutional Neural Network Steps :**
# 
# ### * ** Step 1. Upload Dataset :** *
# * We will use the Sign Language MNIST dataset.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
print(train.shape)
print(test.shape)
train.head()


# In[ ]:


from IPython.display import Image
Image("../input/sign-language-mnist/amer_sign3.png")


# In[ ]:


test.tail()


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1)


# In[ ]:


plt.figure(figsize = (15,7))
g = sns.countplot(Y_train, palette ="icefire")
plt.title("Number of sign classes")
Y_train.value_counts()


# In[ ]:


img = X_train.iloc[2].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(X_train.iloc[3,0])
plt.axis("off")
plt.show()


# In[ ]:


img = X_train.iloc[1].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(X_train.iloc[3,0])
plt.axis("off")
plt.show()


# In[ ]:


# Normalization

X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
test = ( test - np.max(test))/(np.max(test)-np.min(test))
print("X_train Shape : ", X_train.shape)
print("test shape : ", test.shape)


# In[ ]:


test = test.drop(["label"], axis = 1)


# In[ ]:


test.shape


# In[ ]:


X_train.head()


# In[ ]:


# Reshape

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("X_train shape : ", X_train.shape)
print("test shape : ",test.shape)


# In[ ]:


Y_train.value_counts()


# In[ ]:


Y_train = Y_train.values.reshape(-1,1)
Y_train.shape


# In[ ]:


# Label Encoding
from keras.utils.np_utils import to_categorical
Y_train= to_categorical(Y_train)


# In[ ]:


from numpy import argmax
inverted = argmax(Y_train[4])
print(inverted)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train, random_state = 42, test_size = 0.1)


print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)


# In[ ]:


# Some examples
plt.imshow(X_train[2][:,:,0],cmap='gray')
plt.show()


# In[ ]:


# Some examples
plt.imshow(X_train[3][:,:,0],cmap='gray')
plt.show()


# ### * ** Step 2. Convolutional Layer :** *
# * The most critical component in the model is the convolutional layer. This part aims at reducing the size of the image for faster computations of the weights and improve its generalization.
# * It is responsible for detecting the properties of the image. This layer applies some filters to the image to remove the low and high level features in the image.
# * The purpose of the convolution is to extract the features of the object on the image locally. It means the network will learn specific patterns within the picture and will be able to recognize it everywhere in the picture.
# 
# <a href="https://ibb.co/m6MPW5Q"><img src="https://i.ibb.co/XSvGRxm/convolution-operation-24.png" alt="convolution-operation-24" border="0"></a>
# 

# ### * ** Step 3. ReLu (Non Linearity) :** *
# * All the pixel with a negative value will be replaced by zero.
# * ReLu Function f (x) = max (0, x)
# 
# * Applying the ReLu function to the Feature Map produces a result as follows.
# 
# <a href="https://ibb.co/L9hC4X0"><img src="https://i.ibb.co/8P4jFLm/relu.png" alt="relu" border="0" height="500" width="500"></a>
# 
# 

# ### * ** Step 4. Pooling Layer :** *
# * The purpose of the pooling is to reduce the dimensionality of the input image. 
# * By diminishing the dimensionality, the network has lower weights to compute, so it prevents overfitting.
# * There are many pooling operations, but the most popular is pooling max.
# 
# <a href="https://ibb.co/vPkcyh3"><img src="https://i.ibb.co/T8twj14/pooling.png" alt="pooling" border="0"></a>

# ### * ** Step 5. Flattening Layer:** *
# * The task of this layer is simply to prepare the data in the input of the last and most important layer, Fully Connected Layer.
# * Incoming matrices are converted to one-dimensional arrays.
# 
# <a href="https://ibb.co/Vw8cysm"><img src="https://i.ibb.co/3RH9Xnz/flatten.png" alt="flatten" border="0"></a>

# ### * ** Step 6.  Fully Connected Layer (FC Layer):** *
# * It takes the data from the Flattening process and performs the learning process through the Neural network.
# 
# <a href="https://ibb.co/BnwTcc1"><img src="https://i.ibb.co/GHFCnnh/fc.jpg" alt="fc" border="0" height="1000" width="2650"></a>

# In[ ]:


# Create CNN 

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size=(3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(25, activation = "softmax"))


# In[ ]:


# Optimizer

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)


# In[ ]:


# Model Compile

model.compile(optimizer = optimizer, loss ="categorical_crossentropy", metrics =["accuracy"])


# In[ ]:


epochs = 50
batch_size = 255


# In[ ]:


# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




