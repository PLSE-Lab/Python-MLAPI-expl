#!/usr/bin/env python
# coding: utf-8

# This kaggle was written under the guidance of [DATAI](https://www.kaggle.com/kanncaa1)
# 
# https://www.kaggle.com/kanncaa1/convolutional-neural-network-cnn-tutorial
# 
# # Convolutional Neural Networks (CNN)
# <font color='blue'>
# <br>Content: 
# * [Loading the Data Set](#1)
# * [Normalization, Reshape and Label Encoding ](#2)
# * [Train Test Split](#3)
# * [Convolutional Neural Network](#4)
#     * [What is Convolution Operation?](#5)
#     * [Same Padding](#6)
#     * [Max Pooling](#7)
#     * [Flattening](#8)
#     * [Full Connection](#9)
# * [Implementing with Keras](#10)
#     * [Create Model](#11)
#     * [Define Optimizer](#12)
#     * [Compile Model](#13)
#     * [Epochs and Batch Size](#14)
#     * [Data Augmentation](#15)
#     * [Fit the Model](#16)
#     * [Evaluate the Model](#17)
# * [Deep Learning Tutorial for Beginners](https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners)
# * [Artificial Neural Network with Pytorch](https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers)
# * [Convolutional Neural Network with Pytorch](https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers)
# * [Recurrent Neural Network with Pytorch](https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch)
# * [Conclusion](#18)
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')


# <a id="1"></a>
# ## Loading Data Set
# * We load and visualize the data.

# In[ ]:


# Load train 
train = pd.read_csv("../input/train.csv") 
# Load test 
test= pd.read_csv("../input/test.csv")

# Let's take a look at the data. 
print(train.shape)  
train.head() # show first 5 elements


# In[ ]:


print(test.shape)
test.head()


# * Our data has 42000 row - 785 columns. The first column is label so tell us the number.The other 784 columns is the pixel's of the images.
# * We use the first label column as a "Y_train" for training.
# * After getting the labels we drop the label colums for using "X_train" as a training data.

# In[ ]:


# put labels into y_train
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)


# In[ ]:


# Let's look at a sample
img = X_train.iloc[10].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[10,0])
plt.axis("off")
plt.show()


# In[ ]:


# visualize number of digits classes
plt.figure(figsize=(12,5))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Digit Classes")
Y_train.value_counts()
Y_train.shape


# <a id="2"></a>
# ## Normalization, Reshape and Label Encoding 
# * Normalization
#     * We perform a grayscale normalization to reduce the effect of illumination's differences.
#     * To work faster, we perform normalization.
# * Reshape
#     * Train and test images (28 x 28) 
#     * We reshape all data to 28x28x1 3D matrices.
#     * Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel. 
# * Label Encoding  
#     * Encode labels to one hot vectors.(Make 1 the index according to the number.) 
#         * .........0 1 2 3 4 5 6 7 8 9 .. index
#         * 2 => [0,0,1,0,0,0,0,0,0,0]  
#         * 4 => [0,0,0,0,1,0,0,0,0,0]

# In[ ]:


# Normalize the data (for color - gri)
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# In[ ]:


#Our image already black-white but the data has R-G-B values and we reduce the size.
img = X_train.iloc[0].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()


# In[ ]:


# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# In[ ]:


# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)


# <a id="3"></a>
# ## Train Test Split
# * We split the data into train and test sets.
# * test size is 10%.
# * train size is 90%.

# In[ ]:


# Split the train(X) and the validation (Y) set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)


# <a id="4"></a>
# ## Convolutional Neural Network 
# * CNN is used for image classification, object detection 
#  <a href="https://cdn-images-1.medium.com/max/1600/0*-1Pad7loK_dFOUvS.png"><img src="https://cdn-images-1.medium.com/max/1600/0*-1Pad7loK_dFOUvS.png" alt="gec2" border="0"></a>

# 
# <a id="5"></a>
# ### What is Convolution Operation?
# 
# * We have some image and feature detector(3*3)
# * Feature detector does not need to be 3 by 3 matrix. It can be 5 by 5 or 7 by 7.
# * Feature detector = kernel = filter
# * Feauture detector detects features like edges or convex shapes. Example, if out input is dog, feature detector can detect features like ear or tail of the dog.
# * feature map = conv(input image, feature detector). Element wise multiplication of matrices.
# * feature map = convolved feature
# * Stride = navigating in input image.
# * We reduce the size of image and code can runs faster. However, we lost information. 
# * We create multiple feature maps because we use multiple feature detectors(filters).
# * Lets look at gimp. Edge detect: [0,10,0],[10,-4,10],[0,10,0]
#  <a href="https://imgbb.com/"><img src="https://image.ibb.co/m4FQC9/gec.jpg" alt="gec" border="0"></a>
# * After having convolution layer we use ReLU to break up linearity. Increase nonlinearity. Because images are non linear.
#  <a href="https://ibb.co/mVZih9"><img src="https://preview.ibb.co/gbcQvU/RELU.jpg" alt="RELU" border="0"></a>

# <a id="6"></a>
# ### Same Padding
# * As we keep applying convolutional layers, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features.
# * Input size and output size are same.
#  <a href="https://ibb.co/jUPkUp"><img src="https://preview.ibb.co/noH5Up/padding.jpg" alt="padding" border="0"></a>

# <a id="7"></a>
# ### Max Pooling
# * It makes down-sampling or sub-sampling (Reduces the number of parameters)
# * It makes the detection of features invariant to scale or orientation changes.
# * It reduce the amount of parameters and computation in the network, and hence to also control overfitting. 
#  <a href="https://ibb.co/ckTjN9"><img src="https://preview.ibb.co/gsNYFU/maxpool.jpg" alt="maxpool" border="0"></a>

# <a id="8"></a>
# ### Flattening
#  <a href="https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1"><img src="https://cdn-images-1.medium.com/max/800/1*Lzx2pNLpHjGTKcofsaSH1g.png" alt="flattenigng" border="0"></a>

# <a id="9"></a>
# ### Full Connection
# * Neurons in a fully connected layer have connections to all activations in the previous layer
# * Artificial Neural Network - ANN
#  <a href="https://medium.com/@rohanthomas.me/convolutional-networks-for-everyone-1d0699de1a9d"><img src="https://cdn-images-1.medium.com/max/800/1*DW0Ccmj1hZ0OvSXi7Kz5MQ.jpeg" alt="fullyc" border="0"></a>

# ### Summary
# * This is just a image to see the all process, the operation sequence can be change for different models.
# <a href="https://medium.com/@rohanthomas.me/convolutional-networks-for-everyone-1d0699de1a9d"><img src="https://cdn-images-1.medium.com/max/800/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg" alt="fullyc" border="0"></a>

# <a id="10"></a>
# ## Implementing with Keras

# #### Dropout
# * Dropout is a technique where randomly selected neurons are ignored during training.
#  <a href="https://ibb.co/jGcvVU"><img src="https://preview.ibb.co/e7yPPp/dropout.jpg" alt="dropout" border="0"></a>

# <a id="11"></a>
# ### Create Model
# * conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
# 

# In[ ]:


# 
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential() 
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# <a id="12"></a>
# ### Define Optimizer   
# * Adam optimizer: Change the learning rate

# In[ ]:


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# <a id="13"></a>
# ### Compile Model
# * Categorical crossentropy
# * We use categorical crossentropy. That means that we have multi class.
#  <a href="https://ibb.co/jm1bpp"><img src="https://preview.ibb.co/nN3ZaU/cce.jpg" alt="cce" border="0"></a>
# 

# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# <a id="14"></a>
# ### Epochs and Batch Size
# * Say you have a dataset of 10 examples (or samples). You have a **batch size** of 2, and you've specified you want the algorithm to run for 3 **epochs**. Therefore, in each epoch, you have 5 **batches** (10/2 = 5). Each batch gets passed through the algorithm, therefore you have 5 iterations **per epoch**.
# * reference: https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks

# In[ ]:


epochs = 10  # for better result increase the epochs
batch_size = 250


# <a id="15"></a>
# ### Data Augmentation
# * To avoid overfitting problem, we need to expand artificially our handwritten digit dataset
# * Alter the training data with small transformations to reproduce the variations of digit.
# * For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated.
#  <a href="https://ibb.co/k24CUp"><img src="https://preview.ibb.co/nMxXUp/augment.jpg" alt="augment" border="0"></a>
#     
# 

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


# <a id="16"></a>
# ### Fit the model

# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# <a id="17"></a>
# ### Evaluate the model
# * Test Loss visualization
# 

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:




