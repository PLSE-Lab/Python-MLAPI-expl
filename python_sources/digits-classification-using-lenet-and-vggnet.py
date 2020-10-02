#!/usr/bin/env python
# coding: utf-8

# ### Overview
# This post is about classifying sign language digits using two popular CNN architectures that are often used for digits classification - LeNet and VGGNet.
# 
# For the classificaion problem, I have implemented both the architectures as classes and built models for each of the architectures to evaluate the accuracy. 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.normalization import BatchNormalization
import os


# ### EDA
# Let's start with some basic EDA before we dive into the network architectures. The directory has two files in the npy format - the input image array and the target array.
# 

# In[ ]:


X = np.load('../input/Sign-language-digits-dataset/X.npy')
y = np.load('../input/Sign-language-digits-dataset/Y.npy')
print(X.shape)
print(y.shape)
X = X.reshape(-1, X.shape[1], X.shape[2], 1)


# As we can see, our input consists of around 2000 training images with 64x64 dimensions. The target array is a 10d-vector containing a 1 in the respective index of the class it represents and a 0 otherwise. For example, the target for an image which belongs to class 4 is [0,0,0,0,1,0,0,0,0,0].
# 
# We also reshape our input array so that the number of channels is included in the shape. Since all the images are grayscale, we reshape the input array such that the number of channels is set to 1. 

# In[ ]:


print(min(X.flatten()))
print(max(X.flatten()))


# We can also see that the pixel values have been normalized between 0 and 1. Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. As such it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1.

# We now plot a histogram to get an idea about the number of images in each class. A disproportional amount of images in any one class could end up biasing the model towards that class.

# In[ ]:


y_labels = np.where(y == 1)[1]
bars = np.asarray(range(0,10,1))
n, bins, patches = plt.hist(x=y_labels, bins=len(bars), color='#0504aa',
                            alpha=0.7, rwidth=0.85)
bins += 0.5
plt.xticks(bins,bars)
plt.xlabel('Digit', fontsize=14)
plt.ylabel('Frequency', fontsize=14)


# Each number class has more or less the same number of training images so biasing the model is not a major problem at this point. 

# ### LeNet
# 
# The LeNet architecture is an excellent grassroots architecture that is both small enough to run on your system and large enough to provide interesting and robust results, especially for digit classification problems. The network was conceptualized by Yann LeCunn in his 1998 paper, [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

# ![LeNet Architecture](http://blog.dataiku.com/hs-fs/hubfs/Dataiku%20Dec%202016/Image/le_net.png?width=620&name=le_net.png)
# 
# [(Image source)](https://blog.dataiku.com/deep-learning-with-dss)

# As we can see, the architecture consists of 2 sets of Convolutional Layer => Max Pool Layer, followed by a Hidden Layer that is finally fed into a Softmax classifier,

# In[ ]:


class LeNet:

    def build(width, height, depth, classes):
    
        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(20, 5, padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


# For our implementation of the architecture, a single input to the model is a grayscale image of dimensions 64x64x1. We start with a Conv2D layer having 20 convolutional filters of receptive field F = 5, i.e, 5x5 filters. This is activated using ReLu and fed into a Maxpooling layer of 2x2 with the same stride that reduces the input dimensions by half while keeping the depth the same. 
# 
# The above combination of layers is fed into another similar set where we have 50 convolutional filters of F = 5, followed by ReLu activation and Maxpooling. The output of these two sets is fed into a Fully Connected Layer of 500 nodes that is then directed towards a Softmax classifier which outputs the 10d vector that indicates the class of the image.

# ### VGGNet
# 
# The VGG network architecture was introduced by Simonyan and Zisserman in their 2014 paper, [Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/abs/1409.1556). The VGGNet architecture is generally characterized by the 3x3 convolutional layers across the entire network. The network is also known for stacking multiple Convolutional layers before applying Maxpooling.
# 
# However, the implementation below is a much shallower version of the original VGGNet that I found in Adrian Rosebrock's book, [Deep Learning for Computer Vision](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/). Because we have a relatively small amount of training data, we can demonstrate that a much lighter network still produces an output that is good enough. Nevertheless, the below implementation still adheres to the core principles of VGGNet and also includes batch normalization and dropout layers.

# In[ ]:


class MiniVGGNet:
    
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model


# Below is a table highlighting the specifics of each layer in the network. 
# 
# 
# | Layer               |    Output    | Filter Size/N |
# |---------------------|:------------:|:-------------:|
# | Input image         |  64 x 64 x 1 |               |
# | Conv layer          | 64 x 64 x 32 |    3 x 3/32   |
# | Activation          | 64 x 64 x 32 |               |
# | Batch normalization | 64 x 64 x 32 |               |
# | Conv layer          | 64 x 64 x 32 |    3 x 3/32   |
# | Activation          | 64 x 64 x 32 |               |
# | Batch normalization | 64 x 64 x 32 |               |
# | Maxpool             | 32 x 32 x 32 |     2 x 2     |
# | Dropout             | 32 x 32 x 32 |               |
# | Conv layer          | 32 x 32 x 64 |    3 x 3/64   |
# | Activation          | 32 x 32 x 64 |               |
# | Batch normalization | 32 x 32 x 64 |               |
# | Conv layer          | 32 x 32 x 64 |    3 x 3/64   |
# | Activation          | 32 x 32 x 64 |               |
# | Batch normalization | 32 x 32 x 64 |               |
# | Maxpool             | 16 x 16 x 64 |     2 x 2     |
# | Dropout             | 16 x 16 x 64 |               |
# | Fully connected     |      500     |               |
# | Activation          |      500     |               |
# | Batch normalization |      500     |               |
# | Dropout             |      500     |               |
# | Fully connected     |      10      |               |
# | Softmax             |      10      |               |
# 
# 
# <br><p>
# The advantage of combining convolutional layers together is that the kernels are able to learn a richer set of features before pooling is done. Batch normalization and dropout are "utility layers" which help prevent overfitting on the data.</p>

# ### Training LeNet
# 
# We now split our training and testing data and then instantiate a LeNet object to train the data on.

# In[ ]:


(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.05, random_state=42)
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model1 = LeNet.build(width=64, height=64, depth=1, classes=10)
model1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# We train the model by calling the fit() function of the LeNet model.

# In[ ]:


print("[INFO] training network...")
H = model1.fit(trainX, trainY, validation_split=0.15, batch_size=128, epochs=50, verbose=1)


# We now test the model on the test set.

# In[ ]:


print("[INFO] evaluating network...")
predictions = model1.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in np.unique(y_labels)]))


# Plotting for training and validation loss and accuracy.

# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# As we can see, the decrease in the training loss does not happen immediately. Rather, it gradually reduces and converges at a slower rate, which gives us the impression that running the model for about 50 epochs must have probably overfit the model. This also shows in the comparitively lower precision obtained from the testing set.

# ### Training mini-VGGNet
# 
# We now train the smaller version of VGGNet.

# In[ ]:


model2 = MiniVGGNet.build(width=64, height=64, depth=1, classes=10)
model2.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H1 = model2.fit(trainX, trainY, validation_split=0.15, batch_size=128, epochs=50, verbose=1)


# In[ ]:


print("[INFO] evaluating network...")
predictions = model2.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in np.unique(y_labels)]))


# Plotting for training and validation loss and accuracy.

# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H1.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H1.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# Looking at the graph of the training loss, we can see that the model converges more smoothly in case of the VGGNet. We can also see that the model achieves high training accuracy with possible overfit, but the results from the testing set are satisfying enough to validate the model as good enough.
