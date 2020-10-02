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

import tensorflow as tf
rand_state = 42
tf.set_random_seed(rand_state)
np.random.seed(rand_state)

from skimage import exposure
import cv2
import glob
import time
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


# # Introduction  
# 
# "What would you say if I told you there is a app on the market that tell you if you have a hotdog or not a hotdog. It is very good and I do not want to work on it any more. You can hire someone else." - Jian-Yang, [Silicon Valley](https://www.youtube.com/watch?v=ACmydtFDTGs)
# 
# Sounds simple enough, right? It's actually a little more complex, referring to the theory behind convolutional neural networks (CNN) and their applications in image classification. However, Keras makes it relatively easy to setup and evaluate a CNN!  
# 
# The objective of this project is simple: given a picture, does it contain a hot dog or not?
# 
# ## Data
# 
# This data was extracted from the Food 101 dataset. A full version of the dataset is available [here](https://www.kaggle.com/dansbecker/food-101). This is a binary classification task, while not very interesting a similar approach can be taken to extend this to multiple food categories. Note that multiple terms can refer to what is essentially the same object, in this case the difference appears to primarily be regional: a frankfurter, wienerwurst or hot dog. Thanks to [DanB](https://www.kaggle.com/dansbecker) for extracting the dataset ([kaggle](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog)) 
# 
# To simplify this project, the data is completely balanced: 50% hot dog images and 50% not hot dog images. We will then use 498 images to train our model, and 500 labeled images to test it on.
# 
# **Data augmentation**:  
# Neural networks perform much better when fed large amounts of data, and we only have 500 images for each class. We could just download more images from a source like ImageNet, but I will just augment this data set: add random rotation and or blur to our existing images to get 20,000 images for each class. **I also resize the images to 32 x 32, normalizeded them and performed histogram [equalization](https://en.wikipedia.org/wiki/Histogram_equalization)[](http://).**

# In[21]:


def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))
    
    
def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    angle = np.random.randint(0, 360)
    img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        img = loadBlurImg(path, imgSize)        
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize, hotdogs, notHotdogs):    
    imgSize = (img_size, img_size)
    xHotdog, yHotdog = loadImgClass(hotdogs, 0, classSize, imgSize)
    xNotHotdog, yNotHotdog = loadImgClass(notHotdogs, 1, classSize, imgSize)
    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")
    
    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)
    
    return X, y

def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    
    images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
    return images

def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape + (1,)) 
    return images

def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)


# In[3]:


from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

size = 32
classSize = 20000


hotdogs = glob.glob('../input/seefood/train/hot_dog/**/*.jpg', recursive=True) + glob.glob('../input/seefood/test/hot_dog/**/*.jpg', recursive=True) 

notHotdogs = glob.glob('../input/seefood/train/not_hot_dog/**/*.jpg', recursive=True) + glob.glob('../input/seefood/test/not_hot_dog/**/*.jpg', recursive=True)

scaled_X, y = loadData(size, classSize, hotdogs, notHotdogs)
scaled_X = preprocessData(scaled_X)
y = to_categorical(y)


n_classes=2
print("y shape", y.shape)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                    test_size=0.2, 
                                                    random_state=rand_state)

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)
print("Test shape X:", X_test.shape)
print("Test shape y: ", y_test.shape)

inputShape = (size, size, 1)


# In[4]:


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# # Convolutional Neural Network (CNN)  
# 
# I am using the architecture (network) outlined [here](https://www.kaggle.com/bugraokcu/cnn-with-keras).

# In[17]:


import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=inputShape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

start = time.time()

model.summary()
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='model.h5', monitor='val_acc', save_best_only=True)]

history = model.fit(X_train, y_train,
                      batch_size=32,
                      epochs=100, 
                      callbacks=callbacks,
                      verbose=0,
                      validation_data=(X_test, y_test))

end = time.time()
print('Execution time: ', end-start)

plot_history(history)


# I made a few tweaks to the model, specifically:
#   1. [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) optimization algorithm
#   2. learning rate to 0.0001
#   3. Initialize weights using [He Normal](https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)
#   4. early stopping
# 
# Earlier models that I built (using this architecture) suffered from really bad overfitting, but rescaling the input images, and augmenting the data to get 40 times as many images fixed this problem.
# 
# Our best model occurred at epoch 95 and achieved an accuracy of **97.6**%. This model also slightly underfit the data (not such a bad thing here). With GPU support enabled, training took 631 seconds (for 100 epochs). In comparison, it took 2376 seconds to train the same model for 60 epochs on my local machine (without CUDA support), a speedup of 6.2X.

# # Results  
# 
# We are simply going to evaluate this model on the original test set (500 of each class).

# In[40]:


hotdogs = glob.glob('../input/seefood/test/hot_dog/**/*.jpg', recursive=True) 
notHotdogs = glob.glob('../input/seefood/test/not_hot_dog/**/*.jpg', recursive=True)

scaled_X_test, y_test = loadData(size, 250, hotdogs, notHotdogs)
scaled_X_test = preprocessData(scaled_X_test)

#get the predictions for the test data
predicted_classes = model.predict_classes(scaled_X_test)

# setup the true classes: just 250 hotdogs followed by 250 not hotdogs
y_true = np.concatenate((np.zeros((250,)), np.ones((250,))))
from sklearn.metrics import classification_report
print(classification_report(y_true, predicted_classes, target_names=['hotdog', 'not hotdog']))

