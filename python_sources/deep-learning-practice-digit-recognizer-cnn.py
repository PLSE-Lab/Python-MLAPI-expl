#!/usr/bin/env python
# coding: utf-8

# In[25]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **LOAD DATASET**

# In[26]:


# read train dataset
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head(10)


# In[27]:


# read test dataset

test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head(10)


# In[28]:


# put labels into y_train variable
Y_train = train["label"]
# Drop label column
X_train = train.drop(labels = ["label"],axis=1)


# In[29]:


# visualize number of digits classes

plt.figure(figsize=(15,10))
sns.countplot(Y_train, palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()


# In[30]:


# plot some samples

image = X_train.iloc[5].as_matrix()
image = image.reshape((28,28))
plt.imshow(image, cmap="gray")
plt.title(train.iloc[5,0])
plt.axis("off")
plt.show()


image2 = X_train.iloc[7].as_matrix()
image2 = image2.reshape((28,28))
plt.imshow(image2, cmap="gray")
plt.title(train.iloc[7,0])
plt.axis("off")
plt.show()


# **Normalization, Reshape and Label Encoding**
# 
# * Normalization
#     *  We perform a grayscale normalization to reduce the effect of illumination's differences.
#     *  If we perform normalization, CNN works faster
# * Reshape
#     *  Train and test images(28*28)
#     *  We rehape all data to 28*28*1 3D matrices
#     *  Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel
# * Label Encoding
#     *  Encode labels to one got vectors
#         * 2 => [0,0,1,0,0,0,0,0,0,0]
#         * 4 => [0,0,0,0,1,0,0,0,0,0]

# In[31]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ", test.shape)


# In[32]:


# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ", test.shape)


# In[33]:


# Label Encoding
from keras.utils.np_utils import to_categorical     # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)


# **Train Test Split**

# In[34]:


# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state = 42)
print("x_train shape: ",X_train.shape)
print("x_val shape: ",X_val.shape)
print("xytrain shape: ",Y_train.shape)
print("y_val shape: ",Y_val.shape)


# In[35]:


# Some examples
plt.imshow(X_train[2][:,:,0], cmap="gray")
plt.axis("off")
plt.show()


# **CONVOLUTION NEURAL NETWORK**
# *       CNN is used for image classification, object detection
# 
# 
# **What is Convolution Operation**
# * We have some images and featrues detector(3*3)
# * Feature detector doen not need to be 3 by 3 matrix. It can be 5 by 5 or 7 by 7.
# * Feature detector = kernel = filter
# * Feature detector detecs features like edges or convex shapes.
# * feature map = convolved feature
# * Stride = navigating in input image
# * We reduce the size of image. This is important because code runs faster.However, we lost information.
# * We create multiple feature maps because we use multiple feature detectors(filters)****

# **CREATE MODEL**

# In[36]:


# conv => max pool => dropout => max pool => dropout => fully connected(2 Layer)
# Dropout is a tecnique where randomly selected neurons are ignored during training

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters=8, kernel_size=(5,5), padding="Same", activation="relu", input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters=16, kernel_size= (3,3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[37]:


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# In[38]:


# Compile the model
model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


# **Epochs and Batch Size**
# * Say you have a dataset of 1o examples(or samples). You have a **batch size** of 2, and you have specified you want the algorithm to run for 3 **epochs**.Therefore, in each epoch, you have 5 **batches**(10/2=5). Each batch gets passed through the algorithm, therefore you have 5 iterations **per epoch**.
# 

# In[39]:


epochs = 15 # for better result increase the epochs
batch_size = 150


# In[40]:


# Data argumentation
# In order to recognize the image in different ways, it is prevented from overfitting by applying operations such as rotation and zoom.

datagen = ImageDataGenerator(
        featurewise_center=False,    # set input mean to 0 over the dataset
        samplewise_center=False,     # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,   # divide each input by its std
        zca_whitening=False,    # dimension reduction
        rotation_range=0.5,     # randomly rotate images in the range 5 degrees
        zoom_range=0.5,         # randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5, # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)    # randomly flip images

datagen.fit(X_train)


# In[41]:


# Fit the Model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs=epochs, validation_data=(X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# **Evaluate the model**
# * Test loss visualization
# * Confusion matrix

# In[42]:


# plot the loss and accuracy curves fro training and validation

plt.plot(history.history["val_loss"], color="b", label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[43]:


# confusion matrix

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis=1)
# Convert validation observation to one hot vectors
Y_true = np.argmax(Y_val,axis=1)
# compute the confusion amtrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt=".1f",ax=ax)
plt.xlabel("Prediction Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[49]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[51]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
