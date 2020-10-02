#!/usr/bin/env python
# coding: utf-8

# # Classifying house images #
# 
# 
# Goal: Our goal is to classify the unknown images into clean house or messy house label. For this purpose we will use Keras Sequential model with Binary Crossentropy and sigmoid activation as we have only two target categories. 
# 
# **Contents**:
# 
# 1. [Reading the data](#Reading-the-data)
# 2. [Verifying the images](#Verifying-the-images)
# 3. [Creating train , test and validation set](#Creating-train-,-test-and-validation-set)
# 4. [Creating the Sequential model](#Creating-the-Sequential-model)
# 5. [Results](#Results)
# 
# 
# 
# Let's start by importing the libraries.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import os
import cv2
from sklearn import preprocessing
from pathlib import Path


# ## Reading the data ##

# We will store the path of train , test and validation data in seperate list. Here, all the data store in "clean" directory will have the same label which is : clean, and all the data store in "messy" directory will have the same label: messy. 
# 
# Let's represent clean with 0, and messy with 1.

# In[ ]:


# storing train images path and label

train_path = []
label_train = []

path_train = "../input/messy-vs-clean-room/images/train"

for filename in os.listdir(path_train+"/clean/"):
    train_path.append(path_train + "/clean/" + filename)
    label_train.append(0)

for filename in os.listdir(path_train+"/messy"):
    train_path.append(path_train + "/messy/" + filename)
    label_train.append(1)

print("Number of train images: ", len(train_path))


# In[ ]:


# storing validation images path and label

val_path = []
label_val = []

path_val = "../input/messy-vs-clean-room/images/val"

for filename in os.listdir(path_val+"/clean"):
    val_path.append(path_val + "/clean/" + filename)
    label_val.append(0)
    
for filename in os.listdir(path_val+"/messy"):
    val_path.append(path_val + "/messy/" + filename)
    label_val.append(1)
    
print("Number of validation images: ", len(val_path))


# In[ ]:


# storing test images path, it doesn't have any label

test_path = []

for filename in os.listdir("../input/messy-vs-clean-room/images/test/"):
    test_path.append("../input/messy-vs-clean-room/images/test/" + filename)

print("Number of validation images: ", len(test_path))


# ## Verifying the images ##
# 
# 
# Displaying the one image from each train test and validation data to verify if the images are loaded properly or not.

# In[ ]:


# checking train path
image = cv2.imread(train_path[0]) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# the first image bleongs to clean directory under train
plt.imshow(image)
plt.title("Clean", fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


# checking validation path
image = cv2.imread(val_path[0]) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# the first image bleongs to clean directory under validation
plt.imshow(image)
plt.title("Clean", fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


# checking test path
image = cv2.imread(test_path[0]) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# we dont know which image belongs to which category
plt.imshow(image)
plt.title("unknown", fontsize = 20)
plt.axis('off')
plt.show()


# ## Creating train , test and validation set ##

# We will now create training , testing and validation data set from the retrieved path. We will resize the images so that every image has same shape. Also, the images must be convereted to numpy array before feeding them inside keras model.

# In[ ]:


X_train = []
X_test = []
X_val = []

# reading images for train data
for path in train_path:
    
    image = cv2.imread(path)        
    image =  cv2.resize(image, (100,100))    
    X_train.append(image)
    
# reading images for test data
for path in test_path:
    
    image = cv2.imread(path)        
    image =  cv2.resize(image, (100,100))    
    X_test.append(image)

# reading images for validation data
for path in val_path:
    
    image = cv2.imread(path)
    image =  cv2.resize(image, (100,100))    
    X_val.append(image)


X_test = np.array(X_test)
X_train = np.array(X_train)
X_val = np.array(X_val)


# In[ ]:


print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of X_val: ", X_val.shape)


# In[ ]:


X_train[:2]


# The shape (192, 100, 100, 3) means 192 train images, each with size 100 * 100 and 3 channels - BGR. The array shows the pixel values stored from the image. Each pixel value is in the range 0-255 . Let's divide each pixel value with 255 to get a range from 0-1. 

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

X_train /= 255
X_test /= 255
X_val /= 255


# In[ ]:


# the shape is not going to change
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of X_val: ", X_val.shape)


# Creating y_train and y_val (validation) from the labels list

# In[ ]:


# creating numpy array from the labels list
y_train = keras.utils.to_categorical(label_train, 2)
y_val = keras.utils.to_categorical(label_val, 2)


# In[ ]:


y_val


# In[ ]:


# displaying the shape
print("Shape of y_train: ", y_train.shape)
print("Shape of y_val: ", y_val.shape)


# ## Creating the Sequential model ##

# Now we will create our keras sequential model. How we are creating it is specified in the comments

# In[ ]:


model = Sequential()

# input shape for first layer is 100,100,3 -> 100 * 100 pixles and 3 channels
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3), activation="relu"))

# adding 32 nodes in the second layer
model.add(Conv2D(32, (3, 3), activation="relu"))

# maxpooling will take highest value from a filter of 2*2 shape
model.add(MaxPooling2D(pool_size=(2, 2)))

# it will prevent overfitting
model.add(Dropout(0.25))

# adding more layers similarly
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

# activation function is sigmoid for the binary data
model.add(Dense(2, activation="sigmoid"))

# compiling the model
model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

model.summary()


# We have our model, training data and validation data. Let's train the model now.

# In[ ]:


# training the model
history = model.fit(
    X_train,
    y_train,
    batch_size=30,
    epochs=150,
    validation_data=(X_val , y_val),
    shuffle=True
)


# ## Results ##

# Let's evaluate the results. First we will see how much accuracy we obtained for the train and validation data and then we will check the loss

# In[ ]:


# displaying the model accuracy
plt.plot(history.history['accuracy'], label='train', color="red")
plt.plot(history.history['val_accuracy'], label='validation', color="blue")
plt.title('Model accuracy')
plt.legend(loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[ ]:


# displaying the model loss
plt.plot(history.history['loss'], label='train', color="red")
plt.plot(history.history['val_loss'], label='validation', color="blue")
plt.title('Model loss')
plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# ### Note:
# 
# As the `we dont have y_test or labels for the unknown data`, we `can't find the test accuracy`. Although as we have only `10 images` in test data, we can manually label it but still it no good. `One incorrect` label will `decrease the score by 10%`. So let's just predict the label instead of finding the accuracy.

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


pred


# Displaying the results for all 10 test data

# In[ ]:


fig, axs= plt.subplots(2,5, figsize=[24,12])


count=0
for i in range(2):    
    for j in range(5):  
        
        img = cv2.imread(test_path[count])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        txt = "clean prob: {:.4}% \n messy prob: {:.4}%".format( 100*pred[count][0], 100*pred[count][1])
        
                
        axs[i][j].imshow(img)
        axs[i][j].set_title(txt, fontsize = 14)
        axs[i][j].axis('off')

        count+=1
        
plt.suptitle("All predictions are shown in title", fontsize = 18)        
plt.show()


# ## Thank you ##

# In[ ]:




