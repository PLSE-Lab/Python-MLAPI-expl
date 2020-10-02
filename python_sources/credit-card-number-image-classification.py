#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2 as cv
import os
import random
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[ ]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# In[ ]:


imgs = []
max_img_count = 1000
for i in range(10):
    digit = load_images_from_folder('/kaggle/input/credit-card-number-images/Credit Card Number Dataset/'+str(i))
    imgs.append(digit)


# In[ ]:


imgs_resized = [] #holds resized images
for number in imgs:
    digit_resized = []
    for img in number:
        resized = cv.resize(img, dsize=(58, 85), interpolation=cv.INTER_CUBIC) #average image dimensions = 85.14011299435029 (height) x 58.110734463276835 (width)
        digit_resized.append(resized)
    imgs_resized.append(digit_resized)


# In[ ]:


imgs_equal = [] #holds equal number of images for all categories(0-9)
for number in imgs_resized:
    count=0
    while(count != max_img_count):#generating filler data
        rand_img = random.sample(number, 1)[0]
        rows, cols, _ = rand_img.shape
        matrix = cv.getRotationMatrix2D((cols,rows),random.randint(-10,10),1)
        rotated = cv.warpAffine(rand_img,matrix,(cols,rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        imgs_equal.append(rotated)
        count+=1


# In[ ]:


def onehotencode(index, n):
    return [1.0 if i == index else 0.0 for i in range(n)]


# In[ ]:


labels = []
for i in range(10):
    for j in range(max_img_count):
        labels.append(onehotencode(i, 10))


# In[ ]:


data = np.asarray(imgs_equal)/255 #VERY IMPORTANT TO NORMALIZE PIXELS FROM 0-255 to 0-1
labels = np.asarray(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[ ]:


input_layer = tf.keras.layers.Input((85, 58, 3))
hidden_layer_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3),  activation=tf.keras.activations.relu)(input_layer)
hidden_layer_2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(5,5),  activation=tf.keras.activations.relu)(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Conv2D(filters=10, kernel_size=(7,7),  activation=tf.keras.activations.relu)(hidden_layer_2)
hidden_layer_4 = tf.keras.layers.Flatten()(hidden_layer_3)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(hidden_layer_4)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(tf.keras.optimizers.SGD(0.01), tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train, y_train, 70, 10, validation_data =[x_test, y_test], verbose = 1)


# In[ ]:


if(not os.path.exists("credit_card_model.h5")):#if file doesn't exist
    model.save('credit_card_model.h5')  # creates a HDF5 file 'my_model.h5'

prev_model = tf.keras.models.load_model('credit_card_model.h5')
prev_validation_accuracy = prev_model.evaluate(data, labels, verbose=0)[1]
curr_validation_accuracy = model.evaluate(data, labels, verbose=0)[1]

if(curr_validation_accuracy > prev_validation_accuracy):
    model.save('credit_card_model.h5')  # update file with best model


# In[ ]:


cnn = tf.keras.models.load_model('credit_card_model.h5')

for i, img in enumerate(data): 
    img = np.expand_dims(img, axis=0)
    guess = np.argmax(cnn.predict(img)) #img must be normalized from 0-1 before feeding into predict()
    actual = np.argmax(labels[i])
    print(actual, guess)

