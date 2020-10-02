#!/usr/bin/env python
# coding: utf-8

# # Sign Language Digit Classifier with CNN
# 
# In this notebook I will apply convolutional neural network (CNN) algorithm to a deep learning model in order to classify sign language digits from photographs.
# 
# In the data set, 218 students are used for photographing 10 sign language digits (Between the number 0 and 9) so there should be 2180 images. But the data set consists of 2062 images. Probably some of the images are removed by the data supplier bacause of inproper quality. So there should be around 200 images for each digit. 
# 
# Images are in grayscale (black and white) and each image has 64 x 64 pixels of width and height size.
# 
# Images are already saved as numpy arrays ('X.npy'). So I didn't have to load jpeg images and convert them to numpy arrays. 
# 
# Labels of the images are also saved as a different numpy array file ('Y.npy') and already encoded. So I dind't have to apply categorical encoding for the labels. 
# 
# **It is important to set a goal and draw the work plan accordingly. The aim of this study is to achieve a good classification model with a validation accuracy of at least 0.90.**
# 
# **WORK PLAN:**
#     1. Loading the data sets
#     2. Reshaping data
#     3. Splitting data sets into train and test
#     4. Applying data augmentation to x_train data set
#     5. Creating a CNN model & defining optimizer
#     6. Compiling the model
#     7. Defining epoch and batch sizes
#     8. Fitting the model
#     9. Evaluating the predictions by loss function, validation accuracy and confusion matrix.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1. Loading the Data Sets

# In[ ]:


x_data = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')
y_data = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')

print('Shape of x data: ', x_data.shape)
print('Shape of y data: ', y_data.shape)


# Have a look at some of the images:

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(x_data[0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(x_data[100], cmap='gray')
plt.title('Example Images')
plt.subplot(1,3,3)
plt.imshow(x_data[500], cmap='gray');


# # 2. Reshaping Data
# 
# In order to use the datas with keras library. I should have x_data as (2062, 64, 64, 1) and y_data as (2062, 10) matrix formats.
# 
#     Shape of x_data is (2062, 64, 64, 1), where:
#         * 2062 is the number of images
#         * 64, 64 are the pixel sizes for width and height
#         * 1 is the channel value which represents the grayscale
# 
#     Shape of y_data is (2062, 10), where:
#         * 2062 is the number of images
#         * 10 is the number of classes

# In[ ]:


# numpy reshape function is used to change matrix format:
x_data = x_data.reshape(-1, 64, 64, 1)
print('New shape of x_data: ', x_data.shape)

# y_data is already in proper matrix format:
print('Shape of y_data: ', y_data.shape)


# ## 3. Train and Test Split

# In[ ]:


# Using 80% of the data for training and 20% for testing. 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state = 1)

print('Shape of x_train: ', x_train.shape)
print('Shape of y_train: ', y_train.shape)
print('....')
print('Shape of x_test: ', x_test.shape)
print('Shape of y_test: ', y_test.shape)


# # 4. Data Augmentation
# 
# This data set doesn't have a great number of samples. But I can increase my training data by apply data augmentation. Some ramdom images from the x_train will be rotated, zoomed and shifted. I didn't use flipping the images because it wouldn't be a proper decision for this data set.   

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
            rotation_range = 5,        # 5 degrees of rotation will be applied
            zoom_range = 0.1,          # 10% of zoom will be applied
            width_shift_range = 0.1,   # 10% of shifting will be applied
            height_shift_range = 0.1)  # 10% of shifting will be applied

train_gen.fit(x_train)


# # 5. Creating Model
# 
# I will create a CNN and fully connected neural network (NN) model by intuition. Model plan will be like:
# 
#     model plan = conv --> max pool --> dropout --> conv --> max pool --> dropout --> flatten --> fully connected
# 
# Where:
# 
#     conv            : Filters the image features using kernels
#     max pool        : Extracts the most important feature in a defined matrix (pool_size)
#     dropout         : Helps to eliminate overfitting by randomly choosing and not using the nodes
#     flatten         : Flattens the data into a vector, which will be the input vector for NN
#     fully connected : Means feeding input features to the designed NN 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# Creating model structure
model = Sequential()
# Adding the first layer of CNN
model.add(Conv2D(filters=20, kernel_size=(4,4), padding='Same', activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.15))
# Adding the second layer of CNN
model.add(Conv2D(filters=30, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.15))
# Flattening the x_train data
model.add(Flatten()) 
# Creating fully connected NN with 4 hidden layers
model.add(Dense(220, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))


# Optimizer will be used for compiling the model. Optimizer is used to define our model's learning rate. Learning rate is an important hyperparameters which effects two things:
# 
# a. Finding the smallest cost function (If it is a big value, model can miss to find the lowest value of cost function while applying gradient descent)
# 
# b. Speed of the model (If it is a really small value, model will be slow in learning)

# In[ ]:


# Defining the optimizer

from keras.optimizers import Adam

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99)


# # 6. Compiling Model
# 
# Three parameters are important for compiling the model:
# 
# a. Choosing the defined optimizer
# 
# b. Choosing loss function according to the application (Because this study is categorical classification 'categorical_crossentropy' is chosen
# 
# c. Choosing the parameter how the model is going to evaluate its learning parameter (Which will be accuracy for this study)

# In[ ]:


model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])


# # 7. Defining Epoch and Batch Size
# 
# We have 1649 training images and we will divide them into batches of 100 images. So it will take around 16 iterations to complete 1 epoch. This will repeat 25 times.

# In[ ]:


batch_size = 100
epochs = 25


# # 8. Fitting the Model

# In[ ]:


history = model.fit_generator(train_gen.flow(x_train, y_train, batch_size = batch_size), 
                                                  epochs = epochs, 
                                                  validation_data = (x_test, y_test), 
                                                  steps_per_epoch = x_train.shape[0] // batch_size)


# The model is trained and the it reached to a validation accuracy of 0.92 which is just right for my initial aim. 
# 
# Now let's see the progress of the model learning and compare the prediction with the real labels of the test data.

# # 9. Evaluation of the Predictions
# 
# Normally I should have dived the data set into train and test. 
# 
# And then divide the train data into train and validation data. 
# 
# I should use the train and validation data for my model learning and use test data for predicting and evaluating with the true labels.
# 
# But I didn't choose this path. Because:
# 
# 1. I don't have big number of samples. So I did't want to cut some of my training data for validation.
# 2. This is a study for my learning (And hopefully those who read this notebook)

# In[ ]:


# Visiualize the validation loss and validation accuracy progress:

plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(history.history['val_loss'], color = 'r', label = 'validation loss')
plt.title('Validation Loss Function Progress')
plt.xlabel('Number Of Epochs')
plt.ylabel('Loss Function Value')

plt.subplot(1,2,2)
plt.plot(history.history['val_accuracy'], color = 'g', label = 'validation accuracy')
plt.title('Validation Accuracy Progress')
plt.xlabel('Number Of Epochs')
plt.ylabel('Accuracy Value')
plt.show()


# As one can see that by time the model gets better for predicting labels (The decrease of the loss function and the increase of the accuracy).
# 
# Now let's see the comparison of predictions and the real labels by using confusion matrix.

# In[ ]:


# Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# First of all predict labels from x_test data set and trained model
y_pred = model.predict(x_test)

# Convert prediction classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis = 1)

# Convert validation observations to one hot vectors
y_true_classes = np.argmax(y_test, axis = 1)

# Create the confusion matrix
confmx = confusion_matrix(y_true_classes, y_pred_classes)
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(confmx, annot=True, fmt='.1f', ax = ax)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show();


# The diagonal line in the middle shows how many of the predictions and the true labels match together. 
# 
# One can see the model mixed 8 times, the image represents the digit 0 and guessed those as digit 5.
# 
# This is the conclusion of my study. 
# 
# Lastly I would like to thank to my teachers:
# * The DataI Team https://www.kaggle.com/kanncaa1 , https://dataiteam.com , https://www.udemy.com/user/datai-team/
# * Mustafa Vahit Keskin https://www.veribilimiokulu.com/author/mvk/ , https://www.udemy.com/user/mustafa-vahit-keskin/
# 
# for helping me progress in data science, machine learning and deep learning subjects. 
# 
# I hope you enjoyed my notebook, if so please don't forget to upvote.
