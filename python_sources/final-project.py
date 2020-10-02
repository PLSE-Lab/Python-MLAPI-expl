#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Final Project
# ## Date: May 5th 2019
# ## Author:  [Ching Ching Huang]
# ## Discussants: [TA: Tony Zhang, Professor: Ethan Meyers]

# ## Kaggle competition: Aerial Cactus Identification
# 
# Aerial Cactus Identification: https://www.kaggle.com/c/aerial-cactus-identification
# 
# Human activities such as logging, mining and climate change have been impacting protected natural areas in Mexico. Researchers have started a project, VIGIA, to build a system for autonomous surveillance of protected areas. To do so, the first step is recognizing plants or vegetation in the protected area. The goal of the challenge is to correctly identify a specific type of cactus through aerial images.
# 
# Kaggle provides all the data that are required to solve the challenge.
# - train.csv file: containing the names and the labels of the images. If the image contains a cactus, the label is 1. Otherwise, the label is 0. 
# - train folder: containing 17,500 images which are the images to train the machine.
# - test folder: containing 4000 images which are the images that we want to predict.
# 
# Each image in the train and test folder has width of 32 pixels and height of 32 pixels. 
# 
# There are around 400 people who submitted to this challenge. Around 50 people scored 100% on the leader board and around 400 people scored above 95%. I scrolled through submitted kernels and noticed most people are using Convolutional Neural Network (CNN) algorithm to solve the challenge; however, every kernel has different layers for the CNN algorithm.

# ## Setup
# 
# In order to try different algorithms, we need to load in and process the data to an expected form. We know train.csv contains the image file names and the labels. Hence, we get the path with the file name and add it to train_img list. Each image is converted to a 3D array while loading the data with matplotlib.image package. train_lb stores the labels of the images in train_img. After loading the data from the train folder, we split the train dataset to test and train set with scikit-learn package. We use the similar method to load images from test folder as test dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split

cactus_label = pd.read_csv('../input/train.csv')

#read in training set
train_img = []
train_lb = []
for i in range(len(cactus_label)):
    row = cactus_label.iloc[i]
    fileName = row['id']
    train_lb.append(row['has_cactus'])
    path = "../input/train/train/{}".format(fileName)
    im = mpimg.imread(path)
    train_img.append(im)
    
X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 
X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


import os
test_img = []
sample = pd.read_csv('../input/sample_submission.csv')
folder = '../input/test/test/'
                   
for i in range(len(sample)):
    row = sample.iloc[i]
    fileName = row['id']
    path = folder + fileName
    img = mpimg.imread(path)
    test_img.append(img)
                     
test_img = np.asarray(test_img)


# Before we dive into solving the challenge, let's take a closer look at the data. In the train dataset, 75% of the images contain cacti. We notice that the proportion is very unbalanced. 

# In[ ]:


cactus_label['has_cactus'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
# Data to plot
labels = 'Has Cactus', 'No Cacuts'
sizes = [13136, 4364]
colors = ['yellowgreen', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# ## K Nearest Neighbor (KNN) Algorithm
# 
# KNN is one of the most commonly used classification algorithms. We start with KNN algorithm to make sure the challenge is not too easy. That means if the accuracy rate with KNN is above 80%, then the challenge doesn't need to be solved with other more advanced algorithms. 
# 
# In order to apply the KNN algorithm, we need to modify or process the data before training the machine. When we load the images, each image is stored as a 3D array. The KNN algorithm only takes 1D arrays. Therefore, we need to convert the images to 1D arrays by unstacking each row. Fortunately, there's a flatten method in numpy package to do the work for us. 

# In[ ]:


'''
Convert 3D ararys to 1D array
Paramter: a list of 3D images
Return: a list of 1D images
'''
def imageToFeatureVector(images):
    flatten_img = []
    for img in images:
        data = np.array(img)
        flattened = data.flatten()
        flatten_img.append(flattened)
    return flatten_img


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
start = time.time()

X_train_flatten = imageToFeatureVector(X_train)
X_test_flatten = imageToFeatureVector(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_flatten, y_train) 
score = knn.score(X_test_flatten, y_test)

end = time.time()
print("The run time of KNN is {:.3f} seconds".format(end-start))
print("KNN alogirthm's test score is: {:.3f}".format(score))


# ## Balance Data
# 
# Since we learned the unbalanced data is affecting the prediction, we need to reduce the number of images containing cacti. We will modify the data by loading the same number of images with cacti and without cacti to balance the data. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
cactus_label = pd.read_csv('../input/train.csv')

#read in training set
train_img = []
train_lb = []
has_cactus = 0
no_cactus = 0
for i in range(len(cactus_label)):
    row = cactus_label.iloc[i]
    fileName = row['id'] 
    path = "../input/train/train/{}".format(fileName)
    im = mpimg.imread(path)
    if row['has_cactus'] == 1 and has_cactus < 4364:
        has_cactus+= 1
        train_lb.append(row['has_cactus'])
        train_img.append(im)
    elif row['has_cactus'] == 0 and no_cactus < 4364:
        no_cactus += 1
        train_lb.append(row['has_cactus'])
        train_img.append(im)


    
X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 
X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


import matplotlib.pyplot as plt
# Data to plot
labels = 'Has Cactus', 'No Cacuts'
sizes = [train_lb.count(1), train_lb.count(0)]
colors = ['yellowgreen', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
start = time.time()

X_train_flatten = imageToFeatureVector(X_train)
X_test_flatten = imageToFeatureVector(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_flatten, y_train) 
score = knn.score(X_test_flatten, y_test)

end = time.time()
print("The run time of KNN is {:.3f} seconds".format(end-start))
print("KNN alogirthm's test score is: {:.3f}".format(score))


# ## Discussion
# 
# Before balancing the data, KNN has an accuracy of approximatley 30% which means we can apply other algorithms to solve this challenge. It is not surprising that the accuracy is so low. There are 75% of the training images containing cactus. The unbalanced proportion of training data may lead to poor performance of KNN algorithm. Since there are 17,500 images and each image has 1,024 pixels, it makes sense that it took 400 seconds, almost 7 minutes, to execute. 
# 
# After balancing the data, the test score increased to 49%. Since there are less images in the dataset, it only took 76 seconds to execute. From the modification, we know that unbalanced data was affecting the prediction.

# ## Support Vector Machines (SVM)
# 
# In this section, we apply SVM using a linear kernel (LinearSVC) and a radial basis function (RBF) kernel. In order to find the best parameters for RBF kernel, we use GridSearchCV object on the training data to find the best values for free parameters. This part is similar to worksheet 6. To see whether normalizing the data will enhance the performance, we compare the result from training the machine with original data to the normalized data. 

# In[ ]:


from sklearn.svm import LinearSVC

start = time.time()
linearKernel = LinearSVC().fit(X_train_flatten, y_train)
score = linearKernel.score(X_test_flatten,y_test)
end = time.time()

print("The run time of Linear SVC is {:.3f} seconds".format(end-start))
print("Linear SCV alogirthm's test score is: {:.3f}".format(score))


# In[ ]:


# try normalizing the features...
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
start = time.time()
scaler.fit(X_train_flatten)
X_test_normalized = scaler.transform(X_test_flatten)
X_train_normalized = scaler.transform(X_train_flatten)

linearKernel = LinearSVC().fit(X_train_normalized, y_train)
score = linearKernel.score(X_test_normalized,y_test)
end = time.time()
print("The run time of Linear SVC with normalized features is {:.3f} seconds".format(end-start))
print("Linear SCV with normalized features has test score of: {:.3f}".format(score))


# ## Discussion
# 
# According to the result, Linear SVC with original data has a test score of 81.9% and Linear SVC with normalized data has a test score of 83%. Normalizing the features barely improved the performance. Each GridSearchCV requires more than 3 hours to execute; however, each session on Kaggle is only 6 hours. Other algorithm requires some time to execute too. Therefore, I didn't execute both RBF kernels.

# ## Convolutional Neural Network (CNN) Algorithm
# 
# The last algorithm that I'd like to apply is the CNN algorithm. In order to build a model that recognizes objects correctly, we need to add different layers. 
# 
# In the CNN model, I added three dense layers. The first layer is rectifying linear units with 128 units. The second layer is again rectifying linear units with 64 units. The last layer returns an output of softmax transformation.

# In[ ]:


import tensorflow as tf
start = time.time()
X_train_norm = tf.keras.utils.normalize(X_train, axis=1)
X_test_norm = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#add layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train 
model.fit(X_train_norm, np.array(y_train), epochs=10)

# Evaluate the model on test set
score = model.evaluate(X_test, np.array(y_test), verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

end = time.time()
print("The run time of CNN is {:.3f} seconds".format(end-start))


# ## CNN from other kernel with 99% accuracy
# 
# I've compared my CNN model with other CNN models from the kernel that have scores of 99%. I wonder why my CNN model is not scoring 99%. The code copied below from the kernel serves as the model for comparison.
# 
# The link to the kernel: https://www.kaggle.com/gabrielmv/aerial-cactus-identification-keras

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, DepthwiseConv2D, Flatten
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def create_model():
    model = Sequential()
        
    model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))
    
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))
    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    
    model.add(Dense(470, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation = 'tanh'))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    return model


# In[ ]:


model = create_model()

history = model.fit(X_train, 
            np.array(y_train), 
            batch_size = 128, 
            epochs = 8, 
            validation_data = (X_test, np.array(y_test)),
            verbose = 1)

predictions = model.predict(X_test, verbose = 1)

# Evaluate the model on test set
score = model.evaluate(X_test, np.array(y_test), verbose=0)
# Print test accuracy
print('Test accuracy:', score[1])


# ## Discussion
# 
# I have a test score of 73.6% from the CNN model I've created. The score is lower than I expected. Therefore, I compared my CNN model with another CNN model which has a 96.5% test score. I've noticed that there are more sophisticated layers such as Conv2D. In addition, the CNN model from the kernel scored 99% without balancing the data. These are important factors to consider for future research.

# ## Result
# 
# The result is presented below:
# 
# K Nearest Neighbor: 49.1% 90sec
# 
# Linear SVC: 81.9% 88sec
# 
# Linear SVC with normalized feature: 83% 52sec
# 
# Convolutional Neural Network: 73.6% 8sec
# 
# Hence, we predict the test dataset with SVM using Linear SVC with normalized data.

# In[ ]:


scaler = preprocessing.StandardScaler()
scaler.fit(X_train_flatten)
X_test_normalized = scaler.transform(X_test_flatten)
X_train_normalized = scaler.transform(X_train_flatten)
test_flatten = imageToFeatureVector(test_img)
test_normalized = scaler.transform(test_flatten)
linearKernel = LinearSVC().fit(X_train_normalized, y_train)
predictions = linearKernel.predict(test_normalized)
sample['has_cactus'] = predictions
sample.head()


# In[ ]:


sample.to_csv('sub.csv', index= False)


# ## Conclusion
# 
# From the result, we observed that after preprocessing the data, the run time for each algorithm is quite efficient. Even though CNN is the most efficient solution, the test score is lower than expected. According to the accuracy, SVM using linear SVC with normalized features is the most accuracte algorithm. In the future, I'd like to learn more about building a CNN model with multiple layers and the theory behind of it. In addition, I am interested in applying the same algorithm to identify different vegetation in the protected area. 

# ## Reflection
# 
# What have I learned?
# 
# This is my second time doing image classification. Unlike my first Kaggle challenge, I spent more time understanding and processing the data, which I found to be the crucial step of this project. For example, unbalanced data affects the performance of the model, thus, we need to ensure the data is balanced before fitting in any algorithms. In addition, since each image is stored in a 3D array, it is required to flatten the 3D arrays to feature vectos. To improve the performance, we can also normalize the features. 
# 
# What went well?
# 
# Processing and balancing the data went well for me. Since we've done something similar in worksheet 6, and the documentation for flattening the features with numpy package is very clear, the whole process went smoothly. 
# 
# What were difficult?
# 
# This is my first time working with the Convolutional Neural Network algorithm. For me, understanding and adding the layers were difficult. I read and followed the example code from the documentation to build my own CNN model. However, the result wasn't good enough. Therefore, I compared mine with other CNN models. From those, I learned that I wasn't adding enough layers. I had to have different layers and increase the number of layers. 
# 
# Running GridSearchCV was difficult for me too. Both GridSearchVC alogrithms took more than 6 hours to execute which exceeds the kaggle kernel session. 
# 
# How much time did I spend on this project?
# 
# It took me approximately 24 hours to complete the final project.
# 
# - Searching and understanding the challenge: 2 hours
# - Understanding and processing the data: 2 hours
# - KNN: 1 hours
# - SVM using Linears SVC: 3 hours
# - SVM using GridSearchCV: 4 hours
# - CNN: 5 hours
# - Write up: 3 hours
# - Meeting with Tony: 2 hours
# - Making slides for presentation: 2 hours

# ## Sources
# https://stackoverflow.com/questions/7755684/flatten-opencv-numpy-array
# 
# https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
# 
# https://www.tensorflow.org/guide/keras
# 
# https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
