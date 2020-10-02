#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer
# 
# ## A BEGINNER'S GUIDE
# 
# Using
# - Multi-layer Perceptron (MLP) Model
# - Convolutional Neural Network (CNN) Model
# - Keras Neural Network Library

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set() # setting seaborn default for plots

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.datasets import mnist

# for Multi-layer Perceptron (MLP) model
from keras.models import Sequential
from keras.layers import Dense

# for Convolutional Neural Network (CNN) model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# fix for issue: https://github.com/fchollet/keras/issues/2681
from keras import backend as K
K.set_image_dim_ordering('th')


# ## Loading Train and Test datasets

# In[ ]:


train = pd.read_csv('../input/train.csv')
print (train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
print (test.shape)
test.head()


# In[ ]:


y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
X_test = test

print (y_train.value_counts())


# In[ ]:


sns.countplot(y_train)


# In[ ]:


X_train.head()


# In[ ]:


# check for corrupted images in the datasets
# i.e. check if there are any empty pixel values
print (X_train.isnull().any().sum())
print (X_test.isnull().any().sum())


# ## Get values of data

# In[ ]:


X_train = X_train.values.astype('float32') # pixel values of all images in train set
y_train = y_train.values.astype('int32') # labels of all images
X_test = test.values.astype('float32') # pixel values of all images in test set


# ## Viewing shape and content of data

# In[ ]:


print (X_train.shape)
print (y_train.shape)


# In[ ]:


print (y_train[0])
print (X_train[0])


# ## Plotting images and their class values

# In[ ]:


plt.figure(figsize=[20,8])
for i in range(6):
    plt.subplot(1,6,i+1)
    # Here, we reshape the 784 pixels vector values into 28x28 pixels image
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))


# In[ ]:


# fix random seed for reproducibility
random_seed = 7
np.random.seed(random_seed)


# ## Normalizing input values
# 
# As we can see above, the pixel values for each image are gray scaled between 0 and 255. We now, normalize those values from 0-255 to 0-1.

# In[ ]:


# pixel values are gray scale between 0 and 255
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
print (X_train[1])


# ## Converting target variable values into one-hot format
# 
# The output/target variable is in the format 0 to 9. As this is a multi-class classification problem, we convert the output class values into one-hot format which is simply a binary matrix, i.e.
# 
# >value 0 will be converted to one-hot format as [1, 0, 0, 0, 0, 0, 0, 0, 0]
# >
# >value 1 will be converted to one-hot format as [0, 1, 0, 0, 0, 0, 0, 0, 0]
# >
# >value 2 will be converted to one-hot format as [0, 0, 1, 0, 0, 0, 0, 0, 0]
# >
# >and so on...
# 

# In[ ]:


print (y_train.shape)
print (y_train[0])


# In[ ]:


# one hot encode outputs
# note that we have new variables with capital Y
# Y_train is different than y_train
Y_train = np_utils.to_categorical(y_train)
num_classes = Y_train.shape[1]


# In[ ]:


print (y_train.shape, Y_train.shape)
print (y_train[0], Y_train[0])


# ## Splitting train dataset into training and validation set
# 
# We split the train dataset into two parts in 9:1 ratio. 90% will be the actual training set and the remaining 10% will be the validation/testing set. We train our model using the training set and test the accuracy of the model using the validation set.

# In[ ]:


# Split the entire training set into two separate sets: Training set and Validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, random_state=random_seed)


# In[ ]:


print (X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
num_pixels = X_train.shape[1]


# In[ ]:


print (Y_val)
# converting one-hot format of digits to normal values/labels
y_val = np.argmax(Y_val, 1) # reverse of to_categorical
print (y_val)
# Note that: capital Y_val contains values in one-hot format and small y_val contains normal digit values


# ## Define Simple Perceptron Model
# 
# Generally, neural networks have the following properties:
# - an input layer as a single vector
# - zero or multiple hidden layers after input layer
# - an output layer after hidden layers which represents class scores in classification problem
# - each neuron in a hidden layer is fully connected to all neurons in the previous layer
# - neurons in a single layer function independently and do not have any connection with other neurons of the same layer
# 
# A **single-layer perceptron model** is the simplest kind of neural network where there are only two layers: ***input layer*** and ***output layer***. The inputs are directly fed into the outputs via a series of weights. It's a **[feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network)** where the information moves in only one direction, i.e. forward direction from input nodes to output nodes.
# 
# A **multi-layer perceptron model** is the other kind of neural network where there are one or more hidden layers in between input and output layers. The information flows from input layer to hidden layers and then to output layers. These models can be of **feed-forward** type or they can also use **[back-propagation](https://en.wikipedia.org/wiki/Backpropagation)** method. In back-propagation, the error is calculated in the output layer by computing the difference of actual output and predicted output. The error is then distributed back to the network layers. Based on this error, the algorithm will adjust the weights of each connection in order to reduce the error value. This type of learning is also referred as **deep learning**.
# 
# We create a **simple neural network** model with **one hidden layer** with 784 neurons. Our input layer will also have 784 neurons as we have flattened out training dataset into a single 784 dimensional vector.
# 
# *softmax* activation is used in the output layer.
# 
# *adam* gradient descent optimizer is used to learn weights.

# In[ ]:


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ## Fit and Evaluate Model
# 
# The model is fit over 5 epochs/iteration. It takes a batch of 200 images in each iteration. Validation dataset is used for validation. The epochs may be increased to improve accuracy.
# 
# Finally, validation dataset is used to evaluate the model by calculating the model's classification accuracy.

# In[ ]:


model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=200, verbose=1)


# In[ ]:


model.summary()


# In[ ]:


scores = model.evaluate(X_val, Y_val, verbose=0)
print (scores)
print ('Score: {}'.format(scores[0]))
print ('Accuracy: {}'.format(scores[1]))


# ## Plot correctly and incorrectly predicted images
# 
# Let's plot some images which are correctly predicted and some images which are incorrectly predicted on our validation dataset.

# In[ ]:


# get predicted values
predicted_classes = model.predict_classes(X_val)


# In[ ]:


# get index list of all correctly predicted values
correct_indices = np.nonzero(np.equal(predicted_classes, y_val))[0]

# get index list of all incorrectly predicted values
incorrect_indices = np.nonzero(np.not_equal(predicted_classes, y_val))[0]


# In[ ]:


print ('Correctly predicted: %i' % np.size(correct_indices))
print ('Incorrectly predicted: %i' % np.size(incorrect_indices))


# In[ ]:


plt.figure(figsize=[20,8])
for i, correct in enumerate(correct_indices[:6]):
    plt.subplot(1,6,i+1)
    plt.imshow(X_val[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_val[correct]))
    
plt.figure(figsize=[20,8])
for i, incorrect in enumerate(incorrect_indices[:6]):
    plt.subplot(1,6,i+1)
    plt.imshow(X_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_val[incorrect]))


# ## Confusion Matrix

# In[ ]:


# we have digit labels from 0 to 9
# we can either manually create a class variable with those labels
# class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# or, we can take unique values from train dataset's labels
class_names = np.unique(y_train)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, predicted_classes)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = class_names
predicted_class_names = class_names

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (8,6))

#plt.subplot(121)
ax = sns.heatmap(df_cnf_matrix, annot=True, fmt='d')
ax.set_ylabel('True values')
ax.set_xlabel('Predicted values')
ax.set_title('Confusion Matrix in Numbers')

'''
plt.subplot(122)
ax = sns.heatmap(df_cnf_matrix_percent, annot=True)
ax.set_ylabel('True values')
ax.set_xlabel('Predicted values')
'''


# Out of 4200 validation data, we got the following correct and incorrect predictions by MLP model:
# - Correctly predicted: 4109
# - Incorrectly predicted: 91
# 
# The above confusion matrix heatmap shows that:
# - 9 values of digit 9 were predicted as 7.
# - 6 values of digit 3 were perdicted as 8.
# - 4 values of digit 4, 5 and 9 were predicted as 9, 3 and 4 respectively.
# 
# The accuracy of the model may improve if we increase the epoch/iteration number while fitting the model. Currently, it is set as 5. We can increase it to 10 and see the accuracy output.

# ## Improve Accuracy using Convolution Neural Network (CNN) Model
# 
# [Convolutional Neural Networks (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are similar to Multi-layer Perceptron Neural Networks. They are also made up of neurons that have learnable weights and biases. CNNs have been successfully applied to analyzing visual imagery. They are mostly being applied in image and video recognition, recommender systems and natural language processing.
# 
# A CNN consists of multiple hidden layers. The hidden layers are either *convolutional*, *pooling* or *fully connected*.
# 
# **Convolution layer:** Feature extraction is done in this layer. This layer applies convolution operation to the input and pass the result to the next layer. In the image classification problem, a weight matrix is defined in the convolution layer. A dot product is computed between the weight matrix and a small part (as the size of the weight matrix) of the input image. The weight runs across the image such that all the pixels are covered at least once, to give a convolved output.
# 
# > The weight matrix behaves like a **filter** in an image extracting particular information from the original image matrix. 
# >
# >A weight combination might be extracting edges, while another one might a particular color, while another one might just blur the unwanted noise.
# >
# >The weights are learnt such that the loss function is minimized similar to a Multi-layer Perceptron. 
# >
# >Therefore weights are learnt to extract features from the original image which help the network in correct prediction. 
# >
# >When we have multiple convolutional layers, the initial layer extract more generic features, while as the network gets deeper, the features extracted by the weight matrices are more and more complex and more suited to the problem at hand.
# >
# >*Reference: [Architecture of Convolutional Neural Networks (CNNs) demystified](https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/)*
# 
# ***Stride:*** While computing the dot product, if the weight matrix moves 1 pixel at a time then we call it a stride of 1. Size of the image keeps on reducing as we increase the stride value.
# 
# ***Padding:*** Padding one or more layer of zeros across the image helps to resolve the output image size reduction issue caused by *stride*. Initial size of the image is retained after the padding is done.
# 
# **Pooling layer:** Reduction in number of feature parameters is done in this layer. When the image size is too larger, then we need a pooling layer in-between two convolution layers. This layer helps to reduce the number of trainable parameters of the input image. The sole purpose of pooling is to reduce the spatial size of the image. This layer is also used to control overfitting. 
# - Max pooling: Uses maximum value from each of the cluster of the prior layer
# - Average pooling: Uses the average value from each of the cluster of the prior layer
# 
# **Fully connected layer:** This layer comes after convolution and pooling layers. This layer connects each neuron in one layer to every neuron in another layer. This is similar to the concept of layer connection of Multi-layer perceptron model. Error is computed in the output layer by computing the difference in actual output and predicted output. After that, back-propagation is used to update the weight and biases for error and loss reduction.

# ## Load train and test data
# 
# Let's again load the train and test datasets.

# In[ ]:


train = pd.read_csv('../input/train.csv')
print (train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
print (test.shape)
test.head()


# In[ ]:


y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
X_test = test


# ## Get values of data

# In[ ]:


X_train = X_train.values.astype('float32') # pixel values of all images in train set
y_train = y_train.values.astype('int32') # labels of all images
X_test = test.values.astype('float32') # pixel values of all images in test set


# ## View shape and content of data

# In[ ]:


print (X_train.shape)
print (y_train.shape)
print (X_train[1])


# ## Normalizing input values
# 
# As we can see above, the pixel values for each image are gray scaled between 0 and 255. We now, normalize those values from 0-255 to 0-1.

# In[ ]:


# pixel values are gray scale between 0 and 255
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
print (X_train[1])


# ## Converting target variable values into one-hot format
# 
# The output/target variable is in the format 0 to 9. As this is a multi-class classification problem, we convert the output class values into one-hot format which is simply a binary matrix, i.e.
# 
# >value 0 will be converted to one-hot format as [1, 0, 0, 0, 0, 0, 0, 0, 0]
# >
# >value 1 will be converted to one-hot format as [0, 1, 0, 0, 0, 0, 0, 0, 0]
# >
# >value 2 will be converted to one-hot format as [0, 0, 1, 0, 0, 0, 0, 0, 0]
# >
# >and so on...
# 

# In[ ]:


print (y_train.shape)
print (y_train[0])


# In[ ]:


# one hot encode outputs
# note that we have new variables with capital Y
# Y_train is different than y_train
Y_train = np_utils.to_categorical(y_train)
num_classes = Y_train.shape[1]


# In[ ]:


print (y_train.shape, Y_train.shape)
print (y_train[0], Y_train[0])


# ## Splitting train dataset into training and validation set
# 
# We split the train dataset into two parts in 9:1 ratio. 90% will be the actual training set and the remaining 10% will be the validation/testing set. We train our model using the training set and test the accuracy of the model using the validation set.

# In[ ]:


# Split the entire training set into two separate sets: Training set and Validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, random_state=random_seed)


# In[ ]:


print (X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
num_pixels = X_train.shape[1]


# In[ ]:


print (Y_val)
# converting one-hot format of digits to normal values/labels
y_val = np.argmax(Y_val, 1) # reverse of to_categorical
print (y_val)
# Note that: capital Y_val contains values in one-hot format and small y_val contains normal digit values


# ## Reshaping images
# 
# The image dimension expected by Keras for 2D (two-dimensional) convolution is in the format of **[pixels][width][height]**.
# 
# For RGB color image, the first dimension (pixel) value would be 3 for the red, green and blue components. It's like having 3 image inputs for every single color image. In our case (for MNIST handwritten images), we have gray scale images. Hence, the pixel dimension is set as 1.

# In[ ]:


# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).astype('float32')

print (num_pixels, X_train.shape, X_test.shape, X_val.shape)


# In[ ]:


print (X_train[1])


# ## Define Convolutional Neural Network (CNN) Model
# 
# **Convolution Layer**
# - We define 32 feature maps with the size of 5x5 matrix
# - We use ReLU (Rectified Linear Units) as the activation function
# - This layer expects input image size of 1x28x28 ([pixels][height][weight])
# 
# **Max Pooling Layer**
# - It has a pool size of 2x2
# 
# **Dropout Layer**
# - Configured to randomly exclude 20% of neurons in the layer to reduce overfitting
# 
# **Flatten**
# - Flattens the image into a single dimensional vector which is required as input by the fully connected layer
# 
# **Fully connected Layer**
# - Contains 128 neurons
# - relu is used as an activation function
# - Output layer has num_classes=10 neurons for the 10 classes
# - softmax activation function is used in the output layer
# - adam is used as optimizer to learn and update weights

# In[ ]:


# baseline model for CNN
def baseline_model():
    # create model    
    model = Sequential()    
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Dropout(0.2))
    model.add(Flatten())    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model


# To compile the model, there are [different optimizers](https://keras.io/optimizers/) present in Keras like Stochastic Gradient Descent optimizer, Adam optimizer, RMSprop optimizer, etc.

# In[ ]:


# Example of using RMSprop optimizer
#from keras.optimizers import RMSprop, SGD
#model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])


# ## Fit and Evaluate Model
# 
# The model is fit over 5 epochs/iteration. It takes a batch of 200 images in each iteration. Validation data is used as validation set. The epochs may be increased to improve accuracy.
# 
# Finally, validation data is used to evaluate the model by calculating the model's classification accuracy.

# In[ ]:


model = baseline_model()
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=200, verbose=1)


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


plt.figure(figsize=[10,4])

plt.subplot(121)
plt.plot(range(1, len(history_dict['val_acc'])+1), history_dict['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(122)
plt.plot(range(1, len(history_dict['val_loss'])+1), history_dict['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')


# In[ ]:


model.summary()


# In[ ]:


scores = model.evaluate(X_val, Y_val, verbose=0)
print (scores)
print ('Score: {}'.format(scores[0]))
print ('Accuracy: {}'.format(scores[1]))


# Accuracy (98.64%) of Convolution Neural Network (CNN) model has improved as compared to the accuracy (97.83%) of Multi-layer Perceptron (MLP) model. 
# 
# The accuracy of CNN model can be further increased by:
# - increasing the epoch number while fitting the model
# - adding more convolution and pooling layers to the model

# ## Plot correctly and incorrectly predicted images
# 
# Let's plot some images which are correctly predicted and some images which are incorrectly predicted on our test dataset.

# In[ ]:


# get predicted values
predicted_classes = model.predict_classes(X_val)


# In[ ]:


# get index list of all correctly predicted values
correct_indices = np.nonzero(np.equal(predicted_classes, y_val))[0]

# get index list of all incorrectly predicted values
incorrect_indices = np.nonzero(np.not_equal(predicted_classes, y_val))[0]


# In[ ]:


print ('Correctly predicted: %i' % np.size(correct_indices))
print ('Incorrectly predicted: %i' % np.size(incorrect_indices))


# In[ ]:


plt.figure(figsize=[20,8])
for i, correct in enumerate(correct_indices[:6]):
    plt.subplot(1,6,i+1)
    plt.imshow(X_val[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_val[correct]))
    
plt.figure(figsize=[20,8])
for i, incorrect in enumerate(incorrect_indices[:6]):
    plt.subplot(1,6,i+1)
    plt.imshow(X_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_val[incorrect]))


# ## Confusion Matrix

# In[ ]:


# we have digit labels from 0 to 9
# we can either manually create a class variable with those labels
# class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# or, we can take unique values from train dataset's labels
class_names = np.unique(y_train)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, predicted_classes)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = class_names
predicted_class_names = class_names

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (8,6))

#plt.subplot(121)
ax = sns.heatmap(df_cnf_matrix, annot=True, fmt='d')
ax.set_ylabel('True values')
ax.set_xlabel('Predicted values')
ax.set_title('Confusion Matrix in Numbers')

'''
plt.subplot(122)
ax = sns.heatmap(df_cnf_matrix_percent, annot=True)
ax.set_ylabel('True values')
ax.set_xlabel('Predicted values')
'''


# ### MLP Model outcome
# 
# Out of 4200 validation data, we had got the following correct and incorrect predictions by MLP model:
# - Correctly predicted: 4109
# - Incorrectly predicted: 91
# 
# Using Multi-layer Perceptron (MLP) Model, we had the following heatmap outcome:
# - 9 values of digit 9 were predicted as 7.
# - 6 values of digit 3 were perdicted as 8.
# - 4 values of digit 4, 5 and 9 were predicted as 9, 3 and 4 respectively.
# 
# ### CNN Model outcome
# 
# Out of 4200 validation data, we had got the following correct and incorrect predictions by CNN model:
# - Correctly predicted: 4146
# - Incorrectly predicted: 57
# 
# Using Convolutional Neural Network (CNN) Model, we had the following improvements:
# - Number 9 predicted as 7 has been reduced from 9 to 7 times.
# - Number 3 predicted as 8 has been reduced from 6 to 1 times.
# - Number 4,9 predicted as 9,4 repectively has been reduced from 4 times to 2 times.
# 
# The accuracy of CNN model can be further increased by:
# - increasing the epoch/iteration number while fitting the model
# - adding more convolution and pooling layers to the model

# ## Improving accuracy using multiple CNN layer
# 
# Let's try adding multiple convolution layers (*Conv2D*) and multiple fully-connected layers (*Dense*) as well.
# 
# >The second Convolution layer will have 15 filters with the size of 3x3 matrix.
# >
# >The second fully-connected layer will have 50 neurons.
# 
# We also use 10 epochs this time instead of 5.

# In[ ]:


def baseline_model():
    # create model
    model = Sequential()
    
    model.add(Conv2D(filters=30, kernel_size=(5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[ ]:


# build the model
model = baseline_model()

# fit the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=200)


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


plt.figure(figsize=[10,4])

plt.subplot(121)
plt.plot(range(1, len(history_dict['val_acc'])+1), history_dict['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(122)
plt.plot(range(1, len(history_dict['val_loss'])+1), history_dict['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')


# In[ ]:


model.summary()


# In[ ]:


scores = model.evaluate(X_val, Y_val, verbose=0)
print (scores)
print ('Score: {}'.format(scores[0]))
print ('Accuracy: {}'.format(scores[1]))


# Accuracy has improved from **98.64%** to **99.00%**.

# ## Submission to Kaggle

# In[ ]:


# get predicted values for test dataset
predicted_classes = model.predict_classes(X_test)

submissions = pd.DataFrame({'ImageId': list(range(1, len(predicted_classes) + 1)), 
                            "Label": predicted_classes})

#submissions.to_csv("submission.csv", index=False, header=True)


# ## References
# 
# - [Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
# - [Building a simple neural-network with Keras](https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb)
# - [Architecture of Convolutional Neural Networks (CNNs) demystified](https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/)
# - [Welcome to deep learning (CNN 99%)](https://www.kaggle.com/toregil/welcome-to-deep-learning-cnn-99)
# - [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
# 
