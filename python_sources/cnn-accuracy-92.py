#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, Convolutional Neural Network (CNN), with 2 convolution layers and 2 max pooling layers, is used. The data in file 'fashion-mnist_train.csv' is split into training and validation sets in 80:20 ratio. 
# 
# #### The notebook is divided into 3 parts:
# * Loading and preprocessing the data
# * Building and training a neural network
# * Evaluating the model performace and comparing our predictions with the actual class labels

# ### Step 1 - Loading and preprocessing the data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# loading data
seed = 6
np.random.seed(seed)

training_set = pd.read_csv('../input/fashion-mnist_train.csv')
test_set = pd.read_csv('../input/fashion-mnist_test.csv')


# In[3]:


# checking the characteristics of data
training_set.head()


# In[4]:


# storing the class labels and image values into separate variables.
X = training_set.iloc[:, 1:].values
y = training_set.iloc[:, 0].values


# In[5]:


print ('Input vectors : {}'.format(X.shape))
print ('Class Labels : {}'.format(y.shape))


# In[6]:


# reshaping

X = X.reshape(-1,28,28,1)
X.shape


# In[7]:


# checking the first 9 images

pixels_x = 28
pixels_y = 28

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X[i].reshape(pixels_x, pixels_y)
    plt.imshow(img, cmap = 'gray')

plt.show()


# In[8]:


# splitting the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[9]:


# normalizing the pixel values
X_train = X_train.astype('float32')/255.0
X_validation = X_validation.astype('float32')/255.0


# In[10]:


# checking the class labels
print (y_train[0])
print (y_validation[0])


# In[11]:


# converting the numeric class labels to binary form with one hot encoding

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)

y_train[0]


# ### Step 2 - Building and Training CNN
# 
# ### Following parameters are chosen for our CNN network.
# * Input vector = 28 x 28 image
# * 3 x 3 convolution layer with 32 filters
# * Max pooling layer with 2 x 2 pool size
# * 3 x 3 convolution layer with 64 filters
# * Max pooling layer with 2 x 2 pool size
# * Fully connected layer with 128 nodes
# * Activation functions - Relu for convolutional, max pooling and fully connected layers, softmax for output layer
# * Optimizer - Adam 

# In[12]:


from keras.models import Sequential 
from keras.layers import Conv2D  
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout


# In[13]:


model = Sequential()

# first convolutional and max pooling layers
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# second convolutional and max pooling layers
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())  

model.add(Dense(128, activation='relu'))     # fully connected layer
model.add(Dense(10, activation='softmax'))  # output layer

# Compiling CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[14]:


model.summary()


# In[15]:


# fitting the model to training data

history = model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size = 100, epochs = 10)


# ### Step 3 - Evaluation

# In[16]:


# evaluating the performance of model on validation set
scores = model.evaluate(X_validation, y_validation, verbose = 1)
print ('Accuracy : {}'.format(scores[1]))


# ### Predicting the output on test data. Since, we already have the actual values of class labels from file fashion-mnist_test.csv, we will also check the accuracy between our predictions and actual labels.

# In[17]:


X_test = test_set.iloc[:, 1:].values
y_test = test_set.iloc[:, 0].values

X_test = X_test.reshape(-1, 28, 28, 1)
X_test = X_test.astype('float32')/255.0

X_test.shape


# In[18]:


# predicting and storing the values in y_pred
y_pred = model.predict(X_test)
# selecting the class with highest probability
y_pred = np.argmax(y_pred, axis = 1)

from sklearn.metrics import accuracy_score
print ('Accuracy on Test Set = {:0.2f}'.format(accuracy_score(y_test, y_pred)))


# In[ ]:




