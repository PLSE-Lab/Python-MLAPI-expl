#!/usr/bin/env python
# coding: utf-8

# **Digit Recognizer**
# 
# Try to correctly identify digits from 0 to 9 from MNIST dataset.

# In[ ]:


import pandas as pd                                  # for data manipulation
import matplotlib.pyplot as plt                      # for data representation
import numpy as np                                   # linear algebra
from keras.utils.np_utils import to_categorical      # one hot enconding
from sklearn.model_selection import train_test_split # selecting train and test samples
from keras.models import Sequential                  # to make our model customizable
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.python import keras                  # I'll import loss from here

# Load data:

train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv')


# Let's explore the data in order to prepare our predictors x and y. 

# In[ ]:


print(train_data.head())


# First column show the index of the row. 'label' contains the number which corresponds to each image (*y* value). Each pixel column contains pixels intensity from 0 to 255 in grey scale.

# In[ ]:


print(test_data.head())


# Test file doesn't contain label column.

# In[ ]:


print('train.csv dataset contains %d different images.' %(len(train_data)))
print('test.csv dataset contains %d different images.' %(len(test_data)))


# In[ ]:


# defining x and y:

x = train_data.drop(['label'], axis = 'columns')  # we quit y column from the dataset
y = train_data['label']
'''
plt.figure(1)
plt.title('Y_train')
plt.hist(Y_train, rwidth = 0.9)
plt.xlabel('numbers')
plt.ylabel('count')
plt.show()
'''
# Models usually works better when values are normalized:

x         = x / 255.0
test_data = test_data / 255.0

# We also need Y_train to be categorical in order to train our model

y_categorical = to_categorical(y, num_classes = 10)


# We are going to work with Conv2D neural network so it's necessary to reshape X_train from pixels to 2d images.

# In[ ]:


x = x.values.reshape(-1, 28, 28, 1)  # (28, 28) is image length, 1 is to build a 3D matrix which contains
                                       #  all images and -1 is because of Keras channel dimension.
    
test_data = test_data.values.reshape(-1, 28, 28, 1)


# In[ ]:


for i in range(0, 9):
    plt.subplot(3,3,i+1)
    plt.imshow(x[i][:,:,0])
    plt.title(y[i])


# Once we have seen our numbers it's time to split data into train and test for our model.

# In[ ]:


# Define the model:

'''
model = Sequential()
model.add(Conv2D(12, input_shape = (28, 28, 1), kernel_size = 3, activation = 'relu'))
model.add(Conv2D(2, kernel_size = 3, activation = "relu"))
model.add(Conv2D(2, kernel_size = 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

'''
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


# Compile the model

model.compile(loss = keras.losses.categorical_crossentropy,
                     optimizer = "adam",
                     metrics = ['accuracy'])


# In[ ]:


# Fit the model

history = model.fit(x, y_categorical, batch_size = 100, epochs = 20, validation_split = 0.2)


# In[ ]:


print(history.history.keys())


# In[ ]:


# copied from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


answer = model.predict(test_data)
print(answer) # array of probabilities. We'll take the most probable.


# In[ ]:


answer = np.argmax(answer, axis = 1)
answer = pd.Series(answer, name = "Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), answer], axis = 1)
submission.to_csv("my_submission.csv", index = False)

