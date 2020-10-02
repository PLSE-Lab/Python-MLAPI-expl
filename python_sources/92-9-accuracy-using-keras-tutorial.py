#!/usr/bin/env python
# coding: utf-8

# # Making a Convolutional Neural Network using Keras to classify the images

# ## Data preparation and visualization

# ### Library Imports

# In[ ]:


import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import seaborn as sns
from sklearn import metrics


# ### Importing Datasets

# In[ ]:


train = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")


# #### Printing size (Rows and Columns) of the datasets

# In[ ]:


print(train.shape)
print(test.shape)


# ### Head of the train data

# In[ ]:


train.head()


# Each image is 28x28, so there are 784 pixels per image.
# 
# Each pixel holds a value between 0 and 255 (grayscale).
# 
# 1 column is for label value, seen below

# ### Labels and their descriptions:

# 
# | Label | Description |
# | --- | --- |
# | 0 | T-shirt/top |
# | 1 | Trouser |
# | 2 | Pullover |
# | 3 | Dress |
# | 4 | Coat |
# | 5 | Sandal |
# | 6 | Shirt |
# | 7 | Sneaker |
# | 8 | Bag |
# | 9 | Ankle boot |

# ### Function to convert label values to their desciptions

# In[ ]:


values = {0: "T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}
def lab_to_desc(label):
    return values[label]


# ### Converting the dataframe to a numpy array, so as to be able to visualize it

# In[ ]:


train_arr = np.array(train)
test_arr = np.array(test)
train_arr


# ### Plotting one of the images, and printing its label.

# In[ ]:


fig = plt.figure()

# From training dataset
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(train_arr[1, 1:].reshape(28, 28), cmap="Greys")
print("Image 1 label: ",train_arr[1, 0])

# From testing dataset
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(test_arr[1, 1:].reshape(28, 28), cmap="Greys")
print("Image 2 label: ",test_arr[1, 0])


# We can see that it is an ankle boot (label 9), and a Trouser (label 1)

# ### Normalizing the Data for our Neural Network

# We do this because Neural Networks work well with comparable ranged inputs.
# 
# Dividing by 255 to get values between 0 and 1

# In[ ]:


train_X = train_arr[:, 1:]/255
test_X = test_arr[:, 1:]/255
print(train_X.shape)
print(test_X.shape)


# Making y (output) datasets from the 1st column of the data

# In[ ]:


train_y = train_arr[:, 0]
test_y = test_arr[:, 0]


# ### Reshaping the datasets as 28x28 inputs

# In[ ]:


train_X = train_X.reshape([train_X.shape[0], 28, 28, 1])
print(train_X.shape)
test_X = test_X.reshape([test_X.shape[0], 28, 28, 1])
print(test_X.shape)


# ## Now that our data is ready, we can begin with making a Keras model

# ### Library Imports

# In[ ]:


from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


# ### A basic, small Sequential model first.

# In[ ]:


classifier1 = Sequential()
classifier1.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'))
classifier1.add(MaxPooling2D(pool_size=(2, 2)))
classifier1.add(Flatten())
classifier1.add(Dense(units = 16, activation = 'relu'))
# 10 units in the last Dense layer as there are 10 classes to be classified into
classifier1.add(Dense(units = 10, activation = 'sigmoid'))


# In[ ]:


classifier1.summary()


# In[ ]:


classifier1.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# #### Fitting the data onto the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = classifier1.fit(train_X, train_y, epochs=10, batch_size=256)')


# #### Plotting changes in accuracy

# In[ ]:


print(history.history.keys())

plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# #### Evaluating the model on our test dataset

# In[ ]:


result = classifier1.evaluate(x=test_X, y=test_y)
print("Accuracy of the model is: %.2f percent"%(result[1]*100))


# **Good results, for a relatively small model. Let us visualize the results with a few random images and their results**

# In[ ]:


import random


# In[ ]:


predictions = classifier1.predict_classes(test_X)


# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(14, 14))
# axes is currently in multiple lists, ravel reshapes it to 1D
axes = axes.ravel()

randnum = random.randint(0, 9975)

for i in range(25):
    axes[i].imshow(test_X[randnum + i].reshape(28, 28), cmap="Greys")
    axes[i].set_title('Prediction: %s\n True: %s' %
                      (lab_to_desc(predictions[randnum + i]), lab_to_desc(test_y[randnum + i])))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)


# **Let us try with a bigger, more complicated CNN, hopefully we get better results**

# ### Bigger CNN

# In[ ]:


classifier = Sequential()
classifier.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=10, activation='sigmoid'))


# In[ ]:


classifier.summary()


# In[ ]:


classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# #### Fitting the data on the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = classifier.fit(train_X, train_y, epochs=25, batch_size=256)')


# #### Plotting changes in accuracy

# In[ ]:


print(history.history.keys())

plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# #### Evaluating the model on our test dataset

# In[ ]:


result = classifier.evaluate(x=test_X, y=test_y)
print("Accuracy of the model is: %.2f percent"%(result[1]*100))


# **Better results with a more complicated network, as expected.**

# **Let us visualize the results with a few random images and their results**

# In[ ]:


predictions = classifier.predict_classes(test_X)


# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(14, 14))
# axes is currently in multiple lists, ravel reshapes it to 1D
axes = axes.ravel()

randnum = random.randint(0, 9975)

for i in range(25):
    axes[i].imshow(test_X[randnum + i].reshape(28, 28), cmap="Greys")
    axes[i].set_title('Prediction: %s\n True: %s' %
                      (lab_to_desc(predictions[randnum + i]), lab_to_desc(test_y[randnum + i])))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)


# ### Saving the model locally, so it can be loaded later

# In[ ]:


classifier.save_weights("CNN2.h5")
print("Saved model to disk.")


# # **Hence we have successfully made high accuracy image classifications using Convolutional Neural Networks through Keras**
