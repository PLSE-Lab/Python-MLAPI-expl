#!/usr/bin/env python
# coding: utf-8

# # IAP202-Land_Cover_Classification

# We would like to automatically classify the land cover of different landscapes.  
# ![mozaic](https://docs.google.com/uc?export=download&id=1HypyX6kYGjEdt8M7E_J2JDy29BppoGkr)

# You will need GPU for this course, make sure the GPU option on the right is 'On'.

# ## 1) Loading data

# Our database consists of 27 000 RGB images of size 64*64.  
# We will use the [numpy](https://www.numpy.org/) library to process these data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob


# In[ ]:


name_train = sorted(glob("/kaggle/input/enseeiht/cerfacs/TRAIN/*"))
name_test = sorted(glob("/kaggle/input/enseeiht/cerfacs/TEST/*"))

y_train = np.load("/kaggle/input/enseeiht/cerfacs/y_train.npy")

print (len(name_train), len(name_test))


# Our dataset is divided in a training and a test set containing respectively 20 000 and 7000 images.  
# Let's visualise some of them with [matplotlib](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html):

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

num = np.random.randint(len(name_train))
plt.figure(figsize=(6, 6))
plt.title("Image {} : {}".format(num, y_train[num]))
plt.imshow(Image.open(name_train[num]));


# In[ ]:


figure = plt.figure(figsize=(12, 12))
size = 5
grid = plt.GridSpec(size, size, hspace=0.05, wspace=0.0)

for line in range(size):
    for col in range(size):
        figure.add_subplot(grid[line, col])
        num = np.random.randint(len(name_train))
        plt.imshow(Image.open(name_train[num]))
        plt.axis('off')  


# We can also convert this list into a numpy array, which is more suitable for common mainpulations :

# In[ ]:


X_train = np.array([np.array(Image.open(jpg)) for jpg in name_train])
X_test = np.array([np.array(Image.open(jpg)) for jpg in name_test])
y_train = np.load("/kaggle/input/enseeiht/cerfacs/y_train.npy")

print (X_train.shape, X_test.shape)
print (y_train.shape)


# If we look at the X_train shape for example, we can see that it contains 20 000 images of size (64*64) with 3 channels (RGB).  
# y_train is a list containing the land cover for each of the X_train samples.

# ## 2) Pre processing

# In[ ]:


print (y_train[0])


# ML algorithm cannot work with label directly so we need to use a one hot encoding for our data :  
# Every label will be represented as a binary vector with 0 everywhere and 1 for the index of its class.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

print (f"Shape label raw : {y_train.shape}")

encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

print (f"Shape label One Hot Encoded : {y_train.shape}")
print (f"Label for y_train[0] : {y_train[0]}")


# We then create a validation set to evaluate our models during the training phase :

# In[ ]:


X_train, X_valid = X_train[:15000], X_train[15000:]
y_train, y_valid = y_train[:15000], y_train[15000:]


# And we scale our inputs data so that each pixel value has a value between 0 and 1 :

# In[ ]:


X_train, X_valid, X_test = X_train/255, X_valid/255, X_test/255


# ## 3) Training a model on this dataset

# We first define the architecture of our network, a basic CNN with few convolutional layers and a dense layers with 10 neurons : one for every possible classes.

# In[ ]:


import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation = 'softmax'))


# We then compile this model with an [SGD](https://keras.io/optimizers/) optimizer and a [categorical crossentropy](https://keras.io/metrics/) as loss :

# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


# And we start the training phase :

# In[ ]:


history = model.fit(X_train, y_train, batch_size = 32, 
                   validation_data=(X_valid, y_valid), epochs=30)


# Once our model is trained, we can check its accuracy on the validation set :

# In[ ]:


loss, metrics = model.evaluate(X_valid, y_valid)

print (metrics)


# We obtained an accuracy of about 75%, that's a good start !

# ## 4) The overfitting

# However despite the decent accuracy obtained on the validation set, the accuracy on the training set is almost perfect with about 99.9%.
# This huge gap between the accuracy on the training an validation set is a clear sign of [overfitting](here) : when a model starts memorizing specific pattern of the training set and do not generalize well to new data.
# 
# Therefore, our main concern now will be to reduce this overfitting.
