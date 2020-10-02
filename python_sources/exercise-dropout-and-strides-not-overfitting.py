#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You've built a model to identify clothing types in the **MNIST for Fashion** dataset.  Now you will make your model bigger, specify larger stride lengths and apply dropout. These changes will make your model faster and more accurate.
# 
# This is the last step in the **[Deep Learning Track](https://www.kaggle.com/learn/deep-learning)**.
# 
# # Starter Code
# 
# ## Data Preparation
# **You need to run this cell of code.**
# 

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

num_epochs=6

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)


# ## Sample Model Code
# 
# ```
# fashion_model = Sequential()
# fashion_model.add(Conv2D(12, kernel_size=(3, 3), strides=2,
#                  activation='relu',
#                  input_shape=(img_rows, img_cols, 1)))
# fashion_model.add(Conv2D(12, (3, 3), strides=2, activation='relu'))
# fashion_model.add(Flatten())
# fashion_model.add(Dense(128, activation='relu'))
# fashion_model.add(Dense(num_classes, activation='softmax'))
# 
# fashion_model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer='adam',
#               metrics=['accuracy'])
# 
# fashion_model.fit(train_x, train_y,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split = 0.2)
# ```

# # Adding Strides
# Specify, compile and fit a model much like the model above, but specify a stride length of 2 for each convolutional layer.  Call your new model `fashion_model_1`

# In[ ]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

fashion_model_1 = Sequential()
# Specify the rest of the model
fashion_model_1.add(Conv2D(12, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_1.add(Conv2D(12, (3, 3), strides=2, activation='relu'))
fashion_model_1.add(Flatten())
fashion_model_1.add(Dense(128, activation='relu'))
fashion_model_1.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1
fashion_model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1
fashion_model_1.fit(x, y,
          batch_size=100,
          epochs=num_epochs,
          validation_split = 0.2)


# # Make Model Larger
# You should have noticed that `fashion_model_1` trained pretty quickly.  This makes it reasonable to make the model larger. Specify a new model called `fashion_model_2` that is identical to fashion_model_1, except:
# 1. Add an additional `Conv2D` layer immediately before the Flatten layer. Make it similar to the Conv2D layers you already have, except don't set the stride length in this new layer (we have already shrunk the representation enough with the existing layers)..
# 2. Change the number of filters in each convolutional layer to 24.
# 
# After specifying `fashion_model_2`, compile and fit it

# In[ ]:


# Your code for fashion_model_2 below


fashion_model_2 = Sequential()
# Specify the rest of the model
fashion_model_2.add(Conv2D(24, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_2.add(Conv2D(24, (3, 3), strides=2, activation='relu'))
fashion_model_2.add(Conv2D(24, (3, 3), activation='relu'))
fashion_model_2.add(Flatten())
fashion_model_2.add(Dense(128, activation='relu'))
fashion_model_2.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1
fashion_model_2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1
fashion_model_2.fit(x, y,
          batch_size=100,
          epochs=num_epochs,
          validation_split = 0.2)


# # Add Dropout
# Specify `fashion_model_3`, which is identical to `fashion_model_2` except that it adds dropout immediately after each convolutional layer (so it adds dropout 3 times). Compile and fit this model.  Compare the model's performance on validation data to the previous models.
# 

# In[ ]:


# Your code for fashion_model_3 below

fashion_model_3 = Sequential()
# Specify the rest of the model
fashion_model_3.add(Conv2D(24, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_3.add(Dropout(0.5))
fashion_model_3.add(Conv2D(24, (3, 3), strides=2, activation='relu'))
fashion_model_3.add(Dropout(0.35))
fashion_model_3.add(Conv2D(24, (3, 3), activation='relu'))
fashion_model_3.add(Dropout(0.2))
fashion_model_3.add(Flatten())
fashion_model_3.add(Dense(128, activation='relu'))
fashion_model_3.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1
fashion_model_3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1
fashion_model_3.fit(x, y,
          batch_size=100,
          epochs=num_epochs,
          validation_split = 0.2)


# # Congrats
# You've finished level 1 of the deep learning track.  You have the tools to create and tune computer vision models.  Pick a project and try out your skills.  
# 
# A few fun datasets you might try include:
# - [Written letter recognition](https://www.kaggle.com/olgabelitskaya/classification-of-handwritten-letters)
# - [Flower Identification](https://www.kaggle.com/alxmamaev/flowers-recognition)
# - [Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
# - [10 Monkeys](https://www.kaggle.com/slothkong/10-monkey-species)
# - [Predict Bone Age from X-Rays](https://www.kaggle.com/kmader/rsna-bone-age)
# 
# You have learned a lot. There is still a lot more to learn in deep learning, but you should feel great about your new skills.
# 
