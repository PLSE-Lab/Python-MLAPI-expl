#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You've built a model to identify clothing types in the **MNIST for Fashion** dataset.  Now you will make your model bigger, specify larger stride lengths and apply dropout. These changes will make your model faster and more accurate.
# 
# This is a last step in the **[Deep Learning Track](https://www.kaggle.com/learn/deep-learning)**.
# 
# 
# ## Data Preparation
# **Run this cell of code.**
# 

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_8 import *
print("Setup Complete")

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)


# # 1) Increasing Stride Size in A Layer
# 
# Below is a model without strides (or more accurately, with a stride length of 1)
# 
# Run it. Notice it's accuracy and how long it takes per epoch. Then you will change the stride length in one of the layers.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

batch_size = 16

fashion_model = Sequential()
fashion_model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model.add(Conv2D(16, (3, 3), activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)


# You have the same code in the cell below, but the model is now called `fashion_model_1`.  Change the specification of `fashion_model_1` so the second convolutional layer has a stride length of 2.
# 
# Run the cell after you have done that. How does the speed and accuracy change compared to the first model you ran above?

# In[ ]:


fashion_model_1 = Sequential()
fashion_model_1.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_1.add(Conv2D(16, (3, 3), activation='relu', strides=2))
fashion_model_1.add(Flatten())
fashion_model_1.add(Dense(128, activation='relu'))
fashion_model_1.add(Dense(num_classes, activation='softmax'))

fashion_model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model_1.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)
q_1.check()


# For the solution, uncomment and run the cell below:

# In[ ]:


#q_1.solution()


# You should notice that your model training ran about twice as fast, but the accuracy change was trivial.  
# 
# In addition to being faster to train, this model is also faster at making predictions. This is very important in many scenarios. In practice, you'll need to decide whether that type of speed is important in the applications where you eventually apply deep learning models.
# 
# You could experiment with more layers or more convolutions in each layer. With some fine-tuning, you can build a model that is both faster and more accurate than the original model.

# # Congrats
# You've finished the Deep Learning course.  You have the tools to create and tune computer vision models.  
# 
# If you feel like playing more with this dataset, you can open up a new code cell to experiment with different models (adding dropout, adding layers, etc.)  Or pick a new project and try out your skills.  
# 
# A few fun datasets you might try include:
# - [Written letter recognition](https://www.kaggle.com/olgabelitskaya/classification-of-handwritten-letters)
# - [Flower Identification](https://www.kaggle.com/alxmamaev/flowers-recognition)
# - [Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
# - [10 Monkeys](https://www.kaggle.com/slothkong/10-monkey-species)
# - [Predict Bone Age from X-Rays](https://www.kaggle.com/kmader/rsna-bone-age)
# 
# You have learned a lot, and you'll learn it as you practice. Have fun with it!
