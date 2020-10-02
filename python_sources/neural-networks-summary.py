#!/usr/bin/env python
# coding: utf-8

# # **Neural Networks Summary**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical


# ## Regression

# In[ ]:


model = Sequential()

n_cols = data.shape[1]
model.add(Dense(5, activation='relu', input_shape=n_cols))  # input shape has to be the same as number of columns
model.add(Dense(5, activation='relu'))
model.add(Dense(1))  # output layer

model.compile(optimizer='adam', loss='mean_square_error') 
# adam is more effcicient than gradient descent.
# it adapts the learning rate automatically

model.fit(predictors, target)
predictions = model.predict(test_data)


# ## Classification

# In[ ]:


model = Sequential()

n_cols = data.shape[1]
target = to_categorical(target)

model.add(Dense(5, activation='relu', input_shape=n_cols)) 
model.add(Dropout(0.2))  # dropout is a regularization technique to prevent overfitting. Normally ~0.2-0.4
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  
# for classification the last layer has an activation function which ussually is softmax
# in addition, the output dimension has to be the same as the calsses in the target

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # to measure accuracy in classification

model.fit(predictors, target, 
          epochs=20,  # number of iterations
          batch_size=50,  # size of the splitted bachs (this only works with SGD?)
          validation_split=0.2, )
predictions = model.predict(test_data)


# ## Convolutional Neural Networks (CNN) - supervised
# 
# This are mainly use for images as they reduce dimensionality. Check this [link](https://courses.edx.org/courses/course-v1:IBM+DL0101EN+3T2019/courseware/89227024130b43f684d95376901b65c8/052a444d45914712a597f0c58cbc4391/?child=first)

# In[ ]:


model = Sequential()

input_shape = (N, N, 3)  # 3 for RGB images and 1 for gray scale images

model.add(Conv2D(16, kernel_size=(2, 2),  # size of the filter to use
                 strides=(1, 1),  # steps the filter is moved
                 activation='relu', 
                 input_shape=input_shape)) 
model.add(MaxPool2D(pool_size(2, 2),  strides=(1, 1))
model.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), activation='relu') 
model.add(MaxPool2D(pool_size(2, 2))  
model.add(Flatten())  # so the data can proceed to the fully-connected layer
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='sotmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # to measure accuracy in classification

model.fit(predictors, target)
predictions = model.predict(test_data)


# ## Recurrent Neural Networks (RNN)  - supervised
# 
# This are networks with loops that take into account dependency of data like images in a movie.

# ## Autoencoders  - unsupervised
# 
# These commpress and decompress functions learned from data. For this reason they are data-specific.
# 
# These are used in data de-noising and dimensionality reduction for data visualisation.

# In[ ]:




