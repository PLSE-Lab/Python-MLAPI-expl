#!/usr/bin/env python
# coding: utf-8

# The model will be builded using Keras.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


# First we have to load data from csv files.

# In[ ]:


train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')


# ## Examining data
# Training data consist 42000 rows. In column called label say what our model should return as a prediction, when we give as an input rest of that row data. Each of the next column represents one pixel of the digit image.
# Image is 28px width and 28px height.
# 
# Examplory digit is shown below.

# In[ ]:


sample_image = train_data.drop(columns='label').values[6].reshape(28,28)
plt.imshow(sample_image, cmap=plt.cm.gray.reversed())


# As you can see on the bar graph, counts of samples corresponding to each digit are similar.

# In[ ]:


from collections import Counter
x_test = test_data
y_train = train_data['label'].values
counts = Counter(y_train)
digits_count = [counts.get(i) for i in range(10) ]
plt.title('Digits count in training datasets')
plt.xlabel('digit')
plt.ylabel('count')
plt.bar(range(10), digits_count)


# ## Preprocessing data

# Splitting the data into input data and output data

# In[ ]:


y_train = train_data['label'].values
x_train = train_data.drop(columns='label').values
x_test = x_test.values


# Normalizing the date.
# We want to make that each pixel is in range (0, 1). It can help neural network learn more efficient.

# In[ ]:


x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255


# Output shold be one-hot encoded
# 
# e.g. for digits equal one it should be [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# In[ ]:


y_train = to_categorical(y_train, 10)


# Reshaping our input data. We want that each image have the shape equal (28, 28, 1) (width, height, 1 - because gray scale)

# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# ## Building model

# In[ ]:


model = Sequential([
    Conv2D(32, input_shape=(28,28,1), kernel_size=(4,4), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(4,4), activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ## Results

# In[ ]:


history = model.fit(x_train, y_train, epochs=16, batch_size=32)


# Model on training data reached accuracy equal 0.9964285492897034 and loss 0.01162335854808615

# In[ ]:


plt.figure(figsize=(20,5))
plt.title('loss during training')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(history.history['loss'])

