#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognition using Neural Network

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


print( "Test =",df_test.shape , "\nTrain =",df_train.shape)


# In[ ]:


df_test.head()


# In[ ]:


df_train.head()


# ## Splitting train and test data

# In[ ]:


x_train = df_train.iloc[:, 1:785]
y_train = df_train.iloc[:, 0]
x_test = df_test.iloc[:, 0:784]


# ## Data Filtering

# In[ ]:


scaller = StandardScaler()
x_test = scaller.fit_transform(x_test)
x_train = scaller.fit_transform(x_train)


# ## Deep Learning/Implementation of Neural Network

# In[ ]:


model = Sequential()
# model.add(Flatten())
model.add(Dense(784, activation = "relu", input_shape = (784,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer= "adam", metrics = ["accuracy"])
history = model.fit(x_train, y_train, validation_split = 0.30, epochs=30, batch_size=len(x_train))


# In[ ]:


acc = history.history["accuracy"]
val_acc= history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
print("Accuracy = ", acc[-1], "\nValidation Accuracy = ", val_acc[-1])


# We are getting final **accurcy score** of 96.39% and **validation accuracy score** of 94.78%
# * Neural net specifications:
# * epochs = 30
# * Number of Hidden Layers: 3
# * Hidden Layer 1 nodes: 100
# * Hidden Layer 2 nodes: 300
# * Hidden Layer 3 nodes: 300
# * Output layer nodes: 10  --->   because the dataset has 10 digitd i.e. 0 to 9

# ## Plotting box plot

# In[ ]:


tva = pd.DataFrame(
        {
            "Ta":acc,
            "Va":val_acc
        }
)
tva.boxplot()


# ## Prediction

# In[ ]:


prediction = model.predict_classes(x_test)
print("first digit is ",prediction[0])


# ## Displaying first image in the dataset

# In[ ]:


import matplotlib.pyplot as plt
image = x_test[0]
image = np.array(image, dtype='float')
pixels = image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()


# Hence we can verify that our prediction is correct 

# In[ ]:




