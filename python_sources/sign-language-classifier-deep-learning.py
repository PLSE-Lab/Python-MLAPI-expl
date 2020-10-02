#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook contains code used to build a deep learning model on Keras to predict Sign Language from the MNIST dataset

# In[ ]:


# Importing Modules
import numpy as np
import pandas as pd

# Keras modules
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

# Train test split
from sklearn.model_selection import train_test_split

# Display and plotting
from IPython.display import Image
import plotly_express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))


# # Sign Language
# An image describing the classification task

# In[ ]:


Image(filename = "../input/american_sign_language.PNG")


# # Loading Data

# In[ ]:


# Loading train and test sets
train = pd.read_csv("../input/sign_mnist_train.csv")
test = pd.read_csv("../input/sign_mnist_test.csv")


# In[ ]:


print("Train:")
print(train.head())
print("\nTest:")
print(test.head())


# In[ ]:


# Looking at training data info
print(train.info())


# # Preprocessing & Visualizing Data

# In[ ]:


# Checking distribution of data
label_dist = pd.DataFrame(train['label'].value_counts()).reset_index()
label_dist.columns = ['Label','Count']
px.bar(label_dist,x = "Label", color = "Label", y = "Count")


# Good, we shouldn't have data for J (label 9) and Z (label 25). They involve movement.

# # Setting up Train and Test Sets

# In[ ]:


# Defining X and Ys
X_train = train.iloc[:,1:].copy()
Y_train = train.iloc[:,0].copy()

X_test = test.iloc[:,1:].copy()
Y_test = test.iloc[:,0].copy()


# In[ ]:


# Splitting training model into train and validation sets for deep learning model
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.3)


# In[ ]:


# Rescaling data to fall between 0 and 1
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255


# In[ ]:


# Converting to Numpy array and Reshaping X_train and test data
X_train = np.array(X_train).reshape(X_train.shape[0],28,28,1)
X_val = np.array(X_val).reshape(X_val.shape[0],28,28,1)
X_test = np.array(X_test).reshape(X_test.shape[0],28,28,1)


# In[ ]:


# Categorizing Ys
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)


# # Building Keras Model

# In[ ]:


# Building Keras model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(units = 25, activation = 'softmax'))


# In[ ]:


# Compiling model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# # Training the Model
# Here, we train the model and evaluate the ideal number of epochs before overfitting

# In[ ]:


history = model.fit(X_train,Y_train, validation_data = (X_val, Y_val),epochs = 10, batch_size = 64)


# # Visualizing Validation Loss/Accuracy Over Epochs

# In[ ]:


data = pd.DataFrame(history.history).reset_index()
data.columns = ['Epoch', "Validation_Loss","Validation_Accuracy","Loss","Accuracy"]
trace1 = go.Scatter(
    x = (data['Epoch'] + 1).values,
    y = data['Loss'].values,
    name = "Loss",
    mode = "lines+markers"
)
trace2 = go.Scatter(
    x = (data['Epoch'] + 1).values,
    y = data['Validation_Loss'].values,
    name = "Validation_Loss",
    mode = "lines+markers"
)
trace3 = go.Scatter(
    x = (data['Epoch'] + 1).values,
    y = data['Validation_Accuracy'].values,
    name = "Validation_Accuracy",
    mode = "lines+markers"
)
trace4 = go.Scatter(
    x = (data['Epoch'] + 1).values,
    y = data['Accuracy'].values,
    name = "Accuracy",
    mode = "lines+markers"
)
fig1 =[trace1,trace2]
fig2 = [trace3,trace4]
iplot(fig1)
iplot(fig2)


# It seems like the model only really needs about 4 epochs

# # Final Model

# In[ ]:


# Final Model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(units = 25, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model.fit(X_train,Y_train, validation_data = (X_val, Y_val),epochs = 4, batch_size = 64)


# # Evaluating Model

# In[ ]:


# Model evaluation
model.evaluate(X_test,Y_test)


# Not a bad result!
