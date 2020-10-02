#!/usr/bin/env python
# coding: utf-8

# This is an example of using CNN on MNIST Dataset Using LeNet-5 Architecture .
# 
# Hope everyone watching this likes it.

# Import the important libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# Import the data in the train and test from the input directory

# In[ ]:


# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# check the shape of the train and test data

# In[ ]:


train.shape, test.shape


# Splitting the data into X and y(i.e independent and dependent varaible in simple terms)

# In[ ]:


X = train.drop(labels=["label"], axis=1)
y = train["label"]

# check the shape
X.shape, y.shape


# Now let us check whether the target variable is imbalanced or not in the training data
# 
# 

# In[ ]:


# checking manually
y.value_counts()


# In[ ]:


# Checking by plotting the same
plt.subplots(figsize = (10,8))
plt.title('Counts in numbers to their labels ')
sns.countplot(x=y, data=train)
plt.show()


# We can easily conclude from the above that the data is not unbalanced
# 
# Now we will split the data into training and testing

# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.1 , random_state = 99)
# check the shape now
X_train.shape,X_test.shape,y_train.shape,y_test.shape,test.shape


# Here the shape of our data is not according to the CNN architecture, so we will reshape the data into CNN architecture that is (images,rows,cols,channels) Here the images will be the no of the images used , rows and columns will be the pixels of the images mentioned in the dataset descriptions which are 28 * 28 . since all images are gray scale so it will only use '1' channel

# In[ ]:


X_train=X_train.values.astype('float32')
X_test=X_test.values.astype('float32')
test=test.values.astype('float32')


# Here above we converted the values of the data into float32, by which the above three dataframes got converted into a numpy array

# In[ ]:


# changing the shape of X_train and y_train and test also
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
test=test.reshape(test.shape[0] , 28 , 28 , 1)


# Now check the shape again

# In[ ]:


X_train.shape,X_test.shape,test.shape


# Now we will check the range values of the data.

# In[ ]:


# check the maximum values in the dataset
X_train.max(),X_train.min()


# We can easily that the data range is between 0 to 255, here we need to normalize the data to bring it into the range of 0 to 1 so that our model predicts the data more efficiently

# In[ ]:


X_train=X_train/255
X_test=X_test/255
test=test/255


# In[ ]:


# check the maximum values in the dataset
X_train.max(),X_train.min()


# Now our data has been normalized, we can also scale the data by using MinMaxScaler as well as Standard Scaler

# **Now we will Build the Model**

# Before building the model here we will need to pass the input shape

# In[ ]:


input_shape=X_train[0].shape
input_shape


# ![LeNet-5%20architecture.jpg](attachment:LeNet-5%20architecture.jpg)

# Above is the LeNet Architecture , we will build our model according to this architecture
# 

# In[ ]:


# Build the Model
model= Sequential()
model.add(Conv2D(filters = 6, kernel_size = (5,5), activation='relu', input_shape = (28, 28, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))
model.add(Conv2D(filters=16, kernel_size = 5, activation='relu'))
model.add(MaxPooling2D(pool_size = 2, strides=2))
model.add(Conv2D(filters=120, kernel_size = 5, activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Compile the Model

# In[ ]:


# model.compile(optimizer= Adam(learning_rate =0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Check the summary of the model

# In[ ]:


model.summary()


# Fitting the model

# In[ ]:


history = model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.2)


# Evaluating the Model

# In[ ]:


#Evaluating the Model
model.evaluate(X_test, y_test, verbose=0)


# pd.DataFrame(history.history).plot(figsize=(10,5))
# plt.grid(True)
# plt.show()

# Making the prediction using the model

# In[ ]:


# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred = model.predict_classes(X_test)


# In[ ]:


class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names


# In[ ]:


mat=confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))


# In[ ]:




