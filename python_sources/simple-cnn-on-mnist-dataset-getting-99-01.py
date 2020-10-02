#!/usr/bin/env python
# coding: utf-8

# This is an example of using CNN on MNIST Dataset in the simplest way possible by me.
# 
# Hope everyone watching this likes it.

# Import the important libraries

# In[ ]:


# import the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


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

Here the shape of our data is not according to the CNN architecture, so we will reshape the data into CNN architecture that is (images,rows,cols,channels)

Here the images will be the no of the images used , rows and columns will be the pixels of the images mentioned in the dataset descriptions which are 28 * 28 . since all images are gray scale so it will only use '1' channel
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


# In[ ]:


# import model
model=Sequential()
# layers
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# simple ANN now

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(10, activation='softmax'))


# Now we will check the summary to see how many parameters are we passing in this model

# In[ ]:


model.summary()


# Now we will compile the model,by using the optimizer as adam and loss as 'binary_categorical_crossentropy' and metris as ['accuracy']

# In[ ]:


# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fitting the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history=model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test,y_test))')


# Evaluate the model

# In[ ]:


# evaluating the model with testing data
loss, accuracy=model.evaluate(X_test,y_test)
loss, accuracy


# We will plot the data

# In[ ]:


# plot the figure now
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()


# Making the prediction using the model

# In[ ]:


# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred = model.predict_classes(test)


# In[ ]:


class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names


# In[ ]:


mat=confusion_matrix(y_test, y_pred[:4200])
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))


# In[ ]:


results = pd.Series(y_pred,name="Label")


# In[ ]:


results


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




