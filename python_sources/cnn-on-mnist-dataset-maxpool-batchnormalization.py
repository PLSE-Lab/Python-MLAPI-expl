#!/usr/bin/env python
# coding: utf-8

# Architecture 400-350-150-75
# With Padding, Batchnormalization and Dropout

# Import the important libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, ZeroPadding2D, Conv2D


# Import the data of training and testing from the library

# In[ ]:


# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# Check the shape of the training and testing data 

# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()
# test.head()


# Splitting the data into X and y(i.e independent and dependent varaible in simple terms)

# In[ ]:


X=train.drop(['label'], axis=1)
y=train['label']

X.shape, y.shape


# Now let us check whether the target variable is imbalanced or not in the training data

# In[ ]:


# checking manually
y.value_counts()


# Plotting the same

# In[ ]:


plt.subplots(figsize=(10,8))
plt.title('Counts of the labels')
sns.countplot(x=y)
plt.show()


# We can easily conclude from the above that the data is not unbalanced
# 
# Now we will split the data into training and testing

# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1, random_state=99)
# check the shape now
X_train.shape,X_val.shape,y_train.shape,y_val.shape,test.shape


# Here the shape of our data is not according to the CNN architecture, so we will reshape the data into CNN architecture that is (images,rows,cols,channels) Here the images will be the no of the images used , rows and columns will be the pixels of the images mentioned in the dataset descriptions which are 28 * 28 . since all images are gray scale so it will only use '1' channel

# Here we first we will convert the values of the data into float32, by which the three dataframes will get converted into a numpy array

# In[ ]:


X_train=X_train.values.astype('float32')
X_val=X_val.values.astype('float32')
test=test.values.astype('float32')


# In[ ]:


# changing the shape of X_train and y_train and test also
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val=X_val.reshape(X_val.shape[0], 28, 28, 1)
test=test.reshape(test.shape[0] , 28 , 28 , 1)


# Now check the shape again

# In[ ]:


X_train.shape,X_val.shape,test.shape


# In[ ]:


# check the maximum values in the dataset
X_train.max(),X_train.min()


# We can easily that the data range is between 0 to 255, here we need to normalize the data to bring it into the range of 0 to 1 so that our model predicts the data more efficiently

# In[ ]:


X_train=X_train/255
X_val=X_val/255
test=test/255


# In[ ]:


# check the maximum values in the dataset
X_train.max(),X_train.min()


# Now our data has been normalized, we can also scale the data by using MinMaxScaler as well as Standard Scaler
# 
# **Now we will Build the Model**
# 
# Before building the model here we will need to pass the input shape

# In[ ]:


input_shape=X_train[0].shape
input_shape


# In[ ]:


model = Sequential()
model.add(Conv2D(100,kernel_size=(3, 3), activation='relu',padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))


# model.add(Conv2D(350, kernel_size=(5, 5),activation='relu',padding='same') )
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))


# model.add(Conv2D(150, kernel_size=(5, 5),activation='relu',padding='same') )
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

model.add(Conv2D(50, kernel_size=(3, 3),activation='relu',padding='same') )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


# check the summary[":"]
model.summary()


# Compile the model

# In[ ]:


model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fitting the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'history=model.fit(X_train, y_train, batch_size=60, epochs=10, verbose=1, validation_data=(X_val,y_val))')


# Evaluate the model

# In[ ]:


# evaluating the model with testing data
loss, accuracy=model.evaluate(X_val,y_val)
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


y_pred


# In[ ]:


class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names


# In[ ]:


mat=confusion_matrix(y_val, y_pred[:4200])
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))


# In[ ]:


# predict results
results = model.predict(test)


# In[ ]:


results


# In[ ]:



# select the indix with the maximum probability
results = np.argmax(results,axis = 1)


# In[ ]:


results


# In[ ]:


results = pd.Series(results,name="Label")


# In[ ]:


results


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)


# In[ ]:


submission


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




