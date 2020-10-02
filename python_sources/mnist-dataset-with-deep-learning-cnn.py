#!/usr/bin/env python
# coding: utf-8

# Hi All,
# 
# This Notebook is like self assessment for me. Since I started learning Deep Learning concepts recently. As I am enthusiast to learn new things and apply my knowledge, I executed this Deep Learning "Hello world" dataset with Keras and Convolution Neural Network.
# 
# Thanks for visiting this Notebook, Hope it might be useful for someone else here.
# 
# I built Neural Network on MNIST hand written images pixel dataset to identify their correct label i.e number in image.
# 
# Let's Start....
# 

# # **Importing Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Loading Train and Test data**

# In[ ]:


train_df = pd.read_csv("../input/digit-recognizer/train.csv")
print(train_df.shape)
train_df.head()


# In[ ]:


test_df = pd.read_csv("../input/digit-recognizer/test.csv")
print(test_df.shape)
test_df.head()


# # **Preprocessing of Data**

# Using Countplot, visulizing different labels count in Train data. As labels vary from integer 0 to 9, it is Multi class classification problem

# In[ ]:


print(pd.value_counts(train_df['label']))
print(" ")

sns.countplot(x='label', data=train_df, palette='GnBu_r')


# # **Splitting Train data into Features and Labels**

# In[ ]:


X_train = train_df.iloc[:,1:]
y_train = train_df.iloc[:,0]

X_train.shape, y_train.shape


# # **Reshaping Features of Train and Test dataset**
# 
# dataset reshaped to (values count, height, width, channel)
# 
# for gray scale image channel is equal to 1
# for coloured image channel is equal to 3

# In[ ]:


X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

test_df = test_df.values.reshape(test_df.shape[0],28,28,1)

X_train.shape, test_df.shape


# In[ ]:


X_train[0]


# In[ ]:


y_train[0]


# # **Visualization of one image of Feature data**

# In[ ]:


plt.imshow(np.squeeze(X_train[4]), cmap=plt.get_cmap('gray'))
print('Label : ', plt.title(label=y_train[4]))
plt.show()


# # **Normalization of Train and Test features**
# 
# It is important preprocessing step. It is used to centre the data around zero mean and unit variance.

# In[ ]:


mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean)/std


# In[ ]:


mean = test_df.mean()
std = test_df.std()

test_df = (test_df - mean)/std


# # **Conversion of Labels to categorical**
# 
# Converts a class vector (integers) to binary class matrix.
# 

# In[ ]:


y_train = np_utils.to_categorical(y_train, num_classes=10)

y_train[0]


# # **Building model with Sequential method**

# In[ ]:


model = Sequential()


# # **Adding Convolution layers with Dropout and Maxpooling layers**
# 
# CNN architechture is In -> [[Conv2D -> relu]* 2 -> MaxPool2D -> Dropout] * 2 -> Flatten -> Dense -> Dropout -> Batch Normalization -> Out

# In[ ]:


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))


# # **Flattening of Convolution Output**
# 
# it is use to convert the final feature maps into a one single 1D vector

# In[ ]:


model.add(Flatten())


# # **Building Neural Network**

# In[ ]:


model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(BatchNormalization())

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(BatchNormalization())

model.add(Dense(units=10, activation='softmax'))


# # **Summary of Model**

# In[ ]:


model.summary()


# # **Compilation of Model**
# 
# Loss function helps to measure how poorly our model performs on images with known labels. It is the error rate between the observed labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy".
# 
# the Optimizer function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.
# 
# I choosed RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.
# 
# The Metric function "accuracy" is used is to evaluate the performance our model. This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# # **Fitting Model**
# 
# splitting train set into 80% part used for actual training and 20% that is used to check if the model is overfitting

# In[ ]:


history = model.fit(x=X_train, y=y_train, batch_size=200, epochs=10, validation_split=0.2)
history


# # **Visualization of Loss and Accuracy of model**

# In[ ]:


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # **Predicting model with Test dataset**

# In[ ]:


test_df.shape


# In[ ]:


predict = model.predict_classes(test_df, verbose=1)

pred_df = pd.DataFrame({"ImageId": list(range(1,len(predict)+1)), "Label": predict})

pred_df.head()


# In[ ]:


pred_df.to_csv(path_or_buf="Kaggle_mnist.csv", index=False, header=True)


# **you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated :)**
