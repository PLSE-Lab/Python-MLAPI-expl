#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.chdir('../input')
#checking basic features of train data
df_train = pd.read_csv('fashion-mnist_train.csv')
df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


#checking basic features of test data

df_test = pd.read_csv('fashion-mnist_test.csv')
df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


#check for duplicated observation

df_train.duplicated().sum()


# In[ ]:


#Removing the duplicated values

df_train = df_train.drop_duplicates()


# In[ ]:


#Cross checking duplicated value
print(df_train.duplicated().sum())
print(df_train.shape)


# In[ ]:


#checking the values of label column
 df_train.label.value_counts()


# In[ ]:


#Creating X_train and y_train

y_train = df_train['label']
X_train = df_train.drop('label', axis =1)


# In[ ]:


#checking the statistics of y_train
print(y_train.value_counts().sort_index())
print(y_train.describe())


# In[ ]:


#import the seaborn to check the distribution of target variable

import seaborn as sns

sns.countplot(df_train.label)


# **The above plot shows a uniform distribution of the target variable**

# In[ ]:


#Checking for null values in target variable

y_train.isna().sum()


# ## Data Preprocessing

# ### Feature Scaling
# **We will be dividing all the observation by 255 as the range of pixel value lies between (0,255)**

# In[ ]:


X_train = X_train/255
df_test = df_test/255


# **Converting the target varaible into categorical form**

# In[ ]:


from keras.utils import to_categorical
y_train = to_categorical(y_train)


# In[ ]:


#Checking for unique vectors formation of labels

unique_rows = np.unique(y_train, axis = 0)
unique_rows


# ### Reshaping the Data
# **As we would be using CNN for our model fitting, we have to reshape our data as the CNN takes data in matrix form**

# In[ ]:


#Reshaping and checking the shape of the output
X_train = X_train.values.reshape(-1,28,28,1)
X_train.shape


# In[ ]:


#Checking the shape of  a single image
X_train[0].shape


# ### Spitting the Train data into training and validation set

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.1, random_state = 42) 


# In[ ]:


df_test = df_test.drop('label', axis = 1)
df_test = df_test.values.reshape(-1,28,28,1)
df_test[0].shape


# In[ ]:


import matplotlib.pyplot as plt
g = plt.imshow(X_train[1][:,:,0])


# **Now as the data is in required format we would be building are Network layers so that we can fit are model later**

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, padding = 'Same', kernel_size = (5,5), activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout (0.25))
model.add(Conv2D(filters = 64, kernel_size = (2,2), activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout (0.33))
model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = "relu", padding = "Same" ))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout (0.25))
model.add(Dense(10, activation = 'softmax'))


# In[ ]:


model.summary()


# In[ ]:


#Defining optimizer

from keras.optimizers import RMSprop

optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay =0.0)


# In[ ]:


#Compiling the model
model.compile(optimizer = optimizer, loss  = 'categorical_crossentropy', metrics =['accuracy'])


# In[ ]:


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',
                                           patience = 3,
                                           verbose = 1,
                                           factor = 0.5,
                                           min_lr = 0.00001)


# In[ ]:


#Agumenting the data 

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center = False,
                            samplewise_center = False,
                            featurewise_std_normalization= False,
                            samplewise_std_normalization= False,
                            zca_whitening= False,
                            rotation_range= 20,
                            zoom_range= 0.05,
                            width_shift_range= 0.1,
                            height_shift_range= 0.1,
                            horizontal_flip= False,
                            vertical_flip= False)

datagen.fit(X_train)


# In[ ]:


batch_size = 50
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                             epochs  = 45, validation_data= (X_val, y_val), verbose = 1,
                             steps_per_epoch= X_train.shape[0]//batch_size,
                             callbacks = [learning_rate_reduction])


# In[ ]:




fig, ax  = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color = 'b', label = 'Training loss')
ax[0].plot(history.history['val_loss'], color = 'r', label = 'Validation loss')
legend = ax[0].legend(loc ='best', shadow = True)

ax[1].plot(history.history['acc'], color = 'b', label = 'Training Accuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label = 'Validation Accuracy')
legend = ax[1].legend(loc = 'best', shadow =True)


# In[ ]:



from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_val, axis = 1)
cm = confusion_matrix(y_true, y_pred)
# plot the confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, cmap= "YlGnBu", annot=True, fmt='', ax=ax)


# In[ ]:


results = model.predict(df_test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')

