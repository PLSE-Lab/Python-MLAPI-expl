#!/usr/bin/env python
# coding: utf-8

# # Movie Genre Just by Poster
# 
# Movie's Poster contains many vital information. It not only tells about Who are the Cast but could also tell which Genre movie belong to.
# Using this dataset I tried to build a Movie Genre Prediction just by looking the poster.
# 
# In this model I predicted top 5 Prediction of Genre
# 
# Have a look.
# 
# And if you like my work do UPVOTE. :)

# ## Importing necessary libraries

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from tqdm import tqdm


# ## Reading the dataset
# 
# I used only 2000 images to train this model here beacuse of the limitation og Kaggle's Computation.
# You could train for all images in your System or on Cloud.

# In[ ]:


data = pd.read_csv('../input/movie-classifier/Multi_Label_dataset/train.csv')
data = data.head(2000)
data.head()


# First we need to find how many different Genre are present in Dataset

# In[ ]:


Data_columns = data.columns
Data_columns


# In[ ]:


train_image = []
for i in tqdm(range(data.shape[0])):
    img = image.load_img('../input/movie-classifier/Multi_Label_dataset/Images/'+
                         data['Id'][i]+
                         '.jpg',
                         target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
    
X = np.array(train_image)


# In[ ]:


X.shape


# Split the Output Label

# In[ ]:


y = np.array(data.drop(['Id', 'Genre'],axis=1))
y.shape


# ## Train & Test Split
# 
# I used 0.1 test_size here, so that model is trained over more values

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25, test_size=0.1)


# ## Model Defination
# 
# Whole model structure which I used is build here:

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# So, now let's train the Model
# I trained for 10 Epoches and with Batch Size of 50 at a time

# In[ ]:


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=50)


# ## Let's test for some Images

# In[ ]:


img = image.load_img('../input/movie-classifier/Multi_Label_dataset/Images/tt0102428.jpg',
                     target_size=(400,400,3))
plt.imshow(img)

img = image.img_to_array(img)
img = img/255.0
img = img.reshape(1,400,400,3)

classes = data.columns[2:]
y_pred = model.predict(img)

genre =np.argsort(y_pred[0])[:-6:-1]
for i in range(5):
    print(classes[genre[i]])


# Above are the result for the Poster provided.
# 
# That's look impresive :)
# 
# Do UPVOTE the Notebook.
# And, Happy Coding
