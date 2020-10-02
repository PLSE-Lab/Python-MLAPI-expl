#!/usr/bin/env python
# coding: utf-8

# This a Simple tutorial on **CNN( Convolutional Neural Network)** also known as **ConvNet** with **Keras**, In this tutorial we have a collection of Butterfly Images that are based upon 10 different species of butterfly,
# Given an Image we have to classify the specie of the Butterfly, So the 10 species are :
# * '001': 'Danaus_plexippus'
# * '002': 'Heliconius_charitonius'
# * '003': 'Heliconius_erato'
# * '004': 'Junonia_coenia'
# * '005': 'Lycaena_phlaeas'
# * '006': 'Nymphalis_antiopa'
# * '007': 'Papilio_cresphontes'
# * '008': 'Pieris_rapae'
# * '009': 'Vanessa_atalanta'
# * '010': 'Vanessa_cardui' 
# In this dataset we have a collection of 832 Images, that are labeled as per their specie code
# (i.e '001': 'Danaus_plexippus' , '002': 'Heliconius_charitonius' ... and so on)

# In[ ]:


# List of Libraries that we will need

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import cv2 # Image reading and preprocessing
import keras # To Build our model
from keras.layers import Conv2D , MaxPooling2D # Getting our Layers for ConvNet
from keras.optimizers import SGD # Our Optimizer, but we will be using adam.
from keras.models import Sequential # We will be using Sequential as our model
from keras.layers import Dropout, Dense , Flatten # Our other layers
# 1 : Dropout :   will switch off some neurons in our model simoultaneously
# 2 : Dense   :   will create a Normal layer of neurons
# 3 : Fatten  :   to Flatten our output from Conv layers 
from keras.utils import to_categorical # to make data categorized like converting data into arrays
from sklearn.model_selection import train_test_split # Splitting the data into training and testing
from matplotlib.image import imread #To read the image
import os

categories = []
# Setting variable filenames to path to iterate better 
filenames = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")
for filename in filenames:
        # Splitting the file.png to get the category 
        # Suppose /kaggle/input/butterfly-dataset/leedsbutterfly/images/001000.png
        category = filename.split(".")[0]
        # This will return 001000
        categories.append(category[0:3])
        # This will append the categories with 001
        
print(categories[0:5])


# Creating a Dataframe with the file names and their respesctive categories

# In[ ]:


df = pd.DataFrame({
    "Image" : filenames,
    "Category" : categories
})
df.head()


# Getting the shape of our DataFrame

# In[ ]:


df.shape


# Finding the number of each unique specie in our dataset

# In[ ]:


df['Category'].value_counts()


# Plotting a bar graph for better visualization of what speice is dominant in our dataset

# In[ ]:


df['Category'].value_counts().plot.bar()


# Getting the First 5 occurences of Image attribute from our DataFrame

# In[ ]:


df['Image'].head()


# Reading the Images from the Folder and converting them to numpy arrays for better computation

# In[ ]:


X = []
folder_path = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")
for file in folder_path:
    
    # Reading the Image
    img = cv2.imread("/kaggle/input/butterfly-dataset/leedsbutterfly/images/"+file,cv2.IMREAD_COLOR)
    # Resizing the current Image to a dimension of (128,128,3)
    img = cv2.resize(img,(128,128))
    
    # Converting them to Numpy arrays and appending to our List X
    X.append(np.array(img))
    
# Confirming if Images are converted to our desired dimensions 
print(X[1].shape)
    


# Replacing the Category column values with their original names. 

# In[ ]:


df["Category"] = df["Category"].replace({'001': 'Danaus_plexippus', '002': 'Heliconius_charitonius', '003': 'Heliconius_erato', '004': 'Junonia_coenia', '005': 'Lycaena_phlaeas', '006': 'Nymphalis_antiopa', '007': 'Papilio_cresphontes', '008': 'Pieris_rapae', '009': 'Vanessa_atalanta', '010': 'Vanessa_cardui'}) 


# Creating a numpy array y that has Category 

# In[ ]:


y = df['Category'].values
print(y[0:5])


# Plotting our first Image from list X

# In[ ]:


plt.imshow(X[1])


# Plotting Some random Images

# In[ ]:


import random as rn
fig,ax=plt.subplots(2,5)
plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
fig.set_size_inches(15,15)

for i in range(2):
    for j in range (5):
        l=rn.randint(0,len(y))
        ax[i,j].imshow(X[l][:,:,::-1])
        ax[i,j].set_title(y[l])
        ax[i,j].set_aspect('equal')


# In[ ]:


df.head()


# In[ ]:


print(X[0:5])


# In[ ]:


print(y[0:5])


# Using LabelEncoder to convert our labels into numeric values

# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['Category'] = enc.fit_transform(df['Category'])
print(df.head())


# Converting our LabelEncoded values to numpy array

# In[ ]:


Y = df['Category'].values
print(Y[0:5])
print(Y.ndim)


# Using OneHotEncoder to encode our data so that we can use it in our model
# Suppose we have the specie as 1 i.e 'Danaus_plexippus' OneHotEncoder will encode it as 
# [1,0,0,0,0,0,0,0,0,0]
# For 2 i.e 'Heliconius_charitonius it will be
# [0,1,0,0,0,0,0,0,0,0]

# In[ ]:


Y = Y.reshape(len(Y),1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y)
print(type(Y))


# In[ ]:


Y.ndim
Y.shape
type(Y)


# In[ ]:


Y[1].shape


# In[ ]:


X[1].shape


# In[ ]:


X = np.array(X)
type(X)


# Splitting Our Dataset into Training and Tesing data 

# In[ ]:


X_train , x_test , Y_train , y_test = train_test_split(X , Y ,test_size = 0.3)


# In[ ]:


X_train.shape


# In[ ]:


Y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# Creating our Model

# In[ ]:


model = Sequential()

model.add(Conv2D(32, (5,5), activation = 'relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='softmax'))


# Will display the layers in our model, and Inputs to each layer

# In[ ]:


model.summary()


# In[ ]:


model.layers


# Providing rules for our model, i.e loss type, optimizer to be used and metrics

# In[ ]:


model.compile(loss = "categorical_crossentropy" , optimizer = 'adam' , metrics = ['accuracy'])


# Passing our data to the data

# In[ ]:


model.fit(X_train , Y_train , epochs = 30 , batch_size = 12)


# Evaluating our model on test data

# In[ ]:


loss,accuracy =  model.evaluate(x_test,y_test , batch_size = 32)

print('Test accuracy: {:2.2f}%'.format(accuracy*100))

