#!/usr/bin/env python
# coding: utf-8

# # Recognizing Handwritten Russian Letters with CNN
# 

# **Contents:**
# * Reading the dataset
# * Data-preprocessing
# * Visualization
# * One-hot Encoding and Train-Val Split
# * Building CNN model
# * Evaluating Model performance
# * DataFrame creation
# * Testing model with random images

# # **Reading the dataset**

# In[ ]:


import os
import warnings 
warnings.filterwarnings('ignore') #ignoring unwanted warnings

#Reading file
import h5py
file = h5py.File('../input/classification-of-handwritten-letters/LetterColorImages_123.h5')

#Finding columns
columns = list(file.keys())
print(columns)


# # **Data preprocessing**

# In[ ]:


import numpy as np
import pandas as pd

#Data-preprocessing
background = np.array(file[columns[0]])
labels = np.array(file[columns[2]])
img = np.array(file[columns[1]])

img_rows,img_cols = 32,32
num_images = len(file[columns[1]])
images = img.reshape(num_images,img_rows,img_cols,3)
images = images/255


# # **Visualization**

# In[ ]:


#Visualising the letter
import pylab as pl
pl.figure(figsize=(3,3))
var = 400
pl.title('Label:%s'%labels[var]+' Background:%s'%background[var])
pl.imshow(images[var])
pl.show()


# # **One-hot Encoding target variable and Train-Validation split**

# In[ ]:


from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
num_labels = 33 #33 alphabets in russian language

y = OneHotEncoder(categories='auto').fit_transform(labels.reshape(-1,1)).toarray().astype('int64')  #reshape(-1,1) changes horizontal vector to vertical vector
x = images
X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)


# # **Building our own CNN model**
# > **Dropouts:** It is used to speed up model and prevent overfitting of data 
# > 
# > **MaxPooling2D:** It downsamples the input representation by taking the maximum value over the window(pool_size)

# In[ ]:


#Building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

hand_model = Sequential()

#Input layers
hand_model.add(Conv2D(32, kernel_size = (3, 3),
                     activation = 'relu',
                     input_shape = (32,32,3)))
hand_model.add(Conv2D(64, (3, 3), activation = 'relu'))
hand_model.add(Conv2D(128, (4, 4), activation = 'relu'))
hand_model.add(MaxPooling2D(pool_size = (2, 2)))
hand_model.add(Dropout(0.25))
hand_model.add(Flatten())
hand_model.add(Dense(128, activation = 'relu'))
hand_model.add(Dropout(0.25))

#Output layer
hand_model.add(Dense(num_labels,activation='softmax'))


# In[ ]:


#Compiling model:
hand_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#call-backs:
best_weights = ModelCheckpoint(filepath='best_weights.hdf5',verbose=2,save_best_only=True)
reducing_LR = ReduceLROnPlateau(monitor='val_loss',patience=10,verbose=2,factor=0.75)
stopping = EarlyStopping(monitor='val_loss',patience=20,verbose=2)


# In[ ]:


#fit the model
model_history = hand_model.fit(X_train,y_train,batch_size=64,epochs=100,
                               verbose=1,validation_data=(X_val,y_val),
                               callbacks=[best_weights,reducing_LR,stopping])


# **Evaluating Model Performance**

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()


# # **Best weights performance on Validation data**

# In[ ]:


hand_model.load_weights('best_weights.hdf5')
hand_model.evaluate(X_val,y_val)


# **Creating a Dataframe for valuation**

# In[ ]:


# Read and display images
import matplotlib.pyplot as plt
import glob
import imageio
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def import_data(path,csv_file):
    data = pd.read_csv(path + csv_file)
    data['source'] = csv_file[:-4] + '/'
    return data

#Creating a Dataframe:
path = '../input/classification-of-handwritten-letters/'
csv_files = ['letters.csv','letters2.csv','letters3.csv']
data1 = import_data(path,csv_files[0])
data2 = import_data(path,csv_files[1])
data3 = import_data(path,csv_files[2])
data = pd.concat([data1,data2,data3],ignore_index=True)


del(data1,data2,data3)


# **Resulting DataFrame**

# In[ ]:


data.tail()


# *Storing Russian letters in a string*

# In[ ]:


#All letters in russian:
all_letters = ''
for i in data.letter.unique():
    all_letters += i
print(all_letters)


# **Creating methods for converting image and finding actual label**

# In[ ]:


#Preprocess image:
def to_img(filename):
    img = load_img(filename,target_size=(32,32))
    img = img_to_array(img)
    img = img.reshape(1,32,32,3)
    img = img.astype('float32')
    img = img/255.0
    return img

def actual_value(filename,df,column_name):
    file = os.path.basename(os.path.normpath(filename))
    index_row = df[df['file']==file].index[0]
    return df.loc[index_row,column_name]


# # **Testing our model with random image from dataset**

# In[ ]:


test_img = to_img(path+'letters3/09_236.png')
predicted_letter = hand_model.predict_classes(test_img)
plt.imshow(test_img[0])
print('predicted:',all_letters[predicted_letter[0]])
print('actual:',actual_value(path+'letters3/09_236.png',data,'letter'))


# # ***Testing model with my own handwritting***

# In[ ]:


my_path = '../input/myhandwritten-letters/'
test_img = to_img(my_path+'IMG-8461.jpg')
predicted_letter = hand_model.predict_classes(test_img)
plt.imshow(test_img[0])
print('predicted:',all_letters[predicted_letter[0]])
print('actual:k')


# ***If you like my notebook please do upvote :)***
# 

# ***Please be free to edit this notebook and try out your own ideas***

# In[ ]:




