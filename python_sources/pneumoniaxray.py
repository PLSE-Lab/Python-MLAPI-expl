#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Load other libraries necessary
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ### Let us take a loot at the image

# In[ ]:


#Read the image in the directory
img = Image.open("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0125-0001.jpeg")

#Create a numpy array for the img
img_array = np.array(img)

#Take a look at the shape
print(img_array.shape)

#display the image
imgplot = plt.imshow(img_array)


# ### Let us Resize the image to ease performance of Neural Net

# In[ ]:


#Let us resize the image to approx (128,128) maintaining aspect ratio. This would help store the images and fasten the network
def Img_resize(img,size=(128,128)):
    img = img.resize(size)
    return img
    
size = (128, 128)
img = Img_resize(img,size)


# ### Histogram Equilization

# In[ ]:


img = Image.open('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person101_bacteria_483.jpeg')
plt.imshow(img)


# In[ ]:


print('The histogram of X-Ray is given as follows:')
plt.plot(img.histogram())


# In[ ]:


#Let us equilaize the image histogram
im2 = ImageOps.equalize(img)
f, axarr = plt.subplots(1,2)
axarr[0].imshow(img)
axarr[1].imshow(im2)


# In[ ]:


plt.plot(im2.histogram())


# ### Build the training dataset 

# In[ ]:


# The image is a grey-scale image of dimension (128, 128). Let us start building the training dataset containing the pixels of image flattened and label indicating presence of pneumonia

# First let us build a binary classifciation to classify Pneumonia vs No-Pneumonia X-rays. Further we shall think of classifying Viral and Bacterial Pneumonia


# In[ ]:


# Let us build the pixel_cols required in the dataframe
pixel_cols = []
for i in range(128*128):
    pixel_cols.append('pixel'+str(i))
len(pixel_cols)


# In[ ]:


#define the remaining cols of the dataset
data_col = pixel_cols.copy()
data_col.append('X_Ray_Class')
data_col.append('Pneumonia_Class')
data_col[-5:]


# In[ ]:


#Let us build the training dataframe
train_df = pd.DataFrame(columns = data_col)
train_df


# ### Loading the training datasets into the dataframe

# In[ ]:


#First let us start by loading the NORMAL chest x_rays from the training set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename))
        #Equilize th histogram
        img = ImageOps.equalize(img)
        #Resize the image
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        img_array = img_array + ['Normal','None']
        train_df = train_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1
        print('Training data ',count,' loaded into the dataframe')


# In[ ]:


#Lets take a look at the training dataframe having Normal Dataset
print('size of train dataframe is ',train_df.shape)
train_df.head()


# In[ ]:


#Write to csv for future use. Will speed up preprocessing
train_df.to_csv('train_normal_1_df.csv',index=False)


# ### Let us load the train data corresponding to the pneumonia chest x-rays as well

# In[ ]:


#let us build a dataframe for the images with pneumonia. Combine later.
train_pneumonia_df = pd.DataFrame(columns = data_col)
train_pneumonia_df


# In[ ]:


# Let us load the PNEUMONIA chest x_rays from the training set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename)).convert('L')
       #Equilize th histogram
        img = ImageOps.equalize(img)
        #Resize the image
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        pneumonia_class = filename.split("_")[1]
        img_array = img_array + ['Pneumonia',pneumonia_class]
        train_pneumonia_df = train_pneumonia_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1        
        print('Training data ',count,' loaded into the dataframe')
        


# In[ ]:


train_pneumonia_df


# In[ ]:


train_pneumonia_df.to_csv('normal_train_pneumonia_df.csv',index=False)


# ### Let us Load the validation set as well 

# In[ ]:


val_normal_df = pd.DataFrame(columns = data_col)
val_normal_df


# In[ ]:


# Let us now load the NORMAL chest x_rays from the validation set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename)).convert('L')
        print(filename)
        img = ImageOps.equalize(img)
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        img_array = img_array + ['Normal','None']
        val_normal_df = val_normal_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1        
        print('Validation data ',count,' loaded into the dataframe')
        


# In[ ]:


val_normal_df.to_csv('/kaggle/working/val_normal_df.csv',index=False)
val_normal_df


# In[ ]:


val_pneumonia_df = pd.DataFrame(columns = data_col)
val_pneumonia_df


# In[ ]:


# Let us load the PNEUMONIA chest x_rays from the validation set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename)).convert('L')
        print(filename)
        img = ImageOps.equalize(img)
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        pneumonia_class = filename.split("_")[1]
        img_array = img_array + ['Pneumonia',pneumonia_class]
        val_pneumonia_df = val_pneumonia_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1        
        print('Validation data ',count,' loaded into the dataframe')


# In[ ]:


val_pneumonia_df.to_csv('/kaggle/working/val_pneumonia_df.csv',index=False)
val_pneumonia_df


# ### Let us Load the test set as well 

# In[ ]:


test_normal_df = pd.DataFrame(columns = data_col)
test_normal_df


# In[ ]:


# Let us now load the NORMAL chest x_rays from the testing set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename)).convert('L')
        print(filename)
        img = ImageOps.equalize(img)
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        img_array = img_array + ['Normal','None']
        test_normal_df = test_normal_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1        
        print('Test data ',count,' loaded into the dataframe')


# In[ ]:


test_normal_df.to_csv('/kaggle/working/test_normal_df.csv',index=False)
test_normal_df


# In[ ]:


test_pneumonia_df = pd.DataFrame(columns = data_col)
test_pneumonia_df


# In[ ]:


# Let us load the PNEUMONIA chest x_rays from the validation set
count = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'):
    for filename in filenames:
        img = Image.open(os.path.join(dirname, filename)).convert('L')
        print(filename)
        img = ImageOps.equalize(img)
        img = Img_resize(img)
        img_array = np.array(img).reshape(img.size[0]*img.size[1]).tolist()
        pneumonia_class = filename.split("_")[1]
        img_array = img_array + ['Pneumonia',pneumonia_class]
        test_pneumonia_df = test_pneumonia_df.append(pd.Series(img_array,index=data_col),ignore_index=True)
        count = count + 1        
        print('Test data ',count,' loaded into the dataframe')


# In[ ]:


test_pneumonia_df.to_csv('/kaggle/working/test_pneumonia_df.csv',index=False)
test_pneumonia_df


# ### In case you dont want to waste time, directly upload dataset from preloaded csv

# To avoid hours of toiling to just get the images into a dataframe, I have provided with preloaded dataset where all images have been transformed into dimensions of (128,128) and labels regarding X_Ray_Class and Pneumonia_Class have also been given. The X_Ray_Class in each of the dataframe is either 'Pneuomia' or 'Normal' indicating presence/absence of pneumonia in given X-Rays. The label Pneumonia_Class is 'None' for 'Normal' X-Rays and 'viral' or 'bacterial' for the 'Pneumonia' X-Rays.
# 
# * Training dataset:
# There are 1341 rows in the train_normal csv and 3875 rows in train_pneumonia csv. Both the dataframes have 16386 cols (16384 pixel cols + 2 Classes).
# Validation dataset:
# 
# * Testing dataset:
# There are 234 rows in the test_normal and 390 rows in test_pneumonia.Both the dataframes have 16386 cols (16384 pixel cols + 2 Classes).

# In[ ]:


#Load train,val,test dataset from preloaded_dataset
train_normal_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/train_normal_df.csv')
train_pneumonia_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/train_pneumonia_df.csv')
val_normal_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/val_normal_df.csv')
val_pneumonia_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/val_pneumonia_df.csv')
test_normal_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/test_normal_df.csv')
test_pneumonia_df = pd.read_csv('/kaggle/input/preloaded-equalized-dataset/test_pneumonia_df.csv')


# In[ ]:


train_df = pd.concat([train_normal_df,train_pneumonia_df])
train_df.reset_index(drop=True,inplace=True)
val_df = pd.concat([val_normal_df,val_pneumonia_df])
val_df.reset_index(drop=True,inplace=True)
test_df = pd.concat([test_normal_df,test_pneumonia_df])
test_df.reset_index(drop=True,inplace=True)
print('Size of train dataset is ',train_df.shape)
print('Size of validation dataset is ',val_df.shape)
print('Size of test dataset is ',test_df.shape)


# # Binary Classication

# #### Let us first build a simple CNN that classifies an X-Ray image as either Normal(0) or having Pneumonia(1).
# In this binary classification we do not care about the type of pneumonia present. We shall look into that later on. Hence the Pneumonia_Class in each dataframe is unimportant for this task.

# In[ ]:


#Import all necessary tools to build the model
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn import preprocessing

import keras.backend as K
K.set_image_data_format('channels_last')


# In[ ]:


#convert to one_hot
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# In[ ]:


train_df['X_Ray_Class']


# In[ ]:


#Encode the label. Fit and Transform
le_Y = preprocessing.LabelEncoder()
le_Y.fit(train_df['X_Ray_Class'].values)

train_df['X_Ray_Class'] = train_df['X_Ray_Class'].map(lambda x: le_Y.transform([x])[0])
val_df['X_Ray_Class'] = val_df['X_Ray_Class'].map(lambda x: le_Y.transform([x])[0])
test_df['X_Ray_Class'] = test_df['X_Ray_Class'].map(lambda x: le_Y.transform([x])[0])


# In[ ]:


#Let us create the input(X) and output Y from the given datasets

#get all pixel values into a numpy array as input of dimension (m,128,128,1)
X_train = train_df.loc[:,:'pixel16383'].to_numpy().reshape(train_df.shape[0],128,128,1)
#get the column X_Ray_Class as the output
Y_train = train_df['X_Ray_Class'].to_numpy().reshape(train_df.shape[0],)
#Perform One Hot Encoding
Y_train = convert_to_one_hot(Y_train, 2).T
#Perform same for val set
X_val = val_df.loc[:,:'pixel16383'].to_numpy().reshape(val_df.shape[0],128,128,1)
Y_val = val_df['X_Ray_Class'].to_numpy().reshape(val_df.shape[0],)
Y_val = convert_to_one_hot(Y_val, 2).T
#Perform for test set
X_test = test_df.loc[:,:'pixel16383'].to_numpy().reshape(test_df.shape[0],128,128,1)
Y_test = test_df['X_Ray_Class'].to_numpy().reshape(test_df.shape[0],)
Y_test = convert_to_one_hot(Y_test, 2).T


# In[ ]:


plt.imshow((X_train[45]/255).reshape(128,128))


# In[ ]:


X_train = X_train/255
X_val = X_val/255
X_test = X_test/255


# In[ ]:


#Print the dimensions of the train and test(dev) set to verify.
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of validation examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_val shape: " + str(X_val.shape))
print ("Y_val shape: " + str(Y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# ## Let us build the Simple Model

# In[ ]:


def simpleModel():
    img_rows, img_cols = 128, 128
    input_shape = (img_rows, img_cols, 1)
    
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), padding="same", activation="relu",input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model
    


# In[ ]:


model = simpleModel()
model.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])


# In[ ]:


model.fit(X_train,Y_train,batch_size=64,epochs=10)


# In[ ]:


#Compute the accuracy for the val test
preds = model.evaluate(X_val,Y_val,batch_size=64)

print ("Loss = " + str(preds[0]))
print ("Dev Accuracy = " + str(preds[1]))


# In[ ]:


#Compute the accuracy for the test test
preds = model.evaluate(X_test,Y_test,batch_size=64)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:


model.summary()


# ## VGG - 16 Architecture

# In[ ]:




