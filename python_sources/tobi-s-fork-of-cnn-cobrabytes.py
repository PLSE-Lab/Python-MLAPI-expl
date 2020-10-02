#!/usr/bin/env python
# coding: utf-8

#  <h1><center><font size="6">CobraBytes | CNN with Fashion MNIST Data </font></center></h1>

# # Table of contents
# 1. [GETTING READY](#part1)
#     1. [Import Packages](#part1.1)
#     2. [Read Data](#part1.2)
#     3. [Functions Definition](#part1.3)
# 2. [DATA EXPLORATION](#part2)
#     1. [Data Description](#part2.1)
#     2. [Reconstruct Image](#part2.2)
#     3. [Data Preprocessing](#part2.3)
# 3. [MODEL TRAINING](#part3)
#     1. [THREE LAYER CNN](#part3.2)
#         1. [Three Layer CNN with Dropout](#part3.2.1)
#         2. [Three-Layer CNN - Dropout, Batch Normalization](#part3.2.2)
#         3. [Three-Layer CNN - Data Augmentation - Horizontal Flipping](#part3.2.3)
#         4. [Three-Layer CNN - Horizontal Flipping and Rotating](#part3.2.4)
#         5. [horizontal flipping on low recall categories - oversampling](#part3.2.5)
#         6. [horizontal flipping on low recall categories](#part3.2.6)
#         7. [Gaussian Noise, Dropout and Batch Normalization](#part3.2.7)
#         8. [Regularization and Data Augmentation](#part3.2.8)
#     2. [TWELVE-Layer CNN](#part3.3)
#         1. [TWELVE-Layer CNN with regularization](#part3.3.1)
#         2. [TWELVE-Layer CNN with regularization and data augmentation](#part3.3.2)     
# 4. [MODEL EVALUATION](#part4)

# ## Getting Ready <a name="part1"></a>

# ### IMPORT PACKAGES <a name="part1.1"></a>

# In[ ]:


import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn

import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# **python version 3.6**

# In[ ]:


get_ipython().system('python -V')


# ### read dataset <a name="part1.2"></a>

# In[ ]:


path="../input/fashionmnist/"
train_file = path + "fashion-mnist_train.csv"
test_file  = path + "fashion-mnist_test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


# * we would like to merge the two datasets because we are not sure if the data has been shuffled.

# In[ ]:


data_full=pd.concat([train_data,test_data])
#data_full.describe()
data_full.info()


# In[ ]:


data_full.head()


# ### Functions Definition <a name="part1.3"></a>

# In[ ]:


# this can be added in the end 
from functions_jupyter import *


# ## DATA EXPLORATION <a name="part2"></a>
# 
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
# 
# To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27. The pixel is located on row i and column j of a 28 x 28 matrix.
# For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below. 

# **Labels**
# 
# 0. T-shirt/top
# 1. Trouser
# 2. Pullover
# 3. Dress
# 4. Coat
# 5. Sandal
# 6. Shirt
# 7. Sneaker
# 8. Bag
# 9. Ankle boot.

# ### Data Description <a name="part2.1"></a>

# In[ ]:


class_dict={0:"tshirt",1:"trouser",2:"pullover",3:"dress",4:"coat",5:"sandal",6:"shirt",7:"sneaker",8:"bag",9:"ankle_boot"}
print(class_dict)


# In[ ]:


X_all,y_all=split_input_label(data_full)


# - check missing data

# In[ ]:


print("There are missing values" 
      if data_full.isnull().any().any() 
      else "There are no missing values")


# * check class inbalances.

# In[ ]:


unique, counts = np.unique(y_all,return_counts=True)
products=[class_dict[i] for i in unique]
plt.figure(figsize=(15,5))
plt.bar(products,counts)
plt.ylabel("class count")


# * each class has 7k observations in the entire dataset

# ### reconstruct images <a name="part2.2"></a>

# > reshape for image construction

# In[ ]:


X_all.shape


# In[ ]:


X_all_2d = X_all.reshape(X_all.shape[0], 28, 28)


# **Sample Images:**
# 
# showing an images of each product type (first occurrence of each category in the dataset)

# In[ ]:


_ ,index = np.unique(y_all,return_index=True)
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.title(class_dict[i],fontsize="x-large")
    plt.imshow(X_all_2d[index[i]], cmap="gray")
    plt.tight_layout()


# > to plot any single image, change the index

# In[ ]:


plt.title(class_dict[y_all[40]],fontsize="x-large")
plt.imshow(X_all_2d[40], cmap='gray')


# ### DATA PREPROCESSING <a name="part2.3"></a>

# - normalizing the pixel values to fall into the interval [0,1].

# In[ ]:


data_full.iloc[:, 1:] /= 255


# - mean subtraction:  centering data around the origin along every dimension,  to avoid suboptimal optimization and gradients being all positive caused by all positive inputs. (cited from stanford course C231n. http://cs231n.github.io/neural-networks-1/)
# 

# In[ ]:


data_full.iloc[:, 1:] -= np.mean(data_full.iloc[:, 1:])


# - splitting into training and testing set **Each time of training we resplit the full training data so that we are not implicitly training on the validation data**

# we are splitting the total dataset into training data and a testset. In order to do so we need array formats. Samples are randomly assigned to training and test set with the random seed 708145.

# In[ ]:


X_train_full, X_test, y_train_full, y_test = train_test_split(X_all, y_all, test_size=0.2,random_state=708145)


# - data reshaping: reshape the input into (item.shape[0], 28, 28, 1) and turn labels into categories

# In[ ]:


X_train_full,X_test = reshape(X_train_full,X_test)
y_train_full,y_test = reshape(y_train_full,y_test)


# - split the full training set into training and validation set

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# # MODEL TRAINING <a name="part3"></a>

# ### THREE-LAYER CNN <a name="part3.2"></a>

# For 3-Layer CNN, we carry out regularizations on the basis of Dropout according to the following order:
# - Batch Normalization
# - Data Augmentation (Horizontal Flip of image)
# - Apply Gaussian Noise

# In[ ]:


#patience setting for all models
callback=[EarlyStopping(patience=5)]


# #### THREE-Layer CNN with Dropout <a name="part3.2.1"></a>

# > model_2 is a simple CNN with three Convolutional layers. The Optimizer is AdaM. Total parameter numeber is 1.421 million. Dropout is added.

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


#define the structure of model_2
def gen_cnn_1():
    #set seed for replication
    np.random.seed(1332)
    model = Sequential()
    #layer1
    layer1_model2(model)
    #layer2
    layer2_model2(model)
    #layer 3
    layer3_model2(model)
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model


# In[ ]:


model_1 = gen_cnn_1()
model_1.summary()


# In[ ]:


#fit model_1
history_1 = model_1.fit(X_train, y_train,validation_data = (X_val, y_val),
                    batch_size=128,
                    epochs=200,
                    verbose=1,
                    callbacks=callback)


# - plot confusion matrix with recall value for training and validation set:

# In[ ]:


plot_cm_train_val(X_train,X_val,y_train,y_val,model_1)


# > recognition on shirt is poor (validation recall = 0.66 after around 20 epochs)

# > ploting accuracy and loss for training and validation set

# In[ ]:


plot_acc_loss(history_1)


# #### THREE-Layer CNN - Dropout, Batch Normalization <a name="part3.2.2"></a>

# on the basis of model 1, we regularize by adding batch normalization to introduce stochasticity at training time.

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


def gen_cnn_2():
    np.random.seed(1333)
    model= Sequential()
    #layer 1
    layer1_model2(model)
    model.add(BatchNormalization())    
    #layer 2
    layer2_model2(model)
    model.add(BatchNormalization())   
    #layer 3
    layer3_model2(model)
    model.add(BatchNormalization())    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    #model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.2,momentum=0.4),
              #metrics=['accuracy'])
    return model


# In[ ]:


model_2 = gen_cnn_2()
model_2.summary()


# In[ ]:


history_2 = model_2.fit(X_train_full, y_train_full,validation_data = (X_val, y_val),
                    batch_size=512,
                    epochs=200,
                    verbose=1,
                    callbacks=callback)


# - plot confusion matrix with recall value for training and validation set:

# In[ ]:


plot_cm_train_val(X_train,X_val,y_train,y_val,model_2)


# > plot the accuracy and loss

# In[ ]:


plot_acc_loss(history_2)


# #### Data Augmentation - Horizontal Flipping <a name="part3.2.3"></a>

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


#use imagedatagenerator for data preprocessing and data augmentation. here the data is already normalised
from keras.preprocessing.image import ImageDataGenerator


# - Since the images are slightly asymmetric in horizontal level (across vertival line), we try out horizontal flipping instead of vertical. <br>
# horizontal flipping in ImagedataGenerator will randomly flip the image, each flip is independent from each other. <br>
# This step won't extend the lenght of dataset. Here we also split the training set into training and validation set

# In[ ]:


#get generators for producing iterators
img_gen = ImageDataGenerator(horizontal_flip=True)
val_gen=ImageDataGenerator()


# > **plotting examples of flipped images** <br>
# code reference: https://www.kaggle.com/gimunu/data-augmentation-with-keras-into-cnn

# In[ ]:


def plotImages( images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(6,6))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        if img.ndim != 2:
            img = img.reshape( (28,28))
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()


# In[ ]:


augmented_images, _ = next( img_gen.flow(X_train, y_train, batch_size=4*4))
plotImages(augmented_images)


# > we use the same CNN model generated in 2.2 - model_2

# In[ ]:


bs=512 #batch_size
#generate batches of augmented training data and unaugmented validation data
#set shuffle =False for validation generator s.t. samples and val_gen has same order to produce confusion matrix
train_gen = img_gen.flow(X_train,y_train, batch_size=bs)
val_gen =val_gen.flow(X_val,y_val,batch_size=bs,shuffle=False)


# In[ ]:


model_2 = gen_cnn_2()


# In[ ]:


#fit model, set steps_per_epoch such that all data points are used
history_4=model_2.fit_generator(train_gen,steps_per_epoch=len(X_train) // bs,
                      validation_data=(X_val,y_val),epochs=200,callbacks=callback)


# - plot confusion matrix for validation set

# In[ ]:


val_pred   = model_2.predict_generator(val_gen,len(X_val)//bs+1)
cm_val=conf_matrix(val_pred,y_val)
g2 = seaborn.heatmap(cm_val,annot=True)
g2.set_title('validation confusion matrix')


# In[ ]:


plot_acc_loss(history_4)


# > the distance of model accuracy and loss between train and validation set gets smaller after 15 epochs of training.

# #### Horizontal Flipping and Rotating <a name="part3.2.4"></a>

# In[ ]:


#split training data into training and validation set. Rotate and flip training data
img_gen_rotate = ImageDataGenerator(horizontal_flip=True,rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,validation_split=0.2)
val_gen = ImageDataGenerator()


# In[ ]:


#plot randomly rotated and flipped data
augmented_images, _ = next( img_gen_rotate.flow(X_train_full, y_train_full, batch_size=4*4))
plotImages(augmented_images)


# In[ ]:


model_2 = gen_cnn_2()


# In[ ]:


#generate batches for training and validation set and fit model_2
bs=512 #batch_size
train_gen_r = img_gen_rotate.flow(X_train,y_train, batch_size=bs)
val_gen_r =val_gen.flow(X_val,y_val,batch_size=bs,shuffle=False)
history_4=model_2.fit_generator(train_gen_r,steps_per_epoch=len(X_train) // bs,
                      validation_data=(X_val,y_val), epochs=200,callbacks=callback)


# - plot confusion matrix for validation set

# In[ ]:


val_pred   = model_2.predict_generator(val_gen_r,len(X_val)//bs+1)
cm_val=conf_matrix(val_pred,y_val)
g2 = seaborn.heatmap(cm_val,annot=True)
g2.set_title('validation confusion matrix')


# > with rotation, the computation time increased tremendously.<br>
# For our dataset, it hurts the accuracy. the validation loss increased and validation accuracy decreased significantly after training on 44 epochs. <br>
# The reason can be that all images are presented in rather identical angle. Rotating the training data produced features that test data don't possess.

# #### horizontal flipping on low recall categories - oversampling <a name="part3.2.5"></a>

# - Here we split the training data according to confusion matrix into one set with high recall values and another with lower recall values.<br>
#  we only apply data augmentation to categories with recall values below 0.95 (T-shirt, Pullover, Dress, Coat and shirt), and then combine the selected categories with full unaugmented data into a larger training set.
# 

# In[ ]:


#resplit full training into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


#devide the original full data set according to categories' recall value.
# T-shirt, Pullover, Dress, Coat and shirt have recall values below 0.95
data_full_cat1=data_full.loc[data_full['label'].isin([0,2,3,4,6])]


# In[ ]:


X_all_1,y_all_1=split_input_label(data_full_cat1)
X_train_full_1, X_test_1, y_train_full_1, y_test_1 = train_test_split(X_all_1, y_all_1, test_size=0.2,random_state=708145)


# In[ ]:


#reshape data
X_train_full_1,X_test_1 = reshape(X_train_full_1,X_test_1)
y_train_full_1,y_test_1 = reshape(y_train_full_1,y_test_1)


# In[ ]:


#resplit into training and validation set
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train_full_1, y_train_full_1, test_size=0.2,random_state=708145)


# In[ ]:


#2 generators for generating iterators, apply the generator with horizontal_flip=True to badly performed categories
img_gen_cat1 = ImageDataGenerator(horizontal_flip=True)
img_gen_cat2 = ImageDataGenerator()


# In[ ]:


augmented_images, _ = next( img_gen_cat1.flow(X_train_1, y_train_1, batch_size=4*4))
plotImages(augmented_images)


# In[ ]:


#generate iterators for data from selected categories ,only flip training data of selected categories
train_gen_1 = img_gen_cat1.flow(X_train_1,y_train_1, batch_size=bs)
validation_gen_1 =img_gen_cat2.flow(X_val_1,y_val_1,batch_size=bs,shuffle=False)
#generate iterators for unflipped full category data
train_gen_2 = img_gen_cat2.flow(X_train,y_train, batch_size=bs)
validation_gen_2 =img_gen_cat2.flow(X_val,y_val,batch_size=bs,shuffle=False)


# In[ ]:


from itertools import chain
#concat iterators with augmented and unaugmented data into full iterators
train_gen_all=chain.from_iterable([train_gen_1,train_gen_2])
validation_gen_all=chain.from_iterable([validation_gen_1,validation_gen_2])


# In[ ]:


X_val_da=np.concatenate((X_val_1,X_val),axis=0)
y_val_da=np.concatenate((y_val_1,y_val),axis=0)


# In[ ]:


#fit model, set steps_per_epoch such that all data points are used
history_5=model_2.fit_generator(train_gen_all,steps_per_epoch=(len(X_train_1)+len(X_train))// bs,
                      validation_data=validation_gen_all,validation_steps=len(X_val_da)//bs,epochs=200,callbacks=callback)


# - it can be seen that by oversampling the hardly seperated categories, training data are overfitted. Next we randomly flip the hardly seperated categories without changing the total length of training data.

# In[ ]:


plot_acc_loss(history_5)


# #### horizontal flipping on low recall categories <a name="part3.2.6"></a>

# > This part we randomly flip categories that we have difficulties to seperate and remain the number of training samples to be unchanged.

# In[ ]:


#well performed catogories
data_full_cat2=data_full.loc[data_full['label'].isin([1,5,7,8,9])]


# In[ ]:


#split into inputs and labels
X_all_2,y_all_2 = split_input_label(data_full_cat2)
X_train_full_2, X_test_2, y_train_full_2, y_test_2 = train_test_split(X_all_2, y_all_2, test_size=0.2,random_state=708145)


# In[ ]:


#reshape the data
X_train_full_2,X_test_2 = reshape(X_train_full_2,X_test_2)
y_train_full_2,y_test_2 = reshape(y_train_full_2,y_test_2)


# In[ ]:


#resplit full training into training and validation set
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_full_2, y_train_full_2, test_size=0.2,random_state=708145)


# In[ ]:


#generate iterators #apply no data augmentation on well performed part
train_gen_2 = img_gen_cat2.flow(X_train_2,y_train_2, batch_size=bs)
validation_gen_2 =img_gen_cat2.flow(X_val_2,y_val_2,batch_size=bs,shuffle=False)


# In[ ]:


#concat iterators
train_gen_all=chain(train_gen_1,train_gen_2)
validation_gen_all=chain(validation_gen_1,validation_gen_2)


# In[ ]:


#fit model, set steps_per_epoch such that all data points are used
history_6=model_2.fit_generator(train_gen_all,steps_per_epoch=(len(X_train_1)+len(X_train_2))// bs,
                      validation_data=validation_gen_all,
                      validation_steps=(len(X_val_1)+len(X_val_2))// bs, epochs=200,callbacks=callback)


# - the degree of overfitting is alleviated when we maintain the full data sample size and only flip part of data. But we are still suffering from the overfitting in this case.

# In[ ]:


plot_acc_loss(history_6)


# #### Gaussian Noise, Dropout and Batch Normalization <a name="part3.2.7"></a>

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


X_val.shape


# In[ ]:


from keras.layers import GaussianNoise


# In[ ]:


#add gaussian noise to the 3-layer model
def gen_cnn_4():
    np.random.seed(1333)
    model= Sequential()
    #layer 1
    layer1_model2(model)
    model.add(BatchNormalization())    
    #layer 2
    layer2_model2(model)
    model.add(BatchNormalization())   
    #layer 3
    layer3_model2(model)
    model.add(BatchNormalization()) 
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(GaussianNoise(0.3))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    #model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.2,momentum=0.4),
              #metrics=['accuracy'])
    return model


# In[ ]:


model_4=gen_cnn_4()
gen_cnn_4().summary()


# In[ ]:


#fit model_4
history_5 = model_4.fit(X_train, y_train,
                    batch_size=512,
                    epochs=200,
                    verbose=1,
                    validation_data=(X_val,y_val),callbacks=callback)


# In[ ]:


plot_acc_loss(history_5)


# - plot confusion matrix for training and validation set

# In[ ]:


plot_cm_train_val(X_train,X_val,y_train,y_val,model_4)


# #### Regularization and Data Augmentation <a name="part3.2.8"></a>
# <br>
# Here we apply horizontal flipping, Dropout, batch normalization and Gaussian noise.

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


model_4 = gen_cnn_4()


# In[ ]:


train_gen = img_gen.flow(X_train,y_train, batch_size=bs)
val_gen =val_gen.flow(X_val,y_val,batch_size=bs,shuffle=False)


# In[ ]:


#fit with model_4 which Gaussian noise is added
history_6=model_4.fit_generator(train_gen,steps_per_epoch=len(X_train) // bs,
                      validation_data=val_gen,
                      validation_steps=len(X_val)//bs, epochs=200,callbacks=callback)


# In[ ]:


plot_acc_loss(history_6)


# ### TWELVE-Layer CNN   <a name="part3.3"></a>

# #### TWELVE-Layer CNN with regularization <a name="part3.3.1"></a>

# > Here we use optimizer RMSprop in a 12-layer CNN

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# In[ ]:


from keras.optimizers import Adam


# In[ ]:


def gen_cnn_5():
    np.random.seed(1333)
    model= Sequential()
    #layer 1-3 
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    #pooling and dropout
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    #layer 4-6
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    #pooling and dropout
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.2))
    #layer 7-9
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    #pooling and dropout
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.2))
    #layer10-12
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(BatchNormalization())
    #pooling and dropout
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation = "softmax"))
    optimizer = Adam(lr=0.01, decay=0.01)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    return model


# In[ ]:


model_5 = gen_cnn_5()
model_5.summary()


# In[ ]:


history_5 = model_5.fit(X_train_full, y_train_full,validation_data = (X_val, y_val),
                    batch_size=512,
                    epochs=200,
                    verbose=1,
                    callbacks=callback)


# - plot accuracy and loss for validation set

# In[ ]:


plot_acc_loss(history_5)


# - plot confusion matrix for validation set

# In[ ]:


plot_cm_train_val(X_train,X_val,y_train,y_val,model_5)


# #### TWELVE-Layer CNN with regularization and data augmentation <a name="part3.3.2"></a>

# In[ ]:


#resplit the training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,random_state=708145)


# > Here we apply horizontal flipping on the basis of regularization.

# In[ ]:


model_5 = gen_cnn_5()


# In[ ]:


#fit with model_5, the complex CNN
history_7=model_5.fit_generator(train_gen,steps_per_epoch=len(X_train) // bs,
                      validation_data=validation_gen,
                      validation_steps=len(X_val)//bs, epochs=200,callbacks=callback)


# ### MODEL EVALUATION  <a name="part4"></a>

# Among all the empirical result, model 3 : THREE-Layer CNN - Dropout, Batch Normalization had the best validation accuracy. Here we test its performance on test data.

# In[ ]:


history = model_2.fit(X_train, y_train,
                    batch_size=512,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, y_test),callbacks=callback)
score = model_2.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


plot_cm_train_val(X_train,X_test,y_train,y_test,model_2)

