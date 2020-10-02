#!/usr/bin/env python
# coding: utf-8

#  #                 Introduction
#     
# **Pneumonia is  the filling of air vesicles in the lung with an inflamed fluid. Viruses, bacteria, and rarely fungal infections cause it. Pneumonia can be diagnosed by examinening the X-Ray chest radiography by doctors. We will do it instead of doctors this time.**
#      
# <font color = 'red'>   
#    ## Content
#    
# 1. [Importing The Necessary Libraries](#1)
#     
# 2. [Data Pre-Processing](#2)
#     *     [Resizing](#3)
#     *     [Splitting](#4)
#     *     [Visualisation](#5)
#     *     [Nan Check](#6)
#     *     [Scaling](#7)
#     
#     
# 3. [The ANN Model](#8)
#  
#     *     [Building](#9)
#     *     [Compiling](#10)
#     *     [Fitting](#11)
#     *     [Predictions](#12)
#     *     [Evaluation](#13)
#     
#     
# 4. [The CNN Model](#14)
#     
#     *     [Building](#15)
#     *     [Compiling](#16)
#     *     [Fitting](#17)
#     *     [Predictions](#18)
#     *     [Evaluation](#19)
#     
#     
#     
# 5. [Conclusion](#20)
#     
#     *     [Compare The Results](#21)
#     *     [Prediction On Test Data](#22)
#     *     [Evaluation](#23)
#     
#     
#     

# In[ ]:





# <a id="1"></a> <br>
# ## Importing The Necessary Libraries
# <br>
# 
# **Also I will create two lists for keeping the paths.**

# In[ ]:



import seaborn as sns
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

paths_normal = []
paths_pneumonia = []

import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"):
    for filename in filenames:
        paths_normal.append(os.path.join(dirname, filename))
    
import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"):
    for filename in filenames:
        paths_pneumonia.append(os.path.join(dirname, filename))
    
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf


# <a id="2"></a> <br>
#   # Data Pre-Processing
# **Since the data unlabeled we need to label them by ourselves. So I will use 0 for normal images and 1 for Pneumonia. 1000 image will be enough to train.
# Here, I created (500,1) shaped zeros and ones and concateneted the by rows.** **And also stored the first 500 paths for each label. 
# Now I have 500-500 normal-pnemonia images and labels one after another.**
#   

# In[ ]:


label_normal = np.zeros((500,1))
label_pneumonia = np.ones((500,1))
label = list(np.concatenate((label_normal,label_pneumonia),axis = 0));
paths = paths_normal[0:500] + paths_pneumonia[0:500]


# **I created a data frame  and stored my paths and labels there. Reason I did it, I can reach the correct label for each image by using same index.**

# In[ ]:


d = {'paths': paths, 'label': label
    }
df = pd.DataFrame(data=d)


# <a id="3"></a> <br>
# ## Resizing
# **We have multi various sized images. We can't process them without resizing. 
# I resized them into 100x100 so we will have 10000 pixels for each image.**
# **I also flattened them into lines and stacked them vertically so I can use it for my ANN model.
# Later that I will reshape it for my CNN model.**

# In[ ]:




X = np.zeros((1,100*100),np.uint8)
y = np.zeros((1,1),np.uint8)
for count,ele in enumerate (df.iloc[:,0],0): 
    y_temp = df.iloc[count,1]
    y = np.vstack((y,y_temp))
    X_temp = cv.imread(ele,cv.IMREAD_GRAYSCALE) 
    X_temp = cv.resize(X_temp,(100,100)).reshape(1,100*100)
    X = np.vstack((X,X_temp))
    print("progression : %{}".format((count/10)))
    if count/10 >= 99.9:
        print("Done")
X = X[1:,:]
y = y[1:,:]
        


# <a id="4"></a> <br>
# ## Splitting
# **So far, we have 500 normal and 500 pneumonia images and labels one after another. We need to mix them to avoid overfitting. Also we will be splitted the data into test and validation.**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# <a id="5"></a> <br>
# ## Visualization
# **Let's make some visualization**

# In[ ]:


plt.figure(figsize=(15,15))
for count,i in enumerate(range(0,6),231):
    
    plt.subplot(count)
    if y_train[i]==1:
        plt.title("Pneumonia")
        plt.imshow(X_train[i,:].reshape(100,100),'gray')
        
    elif y_train[i]==0:
        plt.title("Normal")
        plt.imshow(X_train[i,:].reshape(100,100),'gray')
plt.show()        


# <a id="6"></a> <br>
# ## Nan Check
# **This is the part we check whether we have Nan data.**

# In[ ]:


isnan_train = np.isnan(X_train).all()
isnan_test = np.isnan(X_val).all()
print(isnan_train,isnan_test)


# <a id="7"></a> <br>
# ## Scaling
# **We need  to normalize the data otherwise some of them  will perform superiority on others. This is something we don't want.**

# In[ ]:


X_train,X_val = X_train[:,:]/255, X_val[:,:]/255


# <a id="8"></a> <br>
# # The ANN Model
# 
# **We start our model with Sequential. I added lots of layers to increase the accuracy. Numbers of neurons pretty intuitive so, you need to try for your own model. Since our output is 0 or 1, that means our output binary. So I chose 1 neuron and sigmoid activation function as an activator.** 

# <a id="9"></a> <br>
# ## Building

# In[ ]:


#Model
model = tf.keras.Sequential()
model.add(Dense(units = 784/2, activation = 'relu', input_dim=X_train.shape[1]))
model.add(Dense(units = 784/4, activation = 'relu'))
model.add(Dense(units = 784/8, activation = 'relu'))
model.add(Dense(units = 784/16, activation = 'relu'))
model.add(Dense(units = 784/32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))


# <a id="10"></a> <br>
# ## Compiling
# **Now, we need to choose a loss function, an optmizer function and a evaluation metric.
# Output is binary which is 0 or 1 so our loss function should be "binary_crossentropy". "Adam" and "sgd" are the most used optimizers for binary classification. I will use "sgd" since our data is not complex. You can read some extra documents in order to understand how to choose optimizer. "accuracy" is a good way to evaluate the models mostly.**

# In[ ]:


model.compile(loss="binary_crossentropy",optimizer="sgd", metrics = ['accuracy'])


# <a id="11"></a> <br>
# ## Fitting
# **We use "fit" method except we make "Data Augmentation". We need to specify the batch size and epochs here. Again, these are hyper parameters so, you should find the best values by trying them or you can use gridsearchcv.**

# In[ ]:


model.fit(X_train, y_train, batch_size=20, epochs=90)


# <a id="12"></a> <br>
# ## Predictions
# **"predict" method will be use  to make predictions. This method takes  the data we want to test as parameter. And will return float numbers between 0 and 1 which means "normal" and " pneumonia". Therefore,  we need to round the returned value to closest integer.**
# 

# In[ ]:


#Making Predictions on Test data
predicted = model.predict(X_val)
y_head_ann = [0 if i<0.5 else 1 for i in predicted]


# <a id="13"></a> <br>
# ## Evaluation
# **Accuracy is a good metric for evaluation but, will not be enough to understand  whether our model overfitted or not. We can use confusion matrix to understand it well.**

# In[ ]:



print(accuracy_score(y_val, y_head_ann))
cm_ann = confusion_matrix(y_val,y_head_ann)
sns.heatmap(cm_ann, annot=True) ;


# <a id="14"></a> <br>
# # The CNN Model
# **I want to compare the results between ANN and CNN so let's start to building the CNN model.**

# <a id="15"></a> <br>
# ## Building
# **Structure of CNN model will be like  (Conv2D->relu -> MaxPool2D -> Dropout)x2 -> Flatten -> Dense -> Dropout -> Out**
# 
# 
# **Again, CNN will start with "Sequential". Conv2D takes 3D array as input shape so we need to reshape our data. Since I resized my data into (1,10000), 100x100 will be fine as new shape. I will do that reshaping later. I will choose the hyper paramaters such as optimizer,loss function,activation,same as in the ann so we can compare.**

# In[ ]:




#Initialising the CNN
cnn = Sequential()
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[100,100,1]))  # 1 is our canal number it is just 1 because we use grayscale data
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))

#Pooling
cnn.add(layers.MaxPool2D(pool_size=2,strides=2)) #I preffered Max Pooling for this model
cnn.add(Dropout(0.2))

#Second Layer
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(Dropout(0.2))



#Flattening and bulding ANN

cnn.add(Flatten())
cnn.add(Dense(64, activation = "relu"))
cnn.add(Dense(32, activation = "relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation = "sigmoid")) 


# <a id="16"></a> <br>
# ## Compiling
# 

# In[ ]:


# Now we need to choose loss function, optimizer and compile the model
cnn.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])


# <a id="16"></a> <br>
# ## Reshaping
#  **We need to reshape our 10000 pixels to (100,100 for each) image.**

# In[ ]:


X_train = X_train.reshape(-1,100,100,1)
X_val = X_val.reshape(-1,100,100,1)


# <a id="16"></a> <br>
# ## Data Augmentation
# **Data Augmentation is a process we can make some manipulation on images such as rotating, zoom in,zoom out,shifting etc. It is important to avoid overfitting**

# In[ ]:




datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10, 
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)  


datagen.fit(X_train)


# <a id="17"></a> <br>
# ## Fitting
# **We will use fit_generator method since we did data augmentation**

# In[ ]:


cnn.fit_generator(datagen.flow(X_train,y_train, batch_size=20),epochs = 90, validation_data = (X_val,y_val),verbose = 1,steps_per_epoch=len(X_train) // 20)


# <a id="18"></a> <br>
# ## Predicton

# In[ ]:


predicted = cnn.predict(X_val)

y_head_cnn = [0 if i<0.5 else 1 for i in predicted]


# <a id="19"></a> <br>
# ## Evaluation

# In[ ]:



print(accuracy_score(y_val, y_head_cnn))
cm_cnn = confusion_matrix(y_val,y_head_cnn)
sns.heatmap(cm_cnn, annot=True) ;


# <a id="20"></a> <br>
# # Conculison
# **As a result, we can see that our CNN model was more succesful to classify X-ray chest images. So we should use the CNN model to predict the rest of the data.**

# <a id="21"></a> <br>
# ## Compare The Results
# We stored our confusion matrices in different variables so we can compare eachother.

# In[ ]:


plt.figure(figsize=(5, 5))
plt.subplot(221)
plt.title("ANN Confusion Matrix")
sns.heatmap(cm_ann, annot=True) ;

plt.subplot(222)
plt.title("CNN Confusion Matrix")
sns.heatmap(cm_cnn, annot=True) ;
plt.show()


# <a id="22"></a> <br>
# ## Prediction On Test Data
# **We need to pre process the test data in same way we did before.**

# In[ ]:


paths_normal_test = []
paths_pneumonia_test = []

import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"):
    for filename in filenames:
        paths_normal_test.append(os.path.join(dirname, filename))
    
import os
for dirname, _, filenames in os.walk("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"):
    for filename in filenames:
        paths_pneumonia_test.append(os.path.join(dirname, filename))

label_normal_test = np.zeros((500,1))
label_pneumonia_test = np.ones((500,1))
label_test = list(np.concatenate((label_normal_test,label_pneumonia_test),axis = 0));
paths_test = paths_normal_test[0:500] + paths_pneumonia_test[0:500]

d = {'paths': paths_test, 'label': label_test
    }
df_test = pd.DataFrame(data=d)

X = np.zeros((1,100*100),np.uint8)
y = np.zeros((1,1),np.uint8)
for count,ele in enumerate (df.iloc[:,0],0): 
    y_temp = df.iloc[count,1]
    y = np.vstack((y,y_temp))
    X_temp = cv.cvtColor(cv.imread(ele),cv.COLOR_BGR2GRAY)  
    X_temp = cv.resize(X_temp,(100,100)).reshape(1,100*100)
    X = np.vstack((X,X_temp))
    print("progression : %{}".format((count/10)))
    if count/10 >= 99.9:
        print("Done")
X_test = X[1:,:]
y_test = y[1:,:]
        
X_test = X_test[:,:]/255
X_test = X_test.reshape(-1,100,100,1)


# <a id="23"></a> <br>
# ## Evaluation

# In[ ]:


predicted = cnn.predict(X_test)
y_head = [0 if i<0.5 else 1 for i in predicted]

print(accuracy_score(y_test, y_head))
cm = confusion_matrix(y_test,y_head)
sns.heatmap(cm, annot=True) ;

