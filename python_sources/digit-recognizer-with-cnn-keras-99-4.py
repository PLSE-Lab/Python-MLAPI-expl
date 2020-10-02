#!/usr/bin/env python
# coding: utf-8

# **Digit Recognizer with CNN Keras(99.4%)**
# 
#    **Aditya Khandelwal**

# **This is a 6 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend) which is very intuitive. Firstly, I will prepare the data (handwritten digits images) then i will focus on the CNN modeling and evaluation**.
# 
# **I achieved 99.4% of accuracy with this CNN trained on my GPU(NVIDIA GTX 1050Ti). As generally in neural networks the parametrs to train are very huge so it takes a lot of time and that too we run it for multiple epochs. So it did not take much time in my computer as I trained it using my GPU which increases the computational power manifold. For ex: my GPU increased the power and reduced the time consumed to 30x times as compared to the cpu. If you train the model on cpu ite going to take hours if your epoch-no. of iterations are aroudn 20-25 that is needed for good results.**
# 
# **If you want to use GPU in your jupyter notebooks you can simply install the gpu version of tensorflow i.e tensrflow-gpu.**
# **You can read more about it here:https://medium.com/@viveksingh.heritage/how-to-install-tensorflow-gpu-version-with-jupyter-windows-10-in-8-easy-steps-8797547028a4**

# **First we will import all the packages and the libraries that we are going to use in this notebook.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
import keras
from keras import callbacks
from keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loding Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train_y = train["label"]
train_x = train.drop("label",axis = 1)


# In[ ]:


train_x.shape,train_y.shape


# In[ ]:


g = sns.countplot(train_y)


# **So we see that all the numbers are equally distributed.There is no bias towads a particular number in the dataset which is very important for our model to also not get trained biasly.**
# **If the dataset is biased to a particular number then our model will also be biased towards that number and which is not good for nay model.**

# **For building a Convolutional Neural Network we need to convert our output(y) into vector containing categorical values of the output.
# As its a kind of multiclass classification therefore we have to take 10 units in our output layer corrspondign to every unique output value and that's why we have to convert our output number into a 10 sized list containing categorical values.**

# In[ ]:


train_y = to_categorical(train_y)


# **In the MNIST Dataset the dataset is having 784 features corresponding to the 784 pixels present in the image. 
# So before passing our dataset to the Convolutional layers we have to reshape our image from a 1D array with 784 elements to a 3D list having dimensions of 28x28x1.
# The third dimension represents the channels in our image.
# You can read about channels from here : https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network**

# In[ ]:


train_x = train_x.values.reshape(len(train_x),28,28,-1)
test = test.values.reshape(len(test),28,28,-1)


# In[ ]:


train_x.shape


# **Now we will make our Keras model. Our model will be a seqeuntial model and generally all the time you can use seqeuntial model only.There is an another kind of model known as Functional API but you will be using it for very very specific tasks.
# You can read more from here:https://jovianlin.io/keras-models-sequential-vs-functional/**
# 
# **My structure of the model looks something like this:**
# 
# **(Conv2D()-->Conv2D()-->BatchNormalization()-->Maxpooling2D()-->Dropout())*2--->Flatten()-->Dense()-->Dropout()-->Dense()(output)**

# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv2D(32,(3,3), strides=(1, 1), padding='same', activation="relu",input_shape = (28,28,1),data_format = "channels_last", use_bias = True))
model.add(Conv2D(32,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))
model.add(Conv2D(64,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256,activation = "relu", use_bias = True))
model.add(Dropout(0.5))
model.add(Dense(10,activation = "softmax",use_bias = True))


# In[ ]:


optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer = optimizer,loss = "categorical_crossentropy",metrics = ['accuracy'])


# **In order to make the optimizer converge faster and closest to the global minimum of the loss function, i used an annealing method of the learning rate (LR).**
# 
# **The LR is the step by which the optimizer walks through the 'loss landscape'. The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with an high LR and the optimizer could probably fall into a local minima.**
# 
# **Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.**
# 
# **To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).**
# 
# **With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by 0.2 if the accuracy is not improved after 3 epochs.**

# In[ ]:


learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='loss',patience=3, verbose=1,factor=0.2,min_lr=0.00001)


# **DATA AUGMENTATION**
# 
# **In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.**
# 
# **For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated...**
# 
# **Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.**
# 
# **By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.**

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_x)


# **For the data augmentation, i choosed to :**
# 
# * **Randomly rotate some training images by 10 degrees**
# * **Randomly Zoom by 10% some training images**
# * **Randomly shift images horizontally by 10% of the width**
# * **Randomly shift images vertically by 10% of the height**
# * **I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.**
# 
# **Once our model is ready, we fit the training dataset .**
# 

# In[ ]:


model.fit_generator(datagen.flow(train_x,train_y,batch_size = 100),epochs = 30,steps_per_epoch=train_x.shape[0] // 100, callbacks=[learning_rate_reduction])


# In[ ]:


y_pred = model.predict(test)


# In[ ]:


y_pred


# In[ ]:


y_pred = np.array(y_pred)


# In[ ]:


y_pred


# In[ ]:


y_pred_final = []
for i in y_pred:
    y_pred_final.append(np.argmax(i))
    
    


# In[ ]:


results = pd.Series(y_pred_final,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digit_mnist.csv",index=False)


# ** If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated :)**
