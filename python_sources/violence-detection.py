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


# Important libraries for working out with images preprocessing and training
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings("ignore")
import keras
from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,BatchNormalization,Flatten
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator


#  **About the Dataset**<br>
# *  The image dataset is divided into 4 categories of Knives, Rifel, Pistol Gun, and Tanks.<br>
# *  All the four categories of the dataset contains nearly 300 images each.<br>
# *  The labelling of each data that we will be getting is as follows:<br>
#  <br>
# *         Knives: [0,0,0,1]; Source of Dataset : [Knives Google Images](https://www.google.com/search?client=ubuntu&hs=wzV&channel=fs&tbm=isch&sa=1&ei=bAnpXPn9EMyAvgT9yKq4BA&q=knives&oq=knives&gs_l=img.3..35i39l2j0l8.196908.198173..198392...0.0..0.469.1060.1j0j1j1j1......0....1..gws-wiz-img.....0.JcGsBEjb3GE)<br>
# *         Pistol: [1,0,0,0]; Source of Dataset : [Pistol Google Images](https://www.google.com/search?q=gun&client=ubuntu&hs=gtV&channel=fs&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjd88SnrLbiAhVB_XMBHU6ADbMQ_AUIDigB&biw=1546&bih=818)<br>
# *         Rifel:  [0,0,1,0]; Source of Dataset : [Rifel Google Images](https://www.google.com/search?q=rifel&client=ubuntu&hs=RJB&channel=fs&source=lnms&tbm=isch&sa=X&ved=0ahUKEwi7wo69rbbiAhVVg-YKHRvJAd4Q_AUIDigB)<br>
# *         Tanks:  [0,1,0,0]; Source of Dataset : [Tanks Google Images](https://www.google.com/search?client=ubuntu&hs=nyV&channel=fs&tbm=isch&sa=1&ei=JQnpXKaaKpaRwgOrxqP4DQ&q=tanks&oq=tanks&gs_l=img.3..35i39l2j0l8.64613.67160..67306...1.0..0.1000.3454.1j1j5-1j2j1......0....1..gws-wiz-img.....0..0i67j0i10.GDshavd9sMQ)<br>
# * **In the following lines I will be training two neural network algorithms,the **first** one will be the traditional designed and the **second** one will be the transfer learning one in which I will be using **VGG19** neural network algorithm**

# * > > The main reason behind using the transfer learning model is the availability of the small dataset

# In[ ]:


# Looking for the image directories
os.listdir("../input/repository/shobhitsrivastava-ds-Violence-a245c62/Images/")


# In[ ]:


# Setting the image path
path = "../input/repository/shobhitsrivastava-ds-Violence-a245c62/Images/"


# In[ ]:


#Getting data generated from the directories through image data generator
data = ImageDataGenerator(rescale = 1./255, zoom_range = 0.3,horizontal_flip=True,rotation_range= 15).flow_from_directory(path,target_size= (224,224),color_mode= "rgb",classes= ["Rifle","tank","guns","knife images"],batch_size=90)


# In[ ]:


x,y = data.next()
plt.subplot(4,3,2)
for i in range(0,12):
    image = x[i]
    label = y[i]
    print (label)
    plt.imshow(image)
    plt.show()


# In[ ]:


len(data)


# In[ ]:


# Defining the Sequential model
model= Sequential()


# In[ ]:


#Adding up the layers of the network
model.add(Convolution2D(32,(3,3),input_shape=(224,224,3),padding = "Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.2))
model.add(Convolution2D(32,(3,3),padding = "Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.2))
model.add(Convolution2D(64,(3,3),padding = "Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(4,activation="softmax"))


# In[ ]:


# Implementing the callback function so as to stop the algorithm from the furthur traning in case the accuracy dips down
clbk= keras.callbacks.EarlyStopping(monitor='accuracy',mode='min')


# In[ ]:


#Cmpiling the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


# Printing out the model summary
model.summary()


# In[ ]:


# Training the model  
history_1 = model.fit_generator(data,steps_per_epoch=int(1273/20),epochs=10,shuffle=False,callbacks=[clbk])


# In[ ]:


data


# In[ ]:


history_1.history


# In[ ]:


model.save("Mymodel_2.h5")


# In[ ]:


loss= history_1.history["loss"]
acc= history_1.history["acc"]


# In[ ]:


# Plotting the model loss
plt.plot(loss,color="r")
plt.title("Loss progression curve")


# In[ ]:


# Plotting the model accuracy
plt.plot(acc,color="b")
plt.title(" Accuracy progression curve")


# **Transfer Learning**

# In[ ]:


#Importig the transfer learning model VGG19
from keras.applications import VGG19


# In[ ]:


# Assigning weight and input shape
model_sec=VGG19(weights="imagenet",include_top=False,input_shape=(224,224,3))
model_sec.summary()


# In[ ]:


# Generating the data
data_final = ImageDataGenerator(rescale = 1/255, zoom_range = 0.2,horizontal_flip=True,vertical_flip=True).flow_from_directory(path,target_size=(224,224),color_mode="rgb",classes=["Rifle","tank","guns","knife images"],batch_size=90)


# In[ ]:


# Making the strting top layers of the model as non-trainable
for layer in model_sec.layers:
    layer.trainable=False


# In[ ]:


model_2=model_sec.output


# In[ ]:


# Adding the last trainable layers to the model
model_2= Flatten()(model_2)
model_2= Dense(512,activation="relu")(model_2)
model_2= Dropout(0.3)(model_2)
model_2= Dense(256,activation="relu")(model_2)
model_2= Dropout(0.3)(model_2)
pred= Dense(4,activation="softmax")(model_2)
model_final =Model(input=model_sec.input,output=pred)


# In[ ]:


# Compiling the model
model_final.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[ ]:


# Training
history = model_final.fit_generator(data_final,steps_per_epoch=int(1273/80),epochs=8,shuffle=False,callbacks=[clbk])


# In[ ]:


history.history


# In[ ]:


model_final.save("Myfinal_model_2.h5")


# In[ ]:


loss_final= history.history["loss"]
acc_final = history.history["acc"]


# In[ ]:


plt.plot(loss_final,color="r")
plt.title("Loss Progression Curve")


# In[ ]:


plt.plot(acc_final,color="b")
plt.title("Accuracy Progression Curve")

