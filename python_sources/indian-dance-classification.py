#!/usr/bin/env python
# coding: utf-8

# ![dance.jpg](attachment:dance.jpg)

# # Indian Dance Form Classification.
# 
# This notebook is my attempt of the Hackerearth deep learning contest of identifying Indian dance forms. All the credits of dataset goes to them. Although I have made some changes in the dataset. Originally there were 364 images for training data.  
# 
# 
# *********************************************************************
# 
# ### Official Details of the Contest:
# 
# - [Hackerearth Deep Learning Challenge Identify Dance Form](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/)
# 
# #### ABOUT CHALLENGE
# 
# #### Problem statement
# 
# This International Dance Day, an event management company organized an evening of Indian classical dance performances to celebrate the rich, eloquent, and elegant art of dance. Post the event, the company planned to create a microsite to promote and raise awareness among the public about these dance forms. However, identifying them from images is a tough nut to crack.
# 
# You have been appointed as a Machine Learning Engineer for this project. Build an image tagging Deep Learning model that can help the company classify these images into eight categories of Indian classical dance.
# 
# The dataset consists of 364 images belonging to 8 categories, namely **manipuri, bharatanatyam, odissi, kathakali, kathak, sattriya, kuchipudi, and mohiniyattam.**
# 
# *********************************************************************
# 
# ### Things to look forward to:
# 
# - Preparation of Training Data.
# 
# - Fine Tuning of Multiple models.
# 
# - Early stopping.
# 
# - Freezing of layers in base model 
# 
# - Addition of layers in fine tuned model0.
# 
# **This Notebok will give you an enhanced view of transfer learning with multiple pretrained models for Multiclass Image Classification using Keras.**
# 

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from sklearn.metrics import confusion_matrix
# np.random.seed(2)

from keras.utils.np_utils import to_categorical

import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_path = "../input/indian-dance-form-classification/train/"
test_path = "../input/indian-dance-form-classification/test/"

kathak = "../input/indian-dance-form-classification/train/kathak/"
odissi = "../input/indian-dance-form-classification/train/odissi/"
sattriya = "../input/indian-dance-form-classification/train/sattriya/"

kathak_path = os.listdir(kathak)
sattriya_path = os.listdir(sattriya)
odissi_path = os.listdir(odissi)


# ## Visualizing the Data.

# In[ ]:


def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
#     print(labels)
    return image[...,::-1]


plt.imshow(load_img(kathak + kathak_path[2]), cmap='gray')


# In[ ]:


plt.imshow(load_img(odissi + odissi_path[2]), cmap='gray')


# In[ ]:


plt.imshow(load_img(sattriya + sattriya_path[2]), cmap='gray')


# ## Preparing Training Data

# In[ ]:


training_data = []
IMG_SIZE = 224

datadir = "../input/indian-dance-form-classification/train/"


categories = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi',  'manipuri', 'mohiniyattam', 'odissi', 'sattriya']

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except:
                pass
create_training_data()


# In[ ]:


training_data = np.array(training_data)
print(training_data.shape)


# In[ ]:


import random

np.random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])


# In[ ]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


print(np.unique(y, return_counts = True))

print(y[1:10])


# In[ ]:


a,b = np.unique(y, return_counts = True)
print(a)
print(b)
print(categories)


# ## Analysing the Training Data:

# In[ ]:


import plotly.graph_objs as go 
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

trace = go.Bar(x = categories, y = b)
data = [trace]
layout = {"title":"Categories vs Images Distribution",
         "xaxis":{"title":"Categories","tickangle":0},
         "yaxis":{"title":"Number of Images"}}
fig = go.Figure(data = data,layout=layout)
iplot(fig)


# **The data is quite well distributed. I had done some data Augmentation and reorganized it and then uploaded on kaggle.**
# 
# - [My Updated Dataset](https://www.kaggle.com/aditya48/indian-dance-form-classification)
# 
# - Original dataset can be found [here](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/).

# In[ ]:


X = X/255.0


# ## [Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)


# In[ ]:


print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)


# In[ ]:


y_train = to_categorical(y_train, num_classes = 8)
y_test = to_categorical(y_test, num_classes = 8)


# In[ ]:


print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)


# In[ ]:


print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

train_x = tf.keras.utils.normalize(X_train,axis=1)
test_x = tf.keras.utils.normalize(X_test, axis=1)


# ## [Keras Applications](https://keras.io/api/applications/)
# 
# - Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for   prediction, feature extraction, and fine-tuning.
# 
# 

# ## MODELS:
# 
# **I will try out.**
# 
# - [VGG16](https://keras.io/api/applications/vgg/#vgg16-function)
# - [VGG19](https://keras.io/api/applications/vgg/#vgg19-function)
# - [ResNet50](https://keras.io/api/applications/vgg/#resnet50-function)
# - [ResNet101](https://keras.io/api/applications/vgg/#resnet101-function)
# - [Xception](https://keras.io/api/applications/xception/)
#  

# 
# ## 1. VGG-16:
# 

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

# For interupt the training when val loss is stagnant
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


model = keras.applications.VGG16(input_shape = (224,224,3), weights = 'imagenet',include_top=False)

for layer in model.layers:
    layer.trainable = False

last_layer = model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)

# add fully-connected & dropout layers
x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.2)(x)
x = Dense(4096, activation='relu',name='fc-2')(x)
x = Dropout(0.2)(x)

# x = Dense(4096, activation='relu',name='fc-3')(x)
# x = Dropout(0.2)(x)

# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

model2.summary()


# ### [keras.model.fit](https://keras.rstudio.com/reference/fit.html)

# In[ ]:


model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)


# hist = model2.fit(X_train,y_train, batch_size=30, epochs = 100, validation_data = (X_test,y_test), callbacks=[early_stopping])
hist = model2.fit(X_train,y_train, batch_size=30, epochs = 30, validation_data = (X_test,y_test))


# In[ ]:


# Visualizing the training. 

epochs = 30

# The uncomment everything in this cell and run it.

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# ## [VGG 19](https://keras.io/api/applications/vgg/#vgg19-function)
# 
# 

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

# For interupt the training when val loss is stagnant
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


model = keras.applications.VGG19(input_shape = (224,224,3), weights = 'imagenet',include_top=False)

for layer in model.layers:
    layer.trainable = False

last_layer = model.output
# add a global spatial average pooling layer


x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu',name='fc-3')(x)
# x = Dropout(0.5)(x)

# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

model2.summary()


# In[ ]:


model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=4)


hist = model2.fit(X_train,y_train, batch_size=30, epochs = 15, validation_data = (X_test,y_test))


# In[ ]:


## Uncomment everything once you find the number of epochs.

epochs = 15 # should be equal to the number of epochs that the training had took place.
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# ## ResNet 50:
# 
# - Proved to be useless in this case.
# - Maybe I'm doing some mistake.
# - Probably Overfitting.
# 
# ## Xception:

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Just replace Xception with ResNet50 and ResNet101 for trying these models. Honestly both models 
## performed poorly on this dataset.

model = keras.applications.Xception(input_shape = (224,224,3), weights = 'imagenet',include_top=False)
model.summary()

for layer in model.layers:
	layer.trainable = False

last_layer = model.output
# add a global spatial average pooling layer


x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers

x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu',name='fc-2')(x)
# x = Dropout(0.2)(x)


# a softmax layer for 8 classes
num_classes = 8
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

# model2.summary()


# In[ ]:


model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

hist = model2.fit(X_train,y_train, batch_size=10, epochs = 20, validation_data = (X_test, y_test))


# In[ ]:


epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# ### Model Conclusion:
# 
# - VGG16 performed pretty well with some additional layers.
# - ResNet50 was way to bigger gun for this problem.
# - Trying out VGG19 for better results.
# - Other options would be: ResNet18 and 34, Xception.

# **This is the End of the notebook from tutorial standpoint. After this it was specifically for the competition.**

# In[ ]:


# test_data = []
# img_id = []
# IMG_SIZE = 224


# def create_testing_data():
#     path = "../input/indian-dance-form-classification/test/"
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
#             new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#             test_data.append([new_array])
#             img_id.append([img])
#         except:
#             pass
# create_testing_data()


# In[ ]:


# print(len(test_data))
# test_data = np.array(test_data)
# test_data =  test_data.reshape(156,224,224,3)
# print(test_data.shape)
# test_data = test_data/255

# ## ID of Images.
# print(img_id[0])


# In[ ]:


# import pandas as pd
# image = []
# for i in img_id:
#     image.append(i[0])

# image = np.array(image)
# print(image.shape)
# test = pd.DataFrame(image, columns = ['Image'])
# test.head()


# In[ ]:


# import pandas as pd

# predict = model2.predict(test_data)
# predict = np.argmax(predict,axis = 1)
# predict = np.array(predict)
# print(predict.shape)

# print(np.unique(predict,return_counts = True))


# In[ ]:


# x = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi',  'manipuri', 'mohiniyattam','odissi','sattriya']
# y = []
# for i in predict:
#     y.append(x[i])

# print(y)
# y = np.array(y)
# pred = pd.DataFrame(y, columns = ['target'])
# pred.head()


# In[ ]:


# test_csv = pd.concat([test,pred], axis=1)
# # df.sort_values(by=['col1'])
# new = test_csv.sort_values(by=['Image'])
# print(new.head())

# new.to_csv("test.csv",columns = list(test_csv.columns),index=False)


# ## References:
# 
# - [Hackerearth Deep Learning Challenge Identify Dance Form](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/)
# 
# - [Keras FAQ](https://keras.io/getting_started/faq/)
# 
# - [Keras.io](https://keras.io/api/)
# 
# - [Keras Applications](https://keras.io/api/applications/)
# 
# 

# **Do Upvote if you found this kernel useful and feel free to copy and edit. Also if I missed something let me know in the comments.**

# In[ ]:




