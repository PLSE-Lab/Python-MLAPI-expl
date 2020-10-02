#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2


# In[ ]:


os.chdir("../input/brain-mri-images-for-brain-tumor-detection/")


# In[ ]:


yes=os.listdir('./brain_tumor_dataset/yes')
no=os.listdir('./brain_tumor_dataset/no')


# In[ ]:


X_data =[]
for file in yes:
    img = cv2.imread('./brain_tumor_dataset/yes/'+file)
    face = cv2.resize(img, (224, 224) )
    (b, g, r)=cv2.split(face) 
    img=cv2.merge([r,g,b])
    X_data.append(img)

for file in no:
    img = cv2.imread('./brain_tumor_dataset/no/'+file)
    face = cv2.resize(img, (224, 224) )
    (b, g, r)=cv2.split(face) 
    img=cv2.merge([r,g,b])
    X_data.append(img)


# In[ ]:


X = np.squeeze(X_data)
X.shape


# In[ ]:


#show one training sample
from matplotlib import pyplot as plt
plt.imshow(X[5], interpolation='nearest')
plt.show()


# In[ ]:


# normalize data
X = X.astype('float32')
X /= 255


# In[ ]:


target_x=np.full(len(yes),1)
target_y=np.full(len(no),0)
data_target=np.concatenate([target_x,target_y])
data_target


# In[ ]:


len(data_target)


# In[ ]:


#split data
lenght = len(data_target)
index = np.arange(0,lenght,1)
len_train = round(lenght*0.7)
len_test = lenght - len_train
print ("train size :",len_train," test_size :",len_test)


# In[ ]:


from random import sample
X_train, y_train, X_test, y_test = [],[],[],[]
test_index = sample(set(index), len_test)

for i in range(lenght):
    if i not in test_index:
        X_train.append(X[i])
        y_train.append(data_target[i])
    else:
        X_test.append(X[i])
        y_test.append(data_target[i])

X_train = np.squeeze(X_train)
y_train = np.asarray(y_train)
#y_train = np.squeeze(y_train)
X_test = np.squeeze(X_test)
y_test = np.asarray(y_test)
#y_test = np.squeeze(y_test)

print("X_train :",X_train.shape,
      "y_train :",y_train.shape,
     "\nX_test :",X_test.shape,
     "y_test :",y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16


# In[ ]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


y_train.shape


# <h2>VGG 16<h2>

# In[ ]:


model_vgg = VGG16(weights=None, include_top=False, input_shape = (224, 224, 3))#default imagenet

#for layer in model.layers[:5]:
#    layer.trainable = True

x = model_vgg.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
final = Dense(2, activation="sigmoid")(x)
model_final = Model(input = model_vgg.input, output = final)
model_vgg.summary()


# In[ ]:


model_final.compile(loss='categorical_crossentropy',
             optimizer=Adam(),
             metrics=['acc'])


# In[ ]:


hist = model_final.fit(X_train,y_train,
         batch_size=32,
         epochs=10,
         validation_data=(X_test, y_test))


# In[ ]:


def draw_history(history,figsize=(10,5)):
    ax,_ = plt.subplots(figsize=figsize)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('modelaccuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upperleft')
    plt.show()
    ax,_ = plt.subplots(figsize=figsize)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('modelloss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upperleft')
    plt.show()
    
draw_history(hist)


# **VGG16 Transfer Learning **

# In[ ]:


model_vgg = VGG16(weights="imagenet", include_top=False, input_shape = (224, 224, 3))#default imagenet

#for layer in model.layers[:5]:
#    layer.trainable = True

x = model_vgg.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
final = Dense(2, activation="sigmoid")(x)
model_final = Model(input = model_vgg.input, output = final)
model_final.summary()


# In[ ]:


model_vgg = VGG16(weights="imagenet", include_top=False, input_shape = (224, 224, 3))#default imagenet

#for layer in model.layers[:5]:
#    layer.trainable = True

x = model_vgg.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
final = Dense(2, activation="sigmoid")(x)
model_final = Model(input = model_vgg.input, output = final)
model_final.summary()


# In[ ]:


model_final.compile(loss='categorical_crossentropy',
             optimizer=Adam(),
             metrics=['acc'])


# In[ ]:


hist = model_final.fit(X_train,y_train,
         batch_size=32,
         epochs=10,
         validation_data=(X_test, y_test))


# In[ ]:


def draw_history(history,figsize=(10,5)):
    ax,_ = plt.subplots(figsize=figsize)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('modelaccuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upperleft')
    plt.show()
    ax,_ = plt.subplots(figsize=figsize)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('modelloss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upperleft')
    plt.show()
    
draw_history(hist)

