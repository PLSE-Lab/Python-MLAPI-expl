#!/usr/bin/env python
# coding: utf-8

# 
# # **In this notebook you will learn how to fine tune different pretrained models like Resnet, InceptionV3, VGG**
# <br>
# <span style="color:green"> I have made this tutorial for the beginners in computer vision. Hope you would learn something new from my notebook!!</span>

# # Preparing the dataset

# In[ ]:


import glob
import numpy as np
import pandas as pd 


# In[ ]:


glob.glob('../input/intel-image-classification/seg_train/seg_train/*')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[ ]:


def prepare_dataset(path,label):
    x_train=[]
    y_train=[]
    all_images_path=glob.glob(path+'/*.jpg')
    for img_path in all_images_path :
            img=load_img(img_path, target_size=(150,150))
            img=img_to_array(img)
            img=img/255.0
            x_train.append(img)
            y_train.append(label)
    return np.array(x_train),np.array(y_train)


# In[ ]:


paths=glob.glob('../input/intel-image-classification/seg_train/seg_train/*')
l=len('../input/intel-image-classification/seg_train/seg_train/')
labels=[]
for path in paths:
    labels.append(path[l:])
    print(labels)


# In[ ]:


trainX_building, trainY_building  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/buildings/",0)
trainX_forest,trainY_forest  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/forest/",1)
trainX_glacier,trainY_glacier  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/glacier/",2)
trainX_mount,trainY_mount  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/mountain/",3)
trainX_sea,trainY_sea  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/sea/",4)
trainX_street,trainY_street  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/street/",5)

print('train building shape ', trainX_building.shape, trainY_building.shape) 
print('train forest', trainX_forest.shape ,trainY_forest.shape)
print('train glacier', trainX_glacier.shape,trainY_glacier.shape)
print('train mountain', trainX_mount.shape, trainY_mount.shape)
print('train sea',     trainX_sea.shape, trainY_sea.shape)
print('train street', trainX_street.shape ,trainY_street.shape)


# In[ ]:


x_train=np.concatenate((trainX_building,trainX_forest,trainX_glacier,trainX_mount,trainX_sea,trainX_street),axis=0)
y_train=np.concatenate((trainY_building,trainY_forest,trainY_glacier,trainY_mount,trainY_sea,trainY_street),axis=0)


# In[ ]:


print(x_train.shape)
print(y_train.shape)


# In[ ]:


# from sklearn.model_selection import train_test_split
# train_tes_split(x_train)


# Test Dataset

# In[ ]:


testX_building, testY_building  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/buildings/",0)
testX_forest,testY_forest  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/forest/",1)
testX_glacier,testY_glacier  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/glacier/",2)
testX_mount,testY_mount  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/mountain/",3)
testX_sea,testY_sea  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/sea/",4)
testX_street,testY_street  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/street/",5)

x_test=np.concatenate((testX_building,testX_forest,testX_glacier,testX_mount,testX_sea,testX_street),axis=0)
y_test=np.concatenate((testY_building,testY_forest,testY_glacier,testY_mount,testY_sea,testY_street),axis=0)


# # 2. Transfer learning - Different approaches

# You can also refer to this link. It has a documentation about fine-tuning the pretrained model
# https://keras.io/applications/

# ## 2.1 InceptionV3 

# #### 2.1.1 InceptionV3 with pretrained weights file 
# <br>You can find the ptretraiend weight's file here: https://www.kaggle.com/keras/inceptionv3

# In[ ]:


import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

local_weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
# pre_trained_model.summary()


# * the name 'mixed7' in below code is a name of one of the layers of Inceptionv3. You can check this out in the summary of the pretraine model by executing ***pre_trained_model.summary()***
# <br><br>PS : You can also choose different layer 

# In[ ]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

history=model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


# <div><span style="color:green">You can improve the accuracy by changing the architecture of the model <span></div>

# #### 2.2.2 InceptionV3 with inbuilt pretrained weights by the Imagenet

# In[ ]:


import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# local_weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = "imagenet")

# pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
# pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

history=model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


# <div>
# I guess,You would have noticed the difference between approach 2.1.1 and 2.1.2 !!
# <br>
# <br> All the things will remain same.The only change is that we used pretrained file of weights in 2.2.1 and in 2.2.2 we use 'imagenet' in this line: <br>pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
#                                 include_top = False, 
#                                 weights = "imagenet")

# ## 2.2 VGG16

# #### 2.2.1 : VGG16 with inbuilt pretrained weights by the Imagenet

# In[ ]:


from tensorflow.keras.applications import VGG16

pretrained_model=VGG16(input_shape = (150, 150, 3), 
                        include_top = False, 
                        weights = 'imagenet')

for layer in pretrained_model.layers:
     layer.trainable = False

# pretrained_model.summary()
last_layer = pretrained_model.get_layer('block5_pool')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output= last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model_vgg = Model(pretrained_model.input, x) 


model_vgg.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

# model_vgg.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


# #### 2.2.2 : VGG16 with pretrained weights file 
# You will find this file from this link: https://www.kaggle.com/keras/vgg16

# In[ ]:


file='/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model=VGG16(input_shape = (150, 150, 3), 
                        include_top = False, 
                        weights =None)
pretrained_model.load_weights(file)

for layer in pretrained_model.layers:
     layer.trainable = False

last_layer = pretrained_model.get_layer('block5_pool')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model_vgg = Model(pretrained_model.input, x) 


model_vgg.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

# model_vgg.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


# ## 2.3 ResNet50

# #### 2.3.1 Resnet with inbuilt pretrained weights by the Imagenet

# In[ ]:


from tensorflow.keras.applications import ResNet50

#step1
# file_resnet='/kaggle/input/vgg16/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model=ResNet50( input_shape=(150,150,3),
                                  include_top=False,
                                  weights='imagenet'
                                   )
#step2
for layer in pretrained_model.layers:
     layer.trainable = False

# pretrained_model.summary()
        
#step3        
last_layer = pretrained_model.get_layer('conv5_block3_out')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output = last_layer.output

#step4
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)

#step5
model_resnet = Model(pretrained_model.input, x) 

#step6
model_resnet.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

#step7
# model_resnet.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


# ### Now you would have learned how to fine tune the pretrained model... Right!
# Try to do the second approach <b>Resnet using the pretrained weight's file</b>* by your own. All the best!
# <br>Here you will find the file : https://www.kaggle.com/keras/resnet50

# In[ ]:




