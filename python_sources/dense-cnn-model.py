#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import matplotlib.pyplot 
import cv2
print(os.listdir("../input"))
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator

# Any results you write to the current directory are saved as output.


# In[ ]:


training_img = []
label= []

for dir_path in glob.glob("../input/train/*"):
    image_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.png")):
        #image= load_img(image_path)
        #image = img_to_array(image)
        #image = cv2.resize(image,(64,64))
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image,(64,64))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        training_img.append(image)
        label.append(image_label)
training_img=np.array(training_img)

    


# In[ ]:



label_to_id = {v:k for k,v in enumerate(np.unique(label))}


# In[ ]:


id_to_label = {v:k for k,v in label_to_id.items()}


# In[ ]:


id_to_label


# In[ ]:


training_label_id = np.array([label_to_id[x] for x in label])


# In[ ]:


Y = np.array(training_label_id)


# In[ ]:


Y = to_categorical(Y,num_classes=12)


# In[ ]:


from keras.models import Model
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Input,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import RMSprop,Adamax
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,EarlyStopping


# In[ ]:


def conv_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter,(3,3),padding='same',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


# In[ ]:


def transition_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter,(1,1),padding='same',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2,2),strides=(2,2))(x)
    return x


# In[ ]:


def denseblock(x,concat_axis,nb_filter,nb_layers,growth_rate,dropout_rate=None,weight_decay=1E-4):
    list_features = [x]
    for i in range(nb_layers):
        x = conv_layer(x,concat_axis,growth_rate,dropout_rate=None,weight_decay=1E-4)
        list_features.append(x)
        x = Concatenate(axis=concat_axis)(list_features)
        nb_filter += growth_rate
    return x,nb_filter


# In[ ]:


def Densenet(nb_classes,img_dim,depth,nb_dense_block,nb_filter,growth_rate,
             dropout_rate=None,weight_decay=1E-4):
    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1
        
    model_input = Input(shape=img_dim)
    
    assert (depth-4)%3  == 0 , "Depth must be 4*N +3"
    
    nb_layers = int((depth-4 )/ 3) 
    
    x = Conv2D(nb_filter,(3,3),padding='same',use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)
    
    for block_id in range(nb_dense_block-1):
        
        x,nb_filter = denseblock(x,concat_axis,nb_filter,nb_layers,growth_rate,
                                 dropout_rate=None,weight_decay=1E-4)
        x = transition_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4)
        
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    x = Dense(nb_classes,activation='softmax',kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay))(x)
    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
    
    return densenet

    


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(training_img, Y, test_size = 0.05)


# In[ ]:


model =         Densenet(nb_classes=12,
                         img_dim=(64,64,3),
                         depth = 34,
                         nb_dense_block = 6,
                         growth_rate=12,
                         nb_filter=32,
                         dropout_rate=0.25,
                         weight_decay=1E-4)
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer = Adamax(),
              metrics=["accuracy"])


# In[ ]:


model_filepath = 'model.h5'
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=2, verbose=1)
msave = ModelCheckpoint(model_filepath, save_best_only=True)
#aug = ImageDataGenerator(rotation_range=180, width_shift_range=0.1, \
 #   height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
  #  horizontal_flip=True, fill_mode="nearest")


# In[ ]:


model.fit(X_train ,Y_train, batch_size=64,
               validation_data = (X_test,Y_test),
               epochs = 30,
               callbacks=[lr_reduce,annealer,msave],
               verbose = 1)


# In[ ]:





# In[ ]:


data = []
filenames = []
images = os.listdir("../input/test/")
for imageFileName in images:
        imageFullPath = os.path.join("../input/test/", imageFileName)
        img = load_img(imageFullPath)
        arr = img_to_array(img) 
        arr = cv2.resize(arr, (64,64)) 
        data.append(arr)
        filenames.append(imageFileName)


# In[ ]:


test = np.array(data)
test_x = test.reshape(test.shape[0],64,64,3)

model = load_model('model.h5')


# In[ ]:


yFit = model.predict(test_x, batch_size=10, verbose=1)


# In[ ]:


print(type(yFit)) 
print(type(filenames)) 


# In[ ]:


import csv  
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'species']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index, file in enumerate(filenames):
        classesProbs = yFit[index]
        maxIdx = 0
        maxProb = 0;
        for idx in range(0,11):
            if(classesProbs[idx] > maxProb):
                maxIdx = idx
                maxProb = classesProbs[idx]
        writer.writerow({'file': file, 'species': id_to_label[maxIdx]})
print("Writing complete")


# In[ ]:





# In[ ]:





# In[ ]:




