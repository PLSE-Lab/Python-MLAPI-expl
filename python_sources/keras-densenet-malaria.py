#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201, DenseNet121
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import losses, optimizers, activations, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from PIL import Image

import os
from time import time

print(os.listdir("../input"))

INPUT_SHAPE = (50, 50, 3)

DATA_PATH = os.path.join("../input/", "cell_images/cell_images")

# Any results you write to the current directory are saved as output.


# In[ ]:


infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 
uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')

import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# In[ ]:


data = []
labels = []

for i in infected:
    try:
    
        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        blur = cv2.blur(np.array(resize_img) ,(10,10))
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        data.append(np.array(blur))
        labels.append(1)
        labels.append(1)
        labels.append(1)
        labels.append(1)
        
    except AttributeError:
        print('')
    
for u in uninfected:
    try:
        
        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        labels.append(0)
        labels.append(0)
        labels.append(0)
        
    except AttributeError:
        print('')


# In[ ]:


cells = np.array(data)
labels = np.array(labels)

np.save('Cells' , cells)
np.save('Labels' , labels)


# In[ ]:


from sklearn.model_selection import train_test_split

train_x , x , train_y , y = train_test_split(cells , labels , test_size = 0.2 , random_state = 111)

eval_x , test_x , eval_y , test_y = train_test_split(x , y ,test_size = 0.5 , random_state = 111)


# In[ ]:


plt.figure(1 , figsize = (15 ,5))
n = 0 
for z , j in zip([train_y , eval_y , test_y] , ['train labels','eval labels','test labels']):
    n += 1
    plt.subplot(1 , 3  , n)
    sns.countplot(x = z )
    plt.title(j)
plt.show()


# In[ ]:


from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201, DenseNet121
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import losses, optimizers, activations, metrics


# In[ ]:


class Densenet:
    def __init__(self, loss, optmizer, metrics):
        self.loss = loss
        self.optimizer = optmizer
        self.metrics = metrics
    
    def create_model(self, output_space: int) -> Model:
        base_model = DenseNet121(input_shape=INPUT_SHAPE, include_top=False)
        out0 = base_model.output
        out1 = GlobalMaxPooling2D()(out0)
        out2 = GlobalAveragePooling2D()(out0)
        out3 = Flatten()(out0)
        out = Concatenate(axis=-1)([out1, out2, out3])
        out = Dropout(0.5)(out)
        
        predictions = Dense(output_space, activation="sigmoid")(out)
        
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        #model.summary()
        return model


# In[ ]:


from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 2
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

learning_rate = LearningRateScheduler(step_decay)

rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-7)

callbacks_list = [learning_rate, rlrp]


# In[ ]:


model = Densenet(
    losses.binary_crossentropy,
    optimizers.Adam(lr=10e-5),
    ["accuracy"]
).create_model(1)


# In[ ]:


model.fit(train_x, train_y,
          validation_data=(eval_x, eval_y),
          batch_size=512,
          epochs=30,
          verbose=1)


# In[ ]:


accuracy = model.evaluate(test_x, test_y, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# In[ ]:


results = model.predict(test_x)


# In[ ]:


results = np.argmax(results,axis=1)
results = pd.Series(results,name="Label")


# In[ ]:


len(results)


# In[ ]:


submission = pd.concat([pd.Series(range(1,9647),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)


# In[ ]:




