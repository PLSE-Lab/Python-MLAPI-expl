#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv3D
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
import tensorflow.keras as keras
import datetime


# In[ ]:



train = pd.read_csv('/kaggle/input/flower-recognition-he/data/train.csv', dtype = 'str' )
train.head()


# In[ ]:


train['image_id'] = train['image_id'] + ".jpg"
train['category'].nunique()


# In[ ]:


train.head()


# In[ ]:


datagenerator = ImageDataGenerator(rescale=1./255., rotation_range=0,
                                 
                                 fill_mode = 'nearest',
                                  dtype='float32',validation_split=0.7)


# In[ ]:


train_genr = datagenerator.flow_from_dataframe(dataframe=train, directory="../input/flower-recognition-he/data/train",
                                        x_col="image_id", y_col="category",
                                        batch_size=32, shuffle=True, target_size=(150,150),
                                        seed=30, class_mode='categorical',
                                        subset='training' )

val_data_gen = datagenerator.flow_from_dataframe(dataframe=train, directory="../input/flower-recognition-he/data/train",
                                        x_col="image_id", y_col="category",
                                        batch_size=32, shuffle=True, target_size=(150,150),
                                        seed=30, class_mode='categorical',
                                        subset='validation')


# In[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[ ]:


pre_trained_model = InceptionV3(input_shape =(150,150,3 ),
                               include_top = False)

#make all the layer in pretrained model
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()


# In[ ]:



# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


# In[ ]:


last_layer =pre_trained_model.get_layer('mixed7') 
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


x = layers.Flatten()(last_output)
x = layers.Dense(512, activation = 'relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(102, activation = 'softmax')(x)

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = 'sgd', 
              loss ='categorical_crossentropy',
              metrics = ['acc'])

model.summary()


# In[ ]:


callbacks = myCallback() # Your Code Here
history = model.fit_generator(train_genr, epochs=5, 
                              validation_data = val_data_gen,
                              steps_per_epoch = 100,
                              callbacks = [callbacks], verbose =1)


# In[ ]:


def plotloss(history):
       #accuracy
   plt.plot(history.history['acc'])
   plt.plot(history.history['val_acc'])
   plt.legend(['accuracy', 'val_accuracy', ])
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.show()
   
   #loss
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('model accuracy using inceptionV3 and rmsprop')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend([ 'loss', 'validation loss'])
   plt.show()
plotloss(history)


# In[ ]:


model.save('Inception.h5')


# In[ ]:


# from tensorflow.keras.models import load_model
# model  = load_model('/kaggle/output/kaggle/working/flower/flowerIncpetion.h5')
# model.summary()


# In[ ]:


y_pred = model.predict(val_data_gen)
y_pred = np.argmax(y_pred, axis = 1)
print('Confusion Matrix')


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(val_data_gen.classes, y_pred))


# In[ ]:


a = classification_report(val_data_gen.classes, y_pred)
a


# In[ ]:


c=confusion_matrix(val_data_gen.classes, y_pred)
c


# In[ ]:


fig, ax = plt.subplots(figsize=(102,102)) 
import seaborn as sns
sns.heatmap(c, annot=True, ax = ax, fmt="d" )
plt.show()


# In[ ]:





# In[ ]:




