#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


import os
from tensorflow.keras import layers
from tensorflow.keras import Model


# In[ ]:


# Download the inception v3 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

# Print the model summary
#pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


from tensorflow.keras.optimizers import RMSprop
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.1
#x = layers.Dropout(.1)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(5, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(
    optimizer= RMSprop(lr=0.01), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)


# In[ ]:


base_dir = '/kaggle/input/football-players/players'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

train_messi_dir = os.path.join(train_dir, 'Lionel Messi')
train_ronaldo_dir = os.path.join(train_dir, 'Cristiano Ronaldo')
train_dybala_dir = os.path.join(train_dir, 'Paulo Dybala')
train_aguero_dir = os.path.join(train_dir, 'Sergio Aguero')
train_romero_dir = os.path.join(train_dir, 'Sergio Romero')

# Directory with our validation pictures
validation_messi_dir = os.path.join(validation_dir, 'Lionel Messi')
validation_ronaldo_dir = os.path.join(validation_dir, 'Cristiano Ronaldo')
validation_dybala_dir = os.path.join(validation_dir, 'Paulo Dybala')
validation_aguero_dir = os.path.join(validation_dir, 'Sergio Aguero')
validation_romero_dir = os.path.join(validation_dir, 'Sergio Romero')


# In[ ]:


print('total training messi images :', len(os.listdir( train_messi_dir ) ))
print('total training ronaldo images :', len(os.listdir(train_ronaldo_dir ) ))

print('total validation messi images :', len(os.listdir(validation_messi_dir ) ))
print('total validation ronaldo images :', len(os.listdir(validation_ronaldo_dir ) ))


# In[ ]:


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    #tf.keras.layers.MaxPooling2D(2,2), 
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    #tf.keras.layers.MaxPooling2D(2,2), 
   
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    #tf.keras.layers.Dense(1024, activation='relu'), 
    #tf.keras.layers.Dense(512, activation='relu'), 
    #tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(5, activation='softmax')  
])


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )


# In[ ]:


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=2,
                                                    class_mode='categorical',
                                                    target_size=(150, 150)
                                                    )


# In[ ]:


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size= 1,
                                                         class_mode  = 'categorical',
                                                         target_size = (150, 150))


# In[ ]:


history = model.fit_generator(
    train_generator,
    epochs= 5,
    validation_data=validation_generator
)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:


import numpy as np


from keras.preprocessing import image

path='/kaggle/input/football-players/players/test/Lionel Messi/ihgsggsmages.jpg'
img=image.load_img(path, target_size=(150, 150))
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
  
classes = model.predict(images, batch_size=10)

  
print(classes[0])

if classes[0][0]==1:
    print("Cristiano Ronaldo")
    
elif classes[0][1]==1:
    print( "Lionel Messi")
elif classes[0][2]==1:
    print("Paulo Dybala")
elif classes[0][3]==1:
    print("Sergio Aguero")
elif classes[0][4]==1:
    print("Sergio Romero")


# In[ ]:




