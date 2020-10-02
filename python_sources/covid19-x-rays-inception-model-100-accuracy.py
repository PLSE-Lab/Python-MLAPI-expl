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


# ## **Looking at the input images**

# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

data_dir = "../input/covid-19-x-ray-10000-images/dataset"

normal_images = [mpimg.imread(img_path) for img_path in glob.glob(data_dir+'/normal/*')]
covid_images = [mpimg.imread(img_path) for img_path in glob.glob(data_dir+'/covid/*')]

plt.imshow(normal_images[0], cmap='gray')
plt.figure()
plt.imshow(covid_images[0], cmap='gray')


# ## **Image Augmentation using ImageDataGenerator**

# In[ ]:


from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255, 
                                   horizontal_flip=True,
                                   zoom_range=0.2,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   fill_mode='nearest',
                                   validation_split=0.25)


train_generator = train_datagen.flow_from_directory(data_dir,
                                                    target_size=(150,150),
                                                    class_mode='binary',
                                                    batch_size=3,
                                                    subset='training')
validation_generator = train_datagen.flow_from_directory(data_dir,
                                                       target_size=(150,150),
                                                       batch_size=3,
                                                       class_mode='binary',
                                                       subset='validation',
                                                       shuffle=True)


# ## **Importing Local weights**

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


# ## **Creating a pretrained model(Inception)**

# In[ ]:


local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(include_top=False, weights=None, input_shape=(150,150,3))
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable=False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output


# ## **Appending to the pretrained model**

# In[ ]:


from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras


x = keras.layers.Flatten()(last_output)

x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=pre_trained_model.input, outputs=x)


# ## **Model Compilation**

# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])


# ## **Training**

# In[ ]:


history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = 5,
            verbose = 2)


# ## **Plotting the loss and accuracy**

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

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)

plt.show()


# ## **Comparing the labels and predictions**

# In[ ]:


import numpy as np
pred = model.predict(validation_generator)
predicted_class_indices = np.argmax(pred,axis=1)
labels = dict((value,key) for key,value in validation_generator.class_indices.items())
predictions = [labels[key] for key in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)

