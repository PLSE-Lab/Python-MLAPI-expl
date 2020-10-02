#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==2.0.0-beta1')


# In[ ]:


import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.preprocessing import image
import PIL


# In[ ]:


get_ipython().run_line_magic('cd', '../input/')
os.listdir()


# In[ ]:


get_ipython().system('ls')
get_ipython().system('pwd')


# In[ ]:


os.listdir('dataset/DATASET/')


# In[ ]:


print('TRAIN R IS ', len(os.listdir('dataset/DATASET/TRAIN/R')))
print('TRAIN O IS ', len(os.listdir('dataset/DATASET/TRAIN/O')))
print('TEST R IS ', len(os.listdir('dataset/DATASET/TEST/R')))
print('TEST O IS ', len(os.listdir('dataset/DATASET/TEST/O')))


# In[ ]:


from tensorflow.keras.applications import ResNet50
res = ResNet50(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet')
res.summary()


# In[ ]:


for layer in res.layers:
  print(layer, layer.trainable)


# In[ ]:


model = tf.keras.models.Sequential()
model.add(res)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
model.summary()
#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


# In[ ]:


train_gen = ImageDataGenerator(rescale=1./255.)
test_gen = ImageDataGenerator(rescale=1./255.)

train_dir = 'dataset/DATASET/TRAIN/'
test_dir = 'dataset/DATASET/TEST/'

train_generator = train_gen.flow_from_directory(train_dir, batch_size = 32, target_size = (224, 224), class_mode = 'binary')
test_generator = test_gen.flow_from_directory(test_dir, batch_size = 32, target_size = (224, 224), class_mode = 'binary')

print(train_generator)
print(test_generator)


# In[ ]:


hist = model.fit_generator(train_generator, epochs = 15, validation_data = test_generator)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
acc = hist.history['acc']
loss = hist.history['loss']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Testing Accuracy")
plt.title('Training vs Testing Accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Testing Loss")
plt.title('Training vs Testing Loss')
plt.show()


# In[ ]:


get_ipython().system('nvidia-smi')
tf.keras.backend.clear_session()


# In[ ]:


im = PIL.Image.open('dataset/DATASET/TEST/R/R_10521.jpg')
print(im.size)


# In[ ]:


get_ipython().run_line_magic('cd', '')
get_ipython().system('ls')
get_ipython().system('pwd')


# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/')
get_ipython().system('ls')
get_ipython().system('pwd')


# In[ ]:


model.save('eTrash.h5')


# In[ ]:


get_ipython().system('ls')

