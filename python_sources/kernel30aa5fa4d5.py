#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import zipfile


# In[ ]:


from google.colab import files
files.upload()


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets download -d atulyakumar98/pothole-detection-dataset')


# In[ ]:


local_zip="/content/pothole-detection-dataset.zip"
zip_ref=zipfile.ZipFile(local_zip,'r')
zip_ref.extractall("/content")
zip_ref.close()


# In[ ]:


os.mkdir("/content/train")


# In[ ]:


get_ipython().system('cp normal ~/.train')
get_ipython().system('chmod 600 ~/.train/normal')


# In[ ]:


train_normal_dir="/content/normal"
train_pothole_dir="/content/potholes"


# In[ ]:


import shutil
shutil.move("/content/normal","/content/train")


# In[ ]:


shutil.move("/content/potholes","/content/train")


# In[ ]:


pothole_name=os.listdir("/content/train/potholes")
normal_name=os.listdir("/content/train/normal")
train_dir="/content/train"


# In[ ]:


import tensorflow as tf


# In[ ]:


model=tf.keras.models.Sequential([
                                  tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPooling2D(2,2),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  tf.keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)


# In[ ]:


train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  class_mode='binary',
                                                  batch_size=20,
                                                  shuffle=True)


# In[ ]:


history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=100,
                            epochs=25,
                            verbose=2)


# In[ ]:


import matplotlib.pyplot as plt
acc=history.history['acc']
loss=history.history['loss']
epochs=range(len(acc))
plt.plot(acc,epochs,'r',label='Training accuracy')
plt.plot(loss,epochs,'b',label='Training loss')
plt.title("Training accuracy vs loss")
plt.legend()
plt.figure()
plt.show()


# In[ ]:


import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)


# In[ ]:




