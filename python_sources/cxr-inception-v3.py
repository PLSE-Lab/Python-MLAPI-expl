#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt 
from matplotlib.image import imread 
import os
import datetime


# In[ ]:


IMAGE_SIZE = 224
CHANNELS = 1
DATADIR = '../input/chest-xray-pneumonia/chest_xray/'
test_path = DATADIR + '/test/'
valid_path = DATADIR + '/val/'
train_path = DATADIR + '/train/'
BATCH_SIZE = 16
CATEGORIES = ["NORMAL", "PNEUMONIA"]


# In[ ]:


path = os.path.join(train_path,'NORMAL')
f = os.path.join(path,os.listdir(path)[0])
img = imread(f)
plt.imshow(img,cmap="gray")


# In[ ]:


if CHANNELS == 1:
    color_mode = "grayscale"
elif CHANNELS == 3:
    color_mode = "rgb"


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1/255)
train_images = train_datagen.flow_from_directory(train_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode="binary",classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1/255)
test_images = test_datagen.flow_from_directory(test_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode="binary",classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)


# ## Using the model Inception v3

# In[ ]:


(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)


# In[ ]:


base_model = tf.keras.applications.InceptionV3(weights=None, include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)


# In[ ]:


x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)

model = tf.keras.Model(inputs = base_model.input, outputs =output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 8e-05),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# In[ ]:


history = model.fit_generator(train_images,validation_data=test_images,epochs=100,steps_per_epoch=len(train_images)/BATCH_SIZE,validation_steps=len(test_images)/BATCH_SIZE, callbacks=[tensorboard_callback])


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

