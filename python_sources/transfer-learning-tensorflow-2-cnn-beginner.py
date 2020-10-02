#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://camo.githubusercontent.com/200d24b84fb905e680fa1ebaa71af582e3d6e24e/68747470733a2f2f64327776666f7163396779717a662e636c6f756466726f6e742e6e65742f636f6e74656e742f75706c6f6164732f323031392f30362f576562736974652d5446534465736b746f7042616e6e65722e706e67" width=800><br></center>
# 
# 
# ## I decided to create this notebook while working on [Tensorflow in Practice Specialization](https://www.coursera.org/specializations/tensorflow-in-practice) on Coursera. I highly recommend this course, especially for beginners. Most of the ideas here belong to this course.

# # 1. What is transfer learning?
# 
# 
# ## Transfer learning is a methodology where weights from a model trained on one task are taken and either used to construct a fixed feature extractor, as weight initialization and/or fine-tuning. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks.

# ![transfer_learning.png](attachment:transfer_learning.png)

# # 2. Inception V3 Deep Convolutional Architecture
# 
# 
# ## Inception V3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception V3 was trained using a dataset of 1,000 classes (See the list of classes [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)) from the original ImageNet dataset which was trained with over 1 million training images.

# <img src="https://www.researchgate.net/profile/Masoud_Mahdianpari/publication/326421398/figure/fig6/AS:649353890889730@1531829440919/Schematic-diagram-of-InceptionV3-model-compressed-view.png" width=800>

# # 3. Explore the dataset

# In[ ]:


import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as implt
from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[ ]:


train_dir = "/kaggle/input/horses-or-humans-dataset/horse-or-human/train"
test_dir = "/kaggle/input/horses-or-humans-dataset/horse-or-human/validation"

train_humans = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/humans")
train_horses = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/train/horses")

test_humans = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/validation/humans")
test_horses = os.listdir("/kaggle/input/horses-or-humans-dataset/horse-or-human/validation/horses")


# In[ ]:


print("Number of images in the train-set:", len(train_horses) + len(train_humans))
print("Number of images in the test-set:", len(test_horses) + len(test_humans))

print("\nNumber of humans in the train-set:", len(train_humans))
print("Number of horses in the train-set:", len(train_horses))

print("\nNumber of humans in the test-set:", len(test_humans))
print("Number of horses in the test-set:", len(test_horses))


# ## 3.1 Sample images in train-set

# In[ ]:


import random

fig, ax = plt.subplots(2,4, figsize=(15, 8))
for i in range(4):
    x = random.randint(0, len(train_horses))
    ax[0, i].imshow(implt.imread(train_path + '/humans/' + train_humans[x]))
    ax[1, i].imshow(implt.imread(train_path + '/horses/' + train_horses[x]))


# ## 3.2 Sample images in test-set

# In[ ]:


fig, ax = plt.subplots(2,4, figsize=(15, 8))
for i in range(4):
    x = random.randint(0, len(test_horses))
    ax[0, i].imshow(implt.imread(test_path + '/humans/' + test_humans[x]))
    ax[1, i].imshow(implt.imread(test_path + '/horses/' + test_horses[x]))


# # 4. Pre-trained model

# In[ ]:


pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False


# In[ ]:


#Commented out model summary because it's output too long. If you wonder uncomment that line and check layers yourself.
#pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# # 5. Build new layers on top of the pre-trained model

# In[ ]:


from tensorflow.keras.optimizers import RMSprop

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./ 255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# Validation or test data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1./ 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

validation_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size = 20,
                                                  class_mode = 'binary',
                                                  target_size = (150, 150))


# In[ ]:


history = model.fit(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 50,
    epochs = 5,
    validation_steps = 12,
    verbose = 2)


# # 6. Visualize accuracy scores

# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.ylim(bottom=0.8)
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

