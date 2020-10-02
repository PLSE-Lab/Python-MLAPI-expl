#!/usr/bin/env python
# coding: utf-8

# ## Loading Data

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


print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))


# ## Exploring the data

# ### Setting up directories

# In[ ]:


base_dir = '../input/chest-xray-pneumonia/chest_xray' ## our base directory where train , test and validation data is

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training pneumonia/normal pictures
train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
train_normal_dir = os.path.join(train_dir, 'NORMAL')

# Directory with our test pneumonia/normal pictures
test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
test_normal_dir = os.path.join(test_dir, 'NORMAL')


# ### let's see what file name look like 

# In[ ]:


train_pneumonia_fnames = os.listdir( train_pneumonia_dir )
train_normal_fnames = os.listdir( train_normal_dir )

print('pneumonia images',train_pneumonia_fnames[:10],'\n')
print('normal images',train_normal_fnames[:10])


# ### let's find out the total number of images in the training and test directories

# In[ ]:


print('total training pneumonia images :', len(os.listdir(train_pneumonia_dir) ))
print('total training normal images :', len(os.listdir(train_normal_dir ) ))

print('\ntotal test pneumonia images :', len(os.listdir( test_pneumonia_dir ) ))
print('total test normal images :', len(os.listdir( test_normal_dir ) ))


# ### Let's take a look at the images of both type

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images


# In[ ]:


# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_pneumonia_pix = [os.path.join(train_pneumonia_dir, fname) 
                for fname in train_pneumonia_fnames[ pic_index-8:pic_index] 
               ]

next_normal_pix = [os.path.join(train_normal_dir, fname) 
                for fname in train_normal_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_pneumonia_pix+next_normal_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# ###### here first 8 pictures are of pneumonia chest X-ray and next 8 are of a normal chest X-ray

# ## Data Preprocessing
# #### Let's set up data generators that will read pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of 32 images of size 224x224 and their labels (binary).

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 32 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    target_size=(224, 224))     
# --------------------
# Flow validation images in batches of 32 using test_datagen generator
# --------------------
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                         batch_size=32,
                                                         class_mode  = 'binary',
                                                         target_size = (224, 224))


# ## Building a model from scratch

# In[ ]:


import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 224x224 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    #dropout layer
    tf.keras.layers.Dropout(0.2),
    # Only 1 output neuron for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


history = model.fit(train_generator,
                              validation_data=test_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_steps=50)


# ## Evaluating Model

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


# #### it is clear that our model is over fitting, our training accuracy is almost 99% but test accuracy is not that great. will try to experiment with the model by including some transfer learning or better image augmentation in the next version.

# ## Predicting the X-rays 

# ### Taking look at the images in validation directory and predicting it

# In[ ]:


path_val_image = "../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1949_bacteria_4880.jpeg" # copied path of the Pneumonia X-ray image


# In[ ]:


img = mpimg.imread(path_val_image)
plt.imshow(img)
plt.show()


# In[ ]:


from keras.preprocessing import image

img = image.load_img(path_val_image, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x)
print(classes)
if classes>0.5:
    print(" pneumonia")
else:
    print("normal")


# In[ ]:




