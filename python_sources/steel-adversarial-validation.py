#!/usr/bin/env python
# coding: utf-8

# # Steel Adversarial Validation
# In this kernel, we compare the training images to the test images for Kaggle's "Steel Defect Detection" competition. We observe that there is a significant difference. If you select an image at random with 50/50 chance of being train or test, a classifier can distinquish whether it is a train or test image with 85% accuracy! Why is that?

# # Prepare Images
# In this kernel we randomly select 1801 training images. Then we have an equal number of train and test images. Note that if we compare 1801 random train images with a different 1801 random train images, then a classifier cannot do better than 50% detection. See Appendix. 

# In[ ]:


import os, numpy as np
from PIL import Image 

TRAIN_IMG = os.listdir('../input/severstal-steel-defect-detection/train_images')
TEST_IMG = os.listdir('../input/severstal-steel-defect-detection/test_images')
print('Original train count =',len(TRAIN_IMG),', Original test count =',len(TEST_IMG))
print('New train count = 1801 , New test count = 1801')
os.mkdir('../tmp/')
os.mkdir('../tmp/train_images/')
r = np.random.choice(TRAIN_IMG,len(TEST_IMG),replace=False)
for i,f in enumerate(r):
    img = Image.open('../input/severstal-steel-defect-detection/train_images/'+f)
    img.save('../tmp/train_images/'+f)
os.mkdir('../tmp/test_images/')
for i,f in enumerate(TEST_IMG):
    img = Image.open('../input/severstal-steel-defect-detection/test_images/'+f)
    img.save('../tmp/test_images/'+f)


# # Build Adversarial Classifier
# To distinguish train images from test images, we will use pretrained Xception.

# In[ ]:


import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import layers
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt, time
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from keras import applications
base_model = applications.Xception(weights=None, input_shape=(256, 256, 3), include_top=False)
base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid")(x)
model = Model(input = base_model.input, output = predictions)
model.compile(loss='binary_crossentropy', optimizer = "adam", metrics=['accuracy'])


# In[ ]:


img_dir = '../tmp/'
img_height = 256; img_width = 256
batch_size = 32; nb_epochs = 15

train_datagen = ImageDataGenerator(rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    img_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data

annealer = LearningRateScheduler(lambda x: 0.0001 * 0.95 ** x)
h = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [annealer],
    verbose=2)


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(h.history['acc'],label='Train ACC')
plt.plot(h.history['val_acc'],label='Val ACC')
plt.title('TRAIN COMPARED WITH TEST. Training History')
plt.legend()
plt.show()


# # Conclusion
# This is concerning. What is different between the training images and test images that a classifier can tell them apart with 85% accuracy? Note that if we compare a random 1801 training images with another random 1801 training images, a classifier can not distinquish the two groups better than 50%. See Appendix below.
# 
# # Appendix
# For comparision, we demonstrate that a classifier cannot distinquish one random group of 1801 training images from another random group of 1801 training images better than 50%. Therefore it is significant that we can distinquish train from test with 85% accuracy.

# In[ ]:


get_ipython().system(' rm -r ../tmp')
# COMPARE 1801 RANDOM TRAIN WITH 1801 RANDOM TRAIN
TRAIN_IMG = os.listdir('../input/severstal-steel-defect-detection/train_images')
os.mkdir('../tmp/')
os.mkdir('../tmp/train_images/')
r = np.random.choice(TRAIN_IMG,3602,replace=False)
for i,f in enumerate(r[:1801]):
    img = Image.open('../input/severstal-steel-defect-detection/train_images/'+f)
    img.save('../tmp/train_images/'+f)
os.mkdir('../tmp/test_images/')
for i,f in enumerate(r[1801:]):
    img = Image.open('../input/severstal-steel-defect-detection/train_images/'+f)
    img.save('../tmp/test_images/'+f)
    
# BUILD XCEPTION CLASSIFIER
base_model = applications.Xception(weights=None, input_shape=(256, 256, 3), include_top=False)
base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid")(x)
model = Model(input = base_model.input, output = predictions)
model.compile(loss='binary_crossentropy', optimizer = "adam", metrics=['accuracy'])

# DATA PIPELINE
img_dir = '../tmp/'
img_height = 256; img_width = 256
batch_size = 32; nb_epochs = 15

train_datagen = ImageDataGenerator(rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    img_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data

# TRAIN CLASSIFIER
annealer = LearningRateScheduler(lambda x: 0.0001 * 0.95 ** x)
h = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [annealer],
    verbose=2)

# PLOT RESULTS
plt.figure(figsize=(15,5))
plt.plot(h.history['acc'],label='Train ACC')
plt.plot(h.history['val_acc'],label='Val ACC')
plt.title('TRAIN COMPARED WITH TRAIN. Training History')
plt.legend()
plt.show()

