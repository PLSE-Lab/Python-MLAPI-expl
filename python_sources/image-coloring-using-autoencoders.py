#!/usr/bin/env python
# coding: utf-8

# <h1>REQUIREMENTS</h1>

# In[ ]:


get_ipython().system('pip install keras==2.2.4')
get_ipython().system('pip install tensorflow==1.14.0')


# In[ ]:


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from sklearn.model_selection import train_test_split

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# # Read 2000 Classic Paintings' Dataset
# 
# 

# In[ ]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 1)
TRAIN_PATH = '../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/painting/'

train_ids = next(os.walk(TRAIN_PATH))[2]


# (Note that 86 of the train_ids have errors while being loading into our dataset, so we will just skip over them. We don't really need them.)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = np.zeros((len(train_ids)-86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)\nmissing_count = 0\nprint(\'Getting train images ... \')\nsys.stdout.flush()\nfor n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):\n    path = TRAIN_PATH + id_+\'\'\n    try:\n        img = imread(path)\n        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode=\'constant\', preserve_range=True)\n        X_train[n-missing_count] = img\n    except:\n#         print(" Problem with: "+path)\n        missing_count += 1\n\nX_train = X_train.astype(\'float32\') / 255.\nprint("Total missing: "+ str(missing_count))')


# In[ ]:


imshow(X_train[5])
plt.show()


# # Train/Test Split
# Just getting a sample of 20 images to test the model when it is done.

# In[ ]:


x_train, x_test = train_test_split(X_train, test_size=20, random_state=seed)


# # Create the Model
# 
# The model is a combination of an autoencoder and resnet classifier. The best an autoencoder by itself is just shade everything in a brownish tone. The model uses an resnet classifier to give the neural network an "idea" of what things should be colored.

# In[ ]:


inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('../input/inception-resnet-v2-weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()


# In[ ]:


def Colorize():
    embed_input = Input(shape=(1000,))
    
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

model = Colorize()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# # Data Generator Functions

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Image transformer\ndatagen = ImageDataGenerator(\n        shear_range=0.2,\n        zoom_range=0.2,\n        rotation_range=20,\n        horizontal_flip=True)\n\n#Create embedding\ndef create_inception_embedding(grayscaled_rgb):\n    def resize_gray(x):\n        return resize(x, (299, 299, 3), mode='constant')\n    grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])\n    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)\n    with inception.graph.as_default():\n        embed = inception.predict(grayscaled_rgb_resized)\n    return embed\n\n#Generate training data\ndef image_a_b_gen(dataset=x_train, batch_size = 20):\n    for batch in datagen.flow(dataset, batch_size=batch_size):\n        X_batch = rgb2gray(batch)\n        grayscaled_rgb = gray2rgb(X_batch)\n        lab_batch = rgb2lab(batch)\n        X_batch = lab_batch[:,:,:,0]\n        X_batch = X_batch.reshape(X_batch.shape+(1,))\n        Y_batch = lab_batch[:,:,:,1:] / 128\n        yield [X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch\n        ")


# # Checkpoints

# In[ ]:


# # Set a learning rate annealer
# learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.5,
#                                             min_lr=0.00001)
# filepath = "Art_Colorization_Model.h5"
# checkpoint = ModelCheckpoint(filepath,
#                              save_best_only=True,
#                              monitor='loss',
#                              mode='min')

# model_callbacks = [learning_rate_reduction,checkpoint]


# # Train the Model

# In[ ]:


# %%time
# BATCH_SIZE = 20
# model.fit_generator(image_a_b_gen(X_train,BATCH_SIZE),
#             epochs=25,
#             verbose=1,
#             steps_per_epoch=X_train.shape[0]/BATCH_SIZE,
#              callbacks=model_callbacks
#                    )


# In[ ]:


# model.save(filepath)
# model.save_weights("Art_Colorization_Weights.h5")
# inception.save(filepath)
# inception.save_weights("Inception.h5")


# In[ ]:


# from IPython.display import FileLink,FileLinks
# import pickle 

# FileLinks('.')


# # Sample the Results

# In[ ]:


model = Colorize()
model.load_weights('/kaggle/input/image-coloring-pretrained-models/Art_Colorization_Weights.h5')


# In[ ]:


sample = x_test
color_me = gray2rgb(rgb2gray(sample))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

output = model.predict([color_me, color_me_embed])
output = output * 128

decoded_imgs = np.zeros((len(output),256, 256, 3))

for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    decoded_imgs[i] = lab2rgb(cur)
    cv2.imwrite("img_"+str(i)+".jpg", lab2rgb(cur))


# In[ ]:


plt.figure(figsize=(20, 10))
for i in range(10):
    # grayscale
    plt.title("ORIGINAL")
    plt.subplot(3, 10, i + 1)
    plt.imshow(rgb2gray(x_test)[i].reshape(256, 256))
    plt.gray()
    plt.axis('off')
 
    # recolorization
    plt.title("GRAY SCALE")
    plt.subplot(3, 10, i + 1 +10)
    plt.imshow(decoded_imgs[i].reshape(256, 256,3))
    plt.axis('off')
    
    # original
    plt.title("GENERATED")
    plt.subplot(3, 10, i + 1 + 20)
    plt.imshow(x_test[i].reshape(256, 256,3))
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# In[ ]:


i=4
plt.title("RESULTS")
plt.axis('off')
plt.figure(figsize=(50,10))

plt.subplot(1, 10,1)
plt.imshow(rgb2gray(x_test)[i].reshape(256, 256))
plt.gray()
plt.axis('off')

# recolorization
plt.title("GRAY SCALE")
plt.subplot(1, 10,3)
plt.imshow(decoded_imgs[i].reshape(256, 256,3))
plt.axis('off')

# original
plt.title("GENERATED")
plt.subplot(1, 10,5)
plt.imshow(x_test[i].reshape(256, 256,3))
plt.axis('off')

plt.title("ORIGINAL")

plt.tight_layout()
plt.show()


# In[ ]:


X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
img = imread('/kaggle/input/image-coloring-pretrained-models/test.jpg')
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
X_test[0]= img
X_test = X_test.astype('float32')/255


# In[ ]:


sample = X_test
color_me = gray2rgb(rgb2gray(sample))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

output = model.predict([color_me, color_me_embed])
output = output * 128

decoded_imgs = np.zeros((len(output),256, 256, 3))

for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    decoded_imgs[i] = lab2rgb(cur)
    cv2.imwrite("img_"+str(i)+".jpg", lab2rgb(cur))


# In[ ]:


plt.title("RESULTS")
plt.axis('off')
plt.figure(figsize=(50,10))

plt.subplot(1, 10,1)
plt.imshow(X_test[0])
plt.gray()
plt.axis('off')

# recolorization
plt.title("INPUT")
plt.subplot(1, 10,3)
plt.imshow(decoded_imgs[0].reshape(256, 256,3))
plt.axis('off')

plt.title("OUTPUT")


plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




