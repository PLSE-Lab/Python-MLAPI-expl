#!/usr/bin/env python
# coding: utf-8

# # Inspiration
# This [Computerphile Video](https://www.youtube.com/watch?v=_DPRt3AcUEY) inpsired me to try to code a DLSS program <br/>
# I can only work with small images due to memory contraints <br/>
# I am going to try to super sample ```128x128``` images to ```512x512``` images <br/>
# This is increasing the size of the image by ```16``` times

# In[ ]:


import keras
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from random import randrange
from keras.layers import Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.models import load_model
from random import randrange


# In[ ]:


os.listdir('../input/aerial-change-detection-in-video-games/AICDDataset')


# # Data Preparation
# 
# I am compiling a list of ```128x128``` images as my input and ```256x256``` images as my output

# In[ ]:


low = []
paths = []
high = []
for r, d, f in os.walk(r'../input/aerial-change-detection-in-video-games/AICDDataset/Images_Shadows'):
    for file in f:
        if '.png' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    x = np.array(img.resize((128,128)))
    if(x.shape == (128,128,3)):
        low.append(x)
    x = np.array(img.resize((512,512)))
    if(x.shape == (512,512,3)):
        high.append(x)
del paths


# 31/100 images did not contain 3 color channels and were omitted from the training dataset

# In[ ]:


low = np.array(low)/256
low.shape


# In[ ]:


high = np.array(high)/256
high.shape


# Splitting data into train and test sets

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(low, high, test_size=0.25, shuffle=True, random_state=69)
del high
del low


# # CNN Model

# In[ ]:


model = Sequential()

model.add(Conv2D(128, kernel_size=(2,2), input_shape=(128, 128, 3), padding="same"))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(128, kernel_size=(2,2), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(UpSampling2D())

model.add(Conv2D(128, kernel_size=(4,4), padding="same"))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(128, kernel_size=(5,5), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(UpSampling2D())

model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(128, kernel_size=(2,2), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(4,4), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2,2), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(5,5), padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Conv2D(3, kernel_size=(2,2), padding="same"))
model.add(LeakyReLU(alpha=0.2))

model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())


# # Training
# I am using a callback that restores the best weights if the model starts to become over trained

# In[ ]:


# Configure the callback
checkpoint = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto', restore_best_weights=True)
callbacks_list = [checkpoint]


# In[ ]:


history = model.fit(x_train, y_train, epochs=20, batch_size=3, verbose=1,validation_data=(x_test, y_test),callbacks=callbacks_list)


# In[ ]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# # Results

# I am selecting random images from the dataset and comparing the performance of [Nearest Neighbor](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize) Interpolation vs. the DLSS Model

# In[ ]:


paths = []
for r, d, f in os.walk(r'../input/aerial-change-detection-in-video-games/AICDDataset/Images_Shadows'):
    for file in f:
        if '.png' in file:
            paths.append(os.path.join(r, file))


# In[ ]:


index = randrange(len(paths))

#select image
img = Image.open(paths[index])

#create plot
f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
axarr[0].set_xlabel('Original Image', fontsize=10)
axarr[1].set_xlabel('Interpolated Image', fontsize=10)
axarr[2].set_xlabel('Super Sampled Image', fontsize=10)

#original image downsampled
x = img.resize((128,128))
#interpolated image using Nearest Neighbor
y = x.resize((512,512),resample=Image.NEAREST)
#plotting first two images
x = np.array(x)
y = np.array(y)
axarr[0].imshow(x)
axarr[1].imshow(y)
#plotting super sampled image
x = x.reshape(1,128,128,3) / 256
result = np.array(model.predict_on_batch(x))*256
result = result.reshape(512,512,3)
result = result.astype(int)
axarr[2].imshow(result)


# In[ ]:


index = randrange(len(paths))

#select image
img = Image.open(paths[index])

#create plot
f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
axarr[0].set_xlabel('Original Image', fontsize=10)
axarr[1].set_xlabel('Interpolated Image', fontsize=10)
axarr[2].set_xlabel('Super Sampled Image', fontsize=10)

#original image downsampled
x = img.resize((128,128))
#interpolated image using Nearest Neighbor
y = x.resize((512,512),resample=Image.NEAREST)
#plotting first two images
x = np.array(x)
y = np.array(y)
axarr[0].imshow(x)
axarr[1].imshow(y)
#plotting super sampled image
x = x.reshape(1,128,128,3) / 256
result = np.array(model.predict_on_batch(x))*256
result = result.reshape(512,512,3)
result = result.astype(int)
axarr[2].imshow(result)


# In[ ]:


index = randrange(len(paths))

#select image
img = Image.open(paths[index])

#create plot
f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
axarr[0].set_xlabel('Original Image', fontsize=10)
axarr[1].set_xlabel('Interpolated Image', fontsize=10)
axarr[2].set_xlabel('Super Sampled Image', fontsize=10)

#original image downsampled
x = img.resize((128,128))
#interpolated image using Nearest Neighbor
y = x.resize((512,512),resample=Image.NEAREST)
#plotting first two images
x = np.array(x)
y = np.array(y)
axarr[0].imshow(x)
axarr[1].imshow(y)
#plotting super sampled image
x = x.reshape(1,128,128,3) / 256
result = np.array(model.predict_on_batch(x))*256
result = result.reshape(512,512,3)
result = result.astype(int)
axarr[2].imshow(result)


# # Other Datasets
# Lets see how well the model is able to generalize

# In[ ]:


paths = []
for r, d, f in os.walk(r'../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset/homer_simpson'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))


# In[ ]:


index = randrange(len(paths))

#select image
img = Image.open(paths[index])

#create plot
f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
axarr[0].set_xlabel('Original Image', fontsize=10)
axarr[1].set_xlabel('Interpolated Image', fontsize=10)
axarr[2].set_xlabel('Super Sampled Image', fontsize=10)

#original image downsampled
x = img.resize((128,128))
#interpolated image using Nearest Neighbor
y = x.resize((512,512),resample=Image.NEAREST)
#plotting first two images
x = np.array(x)
y = np.array(y)
axarr[0].imshow(x)
axarr[1].imshow(y)
#plotting super sampled image
x = x.reshape(1,128,128,3) / 256
result = np.array(model.predict_on_batch(x))*256
result = result.reshape(512,512,3)
result = result.astype(int)
axarr[2].imshow(result)


# In[ ]:


index = randrange(len(paths))

#select image
img = Image.open(paths[index])

#create plot
f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
axarr[0].set_xlabel('Original Image', fontsize=10)
axarr[1].set_xlabel('Interpolated Image', fontsize=10)
axarr[2].set_xlabel('Super Sampled Image', fontsize=10)

#original image downsampled
x = img.resize((128,128))
#interpolated image using Nearest Neighbor
y = x.resize((512,512),resample=Image.NEAREST)
#plotting first two images
x = np.array(x)
y = np.array(y)
axarr[0].imshow(x)
axarr[1].imshow(y)
#plotting super sampled image
x = x.reshape(1,128,128,3) / 256
result = np.array(model.predict_on_batch(x))*256
result = result.reshape(512,512,3)
result = result.astype(int)
axarr[2].imshow(result)

