#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook takes the preprocessed images from the overview kernel (see input) and builds a very simple CNN for segmenting the images from the strokes. The idea is to show how a model can generally be built using Keras and how the data can be fed into and used to train a model. The kernel does nothing with validation or test splitting which is important for making a usuable model applicable to more general problems

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils.io_utils import HDF5Matrix # for reading in data
import matplotlib.pyplot as plt # showing and rendering figures
# not needed in Kaggle, but required in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


td_path = '../input/interactive-segmentation-overview/training_data.h5'

img_data = HDF5Matrix(td_path, 'images', normalizer=lambda x: x/255.0)
stroke_data = HDF5Matrix(td_path, 'strokes')
seg_data = HDF5Matrix(td_path, 'segmentation')[:]

print('image data', img_data.shape)
print('stroke data', stroke_data.shape)
print('segmentation data', seg_data.shape)
_, _, _, img_chan = img_data.shape
_, _, _, stroke_chan = stroke_data.shape


# # Setup a Simple Model
# Here we setup a simple CNN to try and segment the images

# In[3]:


from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, concatenate, MaxPool2D, UpSampling2D
from keras.layers import ZeroPadding2D, Cropping2D, AveragePooling2D as AvgPooling2D
img_in = Input((256, 256, img_chan), name = 'ImageInput')
img_norm = BatchNormalization()(img_in)
# downsample the image features
img_prep = img_in
for i in range(4):
    img_prep = Conv2D(16*(2**i), 
                      kernel_size = (5, 5), 
                      padding = 'same')(img_prep)
    img_prep = Conv2D(16*(2**i)*2, 
                      kernel_size = (3, 3), 
                      padding = 'same')(img_prep)
    if i==0:
        first_img_prep = img_prep
    img_prep = MaxPool2D((2,2))(img_prep)
    
stroke_in = Input((256, 256, stroke_chan), name = 'StrokesInput')
# downsample the strokes, we use the first convolution just to get the orientation
stroke_ds = Conv2D(32, kernel_size = (5, 5), padding = 'same')(stroke_in)
stroke_ds = MaxPool2D((16,16))(stroke_ds)
stroke_ds = Conv2D(32, kernel_size = (3, 3), padding = 'same')(stroke_ds)
# combine the stroke feature map and image feature map at much lower resolution
comb_in = concatenate([img_prep, stroke_ds], name = 'Combine')
# process the combined maps together
comb_in = Conv2D(128, kernel_size = (3, 3), padding = 'same')(comb_in)
comb_in = Conv2D(64, kernel_size = (3, 3), padding = 'same')(comb_in)
# calculate the average over the whole image for some 'global' features
comb_avg = Conv2D(32, kernel_size = (3, 3), padding = 'same')(comb_in)
comb_avg = AvgPooling2D((16, 16), strides = (1,1), padding = 'same')(comb_avg)

comb_in_avg = concatenate([comb_in, comb_avg])
# scale the combined result up
comb_out = Conv2DTranspose(32, 
                            kernel_size = (16, 16),
                           strides = (16,16),
                           padding = 'same')(comb_in_avg)
# incorporate some full scale information about the image
first_img_prep = Conv2D(32, kernel_size = (3, 3), padding = 'same')(first_img_prep)
comb_out = concatenate([comb_out, first_img_prep])
# prepare a segmentation compatible output
comb_out = Conv2D(1, kernel_size = (1, 1), padding = 'same', activation = 'sigmoid')(comb_out)
# dont try and guess near the edges, it skews the weights
comb_out = ZeroPadding2D((24, 24))(Cropping2D((24, 24))(comb_out))
seg_model = Model(inputs = [img_in, stroke_in], 
                  outputs = [comb_out])
seg_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mae'])
seg_model.summary()


# In[4]:


seg_model.fit([img_data, stroke_data], seg_data, 
              epochs = 8, 
              verbose = True,
              shuffle = 'batch',
              batch_size = 10)


# # Show Results
# Here we show the results on a test image

# In[5]:


test_idx = np.random.choice(range(img_data.shape[0]))
pred_img = seg_model.predict([img_data[test_idx:test_idx+1], stroke_data[test_idx:test_idx+1]])
gt_img = seg_data[test_idx:test_idx+1]
fig, (ax_img, ax_stroke, ax_pred, ax_seg) = plt.subplots(1, 4, figsize = (12, 4))

ax_img.imshow(255*img_data[test_idx])
ax_img.set_title('Image')
ax_img.axis('off')

def_zero_img = np.concatenate([0.1*np.ones_like(stroke_data[0][:,:,0:1]),
                               stroke_data[test_idx]],2)
ax_stroke.imshow(np.argmax(def_zero_img,-1))
ax_stroke.set_title('Strokes')
ax_stroke.axis('off')

ax_pred.imshow(pred_img[0,:,:,0], cmap = 'RdBu', vmin = 0, vmax = 1)
ax_pred.set_title('Prediction')
ax_pred.axis('off')

ax_seg.imshow(gt_img[0,:,:,0], cmap = 'RdBu', vmin = 0, vmax = 1)
ax_seg.set_title('Real Segmentation')
ax_seg.axis('off')


# In[ ]:




