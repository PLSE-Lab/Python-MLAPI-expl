#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ../input/segmentation-models/image_classifiers-1.0.0b1/image_classifiers-1.0.0b1')
get_ipython().system('pip install ../input/segmentation-models/efficientnet-1.0.0b3/efficientnet-1.0.0b3')
get_ipython().system('pip install ../input/segmentation-models/segmentation_models/segmentation_models')


# In[ ]:


from efficientnet.keras import *
import segmentation_models as sm
import cv2
import numpy as np 

import pandas as pd
from tqdm import tqdm_notebook
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.utils import *
from keras.layers import *

from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc
import random

from classification_models.keras import Classifiers


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


IMG_SIZE = (256, 1600, 3)
CLF_IMG_SIZE = (128, 800, 3)
clf_model = 'resnet34'
unet_encoder = 'resnet34'


# In[ ]:


M, preprocess_input = Classifiers.get(clf_model) 
base_clf = M(input_shape=CLF_IMG_SIZE, weights=None, include_top=False)
x = base_clf.output
x = GlobalAveragePooling2D()(x)
x = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(x)
clf = Model(inputs=base_clf.input, outputs=x)

clf.load_weights('../input/severstal-clf/resnet34_clf_fold_4.h5')


# In[ ]:


unet = sm.Unet(unet_encoder, input_shape=IMG_SIZE, classes=5, activation='softmax', encoder_weights=None)
unet.load_weights('../input/unet-resnet34-severstal/unet_resnet34_fold_1_mixup_full_size.h5')


# In[ ]:


def rle_encoding(mask):
    
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels,[0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2:
        runs = np.append(runs,len(pixels))
    runs[1::2] -= runs[0::2]
    
    return ' '.join(str(x) for x in runs)


# In[ ]:


load_dir = '../input/severstal-steel-defect-detection/test_images/'
test_fns = os.listdir(load_dir)
d = { 'x': test_fns}
df = pd.DataFrame(data=d)


# In[ ]:


ctf_loader = ImageDataGenerator(rescale=1/255.)
unet_loader = ImageDataGenerator(rescale=1/255.)
clf_loader = ctf_loader.flow_from_dataframe(df, directory=load_dir, x_col='x', target_size=(128, 800), shuffle=False, class_mode=None, batch_size=8)
unet_loader = unet_loader.flow_from_dataframe(df, directory=load_dir, x_col='x', target_size=(256, 1600), shuffle=False, class_mode=None, batch_size=8)


# In[ ]:


get_ipython().run_cell_magic('time', '', "ImageId_ClassId = []\nEncodedPixels = []\nit = 0\nfor i in range(len(clf_loader)):\n    has_defect = clf.predict(next(clf_loader))\n    masks = unet.predict(next(unet_loader))\n    for j in range(len(has_defect)):\n        hd = has_defect[j].flatten()\n        mask = masks[j]\n        mask = np.argmax(mask, axis=-1)\n        mask = keras.utils.to_categorical(mask, num_classes=5, dtype='uint8')\n        tmp = np.asarray([ test_fns[it] +'_'+str(id) for id in range(1,5) ])\n        it += 1\n        for idx in range(1, 5):\n            ImageId_ClassId.append(tmp[idx - 1])\n            if hd[idx - 1] > 0.5:\n                EncodedPixels.append(rle_encoding(mask[:,:,idx]))\n            else:\n                EncodedPixels.append('')")


# In[ ]:


sub = { 'ImageId_ClassId': ImageId_ClassId, 'EncodedPixels': EncodedPixels}
sub = pd.DataFrame(data=sub)
sub.head(10)

sub_sample = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
sub_sample = sub_sample.drop(['EncodedPixels'], axis = 1)

submission = sub_sample.merge(sub, on = ['ImageId_ClassId'])
submission.head(10)


# In[ ]:


submission.to_csv('submission.csv', index = False)


# <a href='submission.csv'>Download</a>
