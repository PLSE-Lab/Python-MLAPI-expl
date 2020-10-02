#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import skimage.io
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa


from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,TimeDistributed,Flatten,Input,Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np
import gc
from PIL import Image
from zipfile import ZipFile


# In[ ]:


TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'
TRAIN_CSV = '../input/prostate-cancer-grade-assessment/train.csv'
TEST_CSV  = '../input/prostate-cancer-grade-assessment/test.csv'

sz = 224
N = 8
BATCH_SIZE = 64
EPOCHS_ = 5


# In[ ]:


def tile(img, mask):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result

def zipData():
    train_df = pd.read_csv(TRAIN_CSV)
    names = train_df['image_id'].values
    with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,     zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
        for name in tqdm(names):
            img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
            mask = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
            tiles = tile(img,mask)
            for t in tiles:
                img,mask,idx = t['img'],t['mask'],t['idx']
                #if read with PIL RGB turns into BGR
                img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f'{name}_{idx}.png', img)
                mask = cv2.imencode('.png',mask[:,:,0])[1]
                mask_out.writestr(f'{name}_{idx}.png', mask)


# In[ ]:


zipData()
gc.collect()

