#!/usr/bin/env python
# coding: utf-8

# Simple example of U-net for segmentation in Keras

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2

import keras
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import UpSampling2D, Conv2D, Activation, Input, Dropout, MaxPooling2D
from keras import Model
from keras import backend as K
from keras.layers.core import Lambda


# In[ ]:


tr = pd.read_csv('../input/train.csv')
print(len(tr))
tr.head()


# In[ ]:


df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
print(len(df_train))
df_train.head()


# In[ ]:


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


# In[ ]:


img_size = 256


# In[ ]:


def keras_generator(batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):            
            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
            img = cv2.imread( '../input/train_images/'+fn )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            
            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)
            
            img = cv2.resize(img, (img_size, img_size))
            mask = cv2.resize(mask, (img_size, img_size))
            
            x_batch += [img]
            y_batch += [mask]
                                    
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)


# In[ ]:


for x, y in keras_generator(4):
    break
    
print(x.shape, y.shape)


# In[ ]:


plt.imshow(x[3])


# In[ ]:


plt.imshow(np.squeeze(y[3]))


# In[ ]:


#Model

inputs = Input((256, 256, 3))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit model\nbatch_size = 16\nresults = model.fit_generator(keras_generator(batch_size), \n                              steps_per_epoch=100,\n                              epochs=20) ')


# In[ ]:


pred = model.predict(x)
plt.imshow(np.squeeze(pred[3]))


# In[ ]:


testfiles=os.listdir("../input/test_images/")
len(testfiles)


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_img = []\nfor fn in tqdm_notebook(testfiles):\n        img = cv2.imread( '../input/test_images/'+fn )\n        img = cv2.resize(img,(img_size,img_size))       \n        test_img.append(img)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict = model.predict(np.asarray(test_img))\nprint(len(predict))')


# In[ ]:


def mask2rle(img):
    tmp = np.rot90( np.flipud( img ), k=3 )
    rle = []
    lastColor = 0;
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1,1)   
    for i in range( len(tmp) ):
        if (lastColor==0) and tmp[i]>0:
            startpos = i
            lastColor = 1
        elif (lastColor==1)and(tmp[i]==0):
            endpos = i-1
            lastColor = 0
            rle.append( str(startpos)+' '+str(endpos-startpos+1) )
    return " ".join(rle)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred_rle = []\nfor img in predict:      \n    img = cv2.resize(img, (1600, 256))\n    tmp = np.copy(img)\n    tmp[tmp<np.mean(img)] = 0\n    tmp[tmp>0] = 1\n    pred_rle.append(mask2rle(tmp))')


# In[ ]:


img_t = cv2.imread( '../input/test_images/'+ testfiles[4])
plt.imshow(img_t)


# In[ ]:


mask_t = rle2mask(pred_rle[4], img.shape)
plt.imshow(mask_t)


# In[ ]:


sub = pd.read_csv( '../input/sample_submission.csv' )
sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for fn, rle in zip(testfiles, pred_rle):\n    sub['EncodedPixels'][sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn] = rle")


# In[ ]:


sub.head(8)


# In[ ]:


img_s = cv2.imread( '../input/test_images/'+ sub['ImageId_ClassId'][16].split('_')[0])
plt.imshow(img_s)


# In[ ]:


mask_s = rle2mask(sub['EncodedPixels'][16], (256, 1600))
plt.imshow(mask_s)


# In[ ]:


sub.to_csv('submission.csv', index=False)

