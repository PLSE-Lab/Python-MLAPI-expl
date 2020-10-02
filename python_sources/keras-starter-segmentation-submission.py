#!/usr/bin/env python
# coding: utf-8

# Simple example how to use segmentation in Keras

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2

import keras
from keras.layers import UpSampling2D, Conv2D, Activation
from keras import Model


# In[ ]:


tr = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
print(len(tr))
tr.head()


# In[ ]:


# Only ClassId=4

df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
df_train = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)
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
            img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+fn )
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


from keras.applications.vgg16 import VGG16
base_model = VGG16(weights=None, input_shape=(img_size,img_size,3), include_top=False)
base_model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


base_model.trainable = False


# In[ ]:


base_out = base_model.output
up = UpSampling2D(32, interpolation='bilinear')(base_out)
conv = Conv2D(1, (1, 1))(up)
conv = Activation('sigmoid')(conv)

model = Model(input=base_model.input, output=conv)

model.compile(keras.optimizers.Adam(lr=0.0001), 'binary_crossentropy')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'batch_size = 16\nmodel.fit_generator(keras_generator(batch_size),\n              steps_per_epoch=100,                    \n              epochs=5,                    \n              verbose=1,\n              shuffle=True)')


# In[ ]:


pred = model.predict(x)
plt.imshow(np.squeeze(pred[3]))


# In[ ]:


testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")
len(testfiles)


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_img = []\nfor fn in tqdm_notebook(testfiles):\n        img = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+fn )\n        img = cv2.resize(img,(img_size,img_size))       \n        test_img.append(img)")


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


img_t = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ testfiles[4])
plt.imshow(img_t)


# In[ ]:


mask_t = rle2mask(pred_rle[4], img.shape)
plt.imshow(mask_t)


# In[ ]:


sub = pd.read_csv( '../input/severstal-steel-defect-detection/sample_submission.csv', converters={'EncodedPixels': lambda e: ' '} )
sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for fn, rle in zip(testfiles, pred_rle):\n    sub['EncodedPixels'][(sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn) & \\\n                        (sub['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4'))] = rle")


# In[ ]:


img_s = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ sub['ImageId_ClassId'][47].split('_')[0])
plt.imshow(img_s)


# In[ ]:


mask_s = rle2mask(sub['EncodedPixels'][47], (256, 1600))
plt.imshow(mask_s)


# In[ ]:


sub.to_csv('submission.csv', index=False)

