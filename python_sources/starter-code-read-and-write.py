#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from pathlib import Path
import os
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 
np.random.RandomState(seed=1)

print(os.listdir("../input"))


# In[2]:


input_dir  = Path('../input/data/')
train = input_dir / 'distorted'
train_cleaned = input_dir / 'clean'
test = input_dir / 'test_distorted'
train_images = sorted(os.listdir(train))
train_labels = sorted(os.listdir(train_cleaned))
test_images = sorted(os.listdir(test))
X = []
Y = []
X_test=[]
for img in train_images:
    img = load_img(train / img,target_size=(48,48,3))
    img = img_to_array(img).astype('float32')/255.
    X.append(img)

for img in train_labels:
    img = load_img(train_cleaned / img,target_size=(48,48,3))
    img = img_to_array(img).astype('float32')/255.
    Y.append(img)

for img in test_images:
    img = load_img(test / img,target_size=(48,48,3))
    img = img_to_array(img).astype('float32')/255.
    X_test.append(img)

X_test = np.array(X_test)
X = np.array(X)
Y = np.array(Y)
X.shape,Y.shape


# In[3]:


# your model...
resoult=np.zeros((400, 3, 48, 48))


# In[4]:


def save_result(images: np.ndarray, out_path: str):
    
    assert images.shape == (400, 3, 48, 48)
    
    flat_img = images.reshape(400, -1)
    n_rows = np.prod(images.shape)
    
    y_with_id = np.concatenate([np.arange(n_rows).reshape(-1, 1), flat_img.reshape(n_rows, 1)], axis=1)
    np.savetxt(out_path, y_with_id, delimiter=",", fmt=['%d', '%.4f'], header="id,expetced", comments='')


# In[5]:


save_result(resoult,'sub.csv')

