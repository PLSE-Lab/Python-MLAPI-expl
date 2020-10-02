#!/usr/bin/env python
# coding: utf-8

# Reference
# 
# - Awesome Kernel, thanks @martinpiotte : https://www.kaggle.com/martinpiotte/bounding-box-model
# - cropping.model: https://www.kaggle.com/martinpiotte/bounding-box-model/output
# - Data: https://www.kaggle.com/c/humpback-whale-identification/data

# In[ ]:


import os

import PIL
from PIL import Image
from PIL.ImageDraw import Draw
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import load_model
from keras.preprocessing import image


# In[ ]:


ls ../input


# In[ ]:


MODEL_BASE = '../input/bbox-model-whale-recognition'
DATA = '../input/humpback-whale-identification'
TRAIN_DATA = os.path.join(DATA, 'train')
TEST_DATA = os.path.join(DATA, 'test')


# In[ ]:


model = load_model(os.path.join(MODEL_BASE, 'cropping.model'))


# In[ ]:


# # input: (128, 128, 1)
# model.summary()


# In[ ]:


train_paths = [os.path.join(TRAIN_DATA, img) for img in os.listdir(TRAIN_DATA)]
test_paths = [os.path.join(TEST_DATA, img) for img in os.listdir(TEST_DATA)]


# In[ ]:


train_paths[0]


# In[ ]:


img = image.load_img(train_paths[10])


# In[ ]:


img


# In[ ]:


img_arr = image.img_to_array(img)


# In[ ]:


img_arr.shape


# In[ ]:


rimg = img.resize((128, 128), PIL.Image.ANTIALIAS)


# In[ ]:


rimg


# In[ ]:


rimg_arr = image.img_to_array(rimg)


# In[ ]:


rimg_ = rimg.convert('L')


# In[ ]:


rimg_arr_ = image.img_to_array(rimg_)


# In[ ]:


rimg_arr_.shape


# In[ ]:


bbox = model.predict(np.expand_dims(rimg_arr_, axis=0))


# In[ ]:


bbox


# In[ ]:


draw = Draw(rimg_)


# In[ ]:


draw.rectangle(bbox, outline='red')


# In[ ]:


rimg_


# In[ ]:


rimg


# In[ ]:


img_crop = rimg_.crop(tuple(bbox[0]))


# In[ ]:


img_crop


# In[ ]:


def make_bbox_image(img_path):
    """
    :param img: path to image
    """
    main_img = image.load_img(img_path)
    r_img = main_img.resize((128, 128), PIL.Image.ANTIALIAS)
    # convert to 1d image
    rb_img = r_img.convert('L')
    rb_img_arr = image.img_to_array(rb_img)
    bbox = model.predict(np.expand_dims(rb_img_arr, axis=0))
    
    # draw rectangle
    # draw = Draw(rimg)
    # draw.rectangle(bbox, outline='red')
    
    img_crop = r_img.crop(tuple(bbox[0]))
    img_arr = image.img_to_array(img_crop)
    return img_crop


# In[ ]:


train_paths[10]


# In[ ]:


img = make_bbox_image(train_paths[10])


# In[ ]:


plt.imshow(img)


# In[ ]:




