#!/usr/bin/env python
# coding: utf-8

# ## Region of Interest (ROI) detection using Machine Learning ##
# 

# By reading chattob's notebook "Cervix segmentation (GMM)" I got inspired to find the Region of Interest (ROI) using Supervised Learning.

# I'm taking the next cell directly from here: https://www.kaggle.com/chattob/cervix-segmentation-gmm

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os

TRAIN_DATA = "../input/intel-mobileodt-cervical-cancer-screening/train"

types = ['Type_1']#,'Type_2','Type_3']
type_ids = []

for type in enumerate(types):
    type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
    type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
    type_ids.append(type_i_ids[:5])

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Let's reshape the images to a fixed size.

# In[ ]:


reshaped_color = []
for type in enumerate(types):
    image_ids = type_ids[type[0]]
    for image_id in image_ids:
        img = get_image_data(image_id, type[1])
        ar = img.shape[0] * 1.0 / 480
        new_img = cv2.resize(img, (int(img.shape[1] / ar), 480))
        new_img = new_img[:, :360, :]
        x = np.zeros((480, 360, 3), dtype=np.uint8)
        x[:new_img.shape[0], :new_img.shape[1], :] = new_img
        reshaped_color.append(x)
        plt.imshow(x)
        plt.show()


# Then manually, paint anything that it's not within the ROI, for example:

# In[ ]:


names = [0, 10, 1013, 102, 104]
for name in names:
    img = cv2.imread('../input/region-of-interest-roi-detection-using-ml/{}.jpg'.format(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


# Next steps:
# 
# Build the training dataset where features are the x and y position plus the R, G, B intensity of each pixel using a cv.filter2d
# 
# Train it using XGBoost
# 
# Use it to predict the ROI for Test files
