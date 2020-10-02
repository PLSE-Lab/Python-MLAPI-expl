#!/usr/bin/env python
# coding: utf-8

# # Image Orientation
# 
# Thank you to [@xhlulu](https://www.kaggle.com/xhlulu/rsna-generate-metadata-csvs) for extracting all the image metadata
# 
# This notebook ranks and identifies anomolous image orientations, and visualizes some of these orientations.  It does not provide insight into what the orientation vector means ... I am hoping to figure that out, but am wondering if anyone in the community knows how to decode the 6D orientation vector
# 
# I think there are 2 possible approaches to deal with this variable:
#  - randomize the orientation with some augmnetaiton at training time
#  - standardize the orientation during a preprocessing stage
# 
# My goal is to explore option 2

# In[ ]:


import os

import numpy as np 
import pandas as pd 
import skimage
import cv2 as cv
import pydicom
import json

from scipy.spatial.distance import euclidean


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/rsna-generate-metadata-csvs/train_metadata.csv')
train['image_dir'] = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

test = pd.read_csv('../input/rsna-generate-metadata-csvs/test_metadata.csv')
test['image_dir'] = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

metadata = pd.concat([train, test])
train, test = None, None
metadata['image_orientation'] = metadata['ImageOrientationPatient'].apply(lambda x: np.array(eval(x)).astype(np.float32))


# # Types of Image orientation
# 
# We will find all the unique image orientations and sort them from maximum distance to the mean.  Then take top-k = 50 to find our anomoulous orientations

# In[ ]:


orientations = metadata['ImageOrientationPatient'].unique()
orientations = np.array([eval(o) for o in orientations]).astype(np.float32)

mean_orientation = orientations.mean(axis=0)
print('Mean Orientation: {}'.format(mean_orientation))
orientations = orientations[np.argsort([euclidean(o, mean_orientation) for o in orientations])[::-1]]
orientations = pd.DataFrame(orientations)


# In[ ]:


print("Common of Orientations:")
orientations.tail(5)


# In[ ]:


print("Orientations of Intrest:")
orientations.head(30)


# In[ ]:


# finds other orientations with similar euclidean distance
def match_orientation(target, tol=1e-6):
    metadata['dist'] = metadata['image_orientation'].apply(lambda x: euclidean(x, target))
    return metadata[metadata['dist'] < tol]

def show(dcm, ax, rot=0):
    dcm = pydicom.dcmread(dcm.image_dir + dcm.SOPInstanceUID + '.dcm')
    img = dcm.pixel_array  * dcm.RescaleSlope + dcm.RescaleIntercept
    img = np.clip(img, 0, 100)
    img = skimage.transform.rotate(img, rot)
    return ax.imshow(img, cmap='bone')


# # Visualize Orientations
# We will sample every 10th orientation sorted form most to least anomolous

# In[ ]:


o =orientations.values[0]
prtinnp.degrees(np.arctan2(o[0], ))


# In[ ]:


N_SAMPLES = 4

for o in orientations.values[::10]:
    x = match_orientation(o)
    f, ax = plt.subplots(1, N_SAMPLES, figsize=(20, 5))
    for i in range(N_SAMPLES):
        show(x.iloc[i*3], ax[i], rot=0)

    f.suptitle('ORIENTATION: {}'.format(o))
    plt.show()

