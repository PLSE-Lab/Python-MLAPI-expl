#!/usr/bin/env python
# coding: utf-8

# As already noted in this competition  we are going to give a struggle *in  the open sea*. About `75000` images contain no ship at all and succesfully sampling from those may benefit any model. However, stratification using no-ship images would be difficult. In the case of images with ships we can either stratify using
# * the number of ships in the image
# * the area of ship pixels in the mask
# 
# In this notebook, I attempt to group no ship images, based on *color information* using bits and pieces from the following two kernels.
# * [https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering](http://)  
# * [https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization](http://)
# 
# For  every image the dominant color is extracted by applying kmeans on pixel intesities. Then all `75000` are partitioned using this dominant HSV color information into `NUM_CLASSES` classes.  As a result of this analysis we could group no-ship images in a arbitrary number of classes, and sample from those  at random. 

# In[ ]:


import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook, tnrange

from scipy import signal

import cv2
from PIL import Image
import pdb
from tqdm import tqdm
import seaborn as sns
import os 
from glob import glob

import warnings
warnings.filterwarnings("ignore")


# <h2> Setting paths

# In[ ]:


INPUT_PATH = '../input'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train/masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
df = pd.read_csv(DATA_PATH+'/train_ship_segmentations.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')


# ## Find images containing no ships 

# In[ ]:


images_with_no_ship = df.index[df.EncodedPixels.isnull()==True]
print ('Found ' + str(len(images_with_no_ship)) + ' no-ship images') 


# ## Some utility definitions and functions

# In[ ]:


# 
# Number of distinct classes 
NUM_CLASSES = 50
#
# In order to reduce computation time, downsample train images. 
# Sure we loose some pixel information this way.....
IMG_SIZE = 32


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))


def get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist


# In[ ]:


details = []
for imfile in tqdm(images_with_no_ship):
    image_hsv = get_image_data_opencv(imfile, "Train")
    height, width, l = image_hsv.shape
    dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image_hsv, top_colors=1)
    dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
    info = (imfile, width, height, dominant_colors_hsv.squeeze())
    details.append(info)


#  ## Apply kmeans on HSV dominant color

# In[ ]:


cols  = ['image_id', 'image_width', 'image_height', 'hsv_dominant']
trainPD = pd.DataFrame(details, columns=cols)
X = (pd.DataFrame(trainPD['hsv_dominant'].values.tolist())).as_matrix()
kmeans = KMeans(n_clusters=50).fit(X)
clusters = kmeans.predict(X)
trainPD['hsv_cluster'] = clusters
trainPD.head()


# In[ ]:


## View partitioning counts


# In[ ]:


hist = trainPD.groupby('hsv_cluster')['image_id'].count()
hist


# In[ ]:


plt.figure(figsize=(12, 6))
plt.title('#images per partition')
plt.bar(np.arange(50), hist.values)
plt.show()


# In[ ]:


def plot_images(images, images_rows, images_cols):
    f, axarr = plt.subplots(images_rows,images_cols,figsize=(16,images_rows*2))
    for row in range(images_rows):
        for col in range(images_cols):
            image_id = images[row*images_cols + col]
            image = cv2.imread(get_filename(image_id, 'Train'))
            height, width, l = image.shape
            ax = axarr[row,col]
            ax.axis('off')
            ax.set_title("%dx%d"%(width, height))
            ax.imshow(image)


# ## Some arbitrary plots for clusters, 0, 1 and 30....

# In[ ]:


plot_images(trainPD[trainPD['hsv_cluster'] == 0]['image_id'].values, 4, 4)


# In[ ]:


plot_images(trainPD[trainPD['hsv_cluster'] == 1]['image_id'].values, 4, 4)


# In[ ]:


plot_images(trainPD[trainPD['hsv_cluster'] == 30]['image_id'].values, 4, 4)


# ## Save cluster information 

# In[ ]:


trainPD.to_csv('noship_clusters.csv', index = False)

