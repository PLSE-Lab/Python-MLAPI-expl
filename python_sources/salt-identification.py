#!/usr/bin/env python
# coding: utf-8

# **Background Data**
# * Where there is salt, there is oil.
# * But where is the salt?
# * Classifying seismic imaging currently requires human (salt/not salt)
# * Can we use an algorithm to do this instead? (yes)
# * Seismic data is like an ultra-sound of the subsurface
# * It uses wavelengths around 1m to 100m
# * The Society of Exploration Geophysicists has 10K publications using the keyword 'salt'
# * We can think of Earth as layered.
# * Sand gets deposited on top of existing sand. And in comes the salt.
# * There is an entire research branch dedicated to salt tectonics, that is the movement of salt in the subsurface.
# **Our steps**
# * Install Dependencies
# * View the labeled seismic imaging data
# * Plot the depth distribution in the training data
# * Plot the proportion of salt vs depth in the training data
# * Build a U-Net algorithm to learn the mapping between seismic images and the salt filter mask

# In[ ]:


import os
import sys
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tnrange, tqdm
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.feature import canny
from skimage.filters import sobel,threshold_otsu, threshold_niblack,threshold_sauvola
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from scipy import signal
import os
import numpy as np
import imageio
import pandas as pd
import torch
from torch.utils import data
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from PIL import Image
import pdb
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings("ignore")
print("Packages Loaded Successfully")


# In[ ]:


INPUT_PATH = '../input'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train/images")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train/masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
df = pd.read_csv(DATA_PATH+'/depths.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]


# In[ ]:


def get_file_name(image_id, image_type):
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

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_file_name(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[ ]:


nImg = 32  #no. of images that you want to display
np.random.seed(42)
_train_ids = list(train_ids)
np.random.shuffle(_train_ids)
_train_ids = _train_ids[:nImg]
tile_size = (256, 256)
n = 8
alpha = 0.25

m = int(np.ceil(len(_train_ids) * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(_train_ids):
            break
        image_id = _train_ids[counter]; counter+=1
        img = get_image_data(image_id, 'Train')
        
        mask = get_image_data(image_id, "Train_mask")
        img_masked =  cv2.addWeighted(img, alpha, mask, 1 - alpha,0)
#         img_masked = cv2.bitwise_and(img, img, mask=mask)

        img = cv2.resize(img, dsize=tile_size)
        img_masked = cv2.resize(img_masked, dsize=tile_size)
        
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
        complete_image[ys:ye, xs:xe, :] = img[:,:,:]
        
        img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
        complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
        
    if counter == len(_train_ids):
        break    
        
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image[ys:ye,:,:])
    plt.title("Training dataset")
    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image_masked[ys:ye,:,:])
    plt.title("Training dataset: Lighter Color depicts salt")


# In[ ]:




