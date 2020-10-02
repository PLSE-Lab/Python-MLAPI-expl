#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
from plotly import tools
import os
import seaborn as sns
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import cv2
from PIL import Image
from plotly.offline import iplot
import cufflinks
#from tpu_helper import *
import cv2 as cv
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


# # SIIM-ISIC Melanonma Classification
# 
# This competition is to identify melanoma (a type of skin cancer) using lesion images.
# 
# The given data set includes 4 folders jpeg, tfrecords, test, train, and three .csv files: test.csv, train.csv and sample_submission.csv.
# 
# Here, the train.csv and test.csv are metadata which contains information about images provided in train and test folder where images stored in DICOM format.
# 
# Images are also provided in JPEG and TFRecord format (in the jpeg and tfrecords directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

# In[ ]:


train_images_dir = '../input/siim-isic-melanoma-classification/train/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/siim-isic-melanoma-classification/test/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]


# In[ ]:


fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images_dir + train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


# # Explanatory Data Analysis

# **File train.csv have 8 columns, 33126 rows**
# * image_name - unique identifier, points to filename of related DICOM image
# * patient_id - unique patient identifier
# * sex - the sex of the patient (when unknown, will be blank)
# * age_approx - approximate patient age at time of imaging
# * anatom_site_general_challenge - location of imaged site
# * diagnosis - detailed diagnosis information (train only): unknown, nevus, melanoma,...
# * benign_malignant - indicator of malignancy of imaged lesion: benign = harmless, malignant = harmful.
# * target - binarized version of the target variable (1:melanoma; 0: non)

# In[ ]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
train.head()


# * **File train.csv have only 5 columns, 10982 rows** :image_name, patient_id, sex, age_approx, anatom_site_general_challenge

# In[ ]:


test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test.head()


# **Submission**: assign each image in the test set (10982 rows) to a target variable: from 0 to 1 (to indicate the percentage of having melanoma or non)

# In[ ]:


sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
sample_submission.head()


# # Data Preprocessing
# 
# As we can see, the size of these lension images are not the same. 
# 
# Therefore, we do need to resize the images and also do the augmentation. cv2 module can be used to complete these two tasks.
# 
# This part is from https://www.kaggle.com/nxrprime/siim-eda-augmentations-model-seresnet-unet

# ****Augmentation and Visualization****

# We will first try to visualize in grayscale (only gray colors) so that it is possible for us to clearly visualize the varied differences in color, region, and shape.

# In[ ]:


import cv2
def view_images_aug1(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug1(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# The second method we will use is Ben Graham's method from APTOS.

# In[ ]:


def view_images_aug2(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug2(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# We can further observe the clear color distinctions by using Neuron Engineer's method (an improved version of Ben Graham's).

# In[ ]:


def view_images_aug3(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug3(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Here we can finally visualize the clear distinctions in our data. The clear regions, the clear color differences, the clear everything! 
# 
# Circular crop may not be feasible for images where the tumor is on the edge of the image. It does not seem so feasible, so I would recommend you try to be smarter in your methods for preprocessing. Remember, you can build upon Ben Graham's work as a starting point, then try Neuron Engineer's or circle crop or even build you own method.

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def view_images_aug4(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug4(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Now we are using auto-cropping as a method of preprocessing, which is a more "refined" circle crop if you will. Think of circle crop as C, and think of auto-cropping as C++. Auto-cropping indeed is powerful, but the risk is that you will lose valuable data in the image.

# In[ ]:


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= crop_image_from_gray(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Another thing you can do is background subtraction.

# In[ ]:


fgbg = cv.createBackgroundSubtractorMOG2()
    
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= fgbg.apply(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Also, you can use the albumentations library to create a lot of simulated images for your model. Remember, your model MUST BE ABLE TO GENERALIZE!

# In[ ]:


import albumentations as A
image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
albumentation_list = [A.RandomSunFlare(p=1), A.RandomFog(p=1), A.RandomBrightness(p=1),
                      A.RandomCrop(p=1,height = 512, width = 512), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","RandomFog","RandomBrightness",
               "RandomCrop","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# We have much more augmentations we can try like:

# In[ ]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
albumentation_list = [A.RandomSunFlare(p=1), A.GaussNoise(p=1), A.CLAHE(p=1),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# **Resized Images**
# 
# The images should be resized to speed up the testing progress.
# The original kernel is here https://www.kaggle.com/tunguz/image-resizing-32x32-train
# 

# In[ ]:


import gc
import json
import math
import cv2
import PIL
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import scipy
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing import image


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


#These code will run for very long time. Then, we can use the alread-done files created in 
#here https://www.kaggle.com/tunguz/siimisic-melanoma-resized-images
"""
image_Size = 32
def preprocess_image(image_path, desired_size=image_Size):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

# get the number of training images from the target\id dataset
N = train.shape[0]
# create an empty matrix for storing the images
x_train = np.empty((N, image_Size, image_Size, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, image_id in enumerate(tqdm(train['image_name'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg'
    )
"""


# In[ ]:


"""
Ntest = test.shape[0]
# create an empty matrix for storing the images
x_test = np.empty((Ntest, image_Size, image_Size, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, image_id in enumerate(tqdm(test['image_name'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/siim-isic-melanoma-classification/jpeg/test/{image_id}.jpg'
    )
"""


# In[ ]:


"""
np.save('x_train_32', x_train)
np.save('x_test_32', x_test)
"""


# # The Baseline Model
# 
# The most simple baseline here is to assign every case 0.5. That means for every cases we have 50% of melanoma. The score for this kind of submission is 0.5 (Ofcourse!).
# 
# Now, we can try to implement our first baseline model with simple CNN.

# 1. Before using machine learning tools, let try with the basic statistic based off of mean and count of the target variable to get final prediction.
# 
# This part is from this notebook https://www.kaggle.com/titericz/simple-baseline

# In[ ]:


#Filling NA
train['sex'] = train['sex'].fillna('na')
train['age_approx'] = train['age_approx'].fillna(0)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

test['sex'] = test['sex'].fillna('na')
test['age_approx'] = test['age_approx'].fillna(0)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')


# In[ ]:


features = ['sex','age_approx','anatom_site_general_challenge']
Mean_train = train.target.mean() #mean of target columns in train set (0.01762965646320111)
#Grouping the train set using features, then adding the mean and count of each group
groupped_train = train.groupby(features)['target'].agg(['mean','count']).reset_index()
groupped_train.head()


# In[ ]:


#Writing the prediction with mean of each group
#A paremeter L is added for the bias of the whole train set.
L=15
groupped_train['prediction'] = ((groupped_train['mean']*groupped_train['count'])+(Mean_train*L))/(groupped_train['count']+L)
del groupped_train['mean'], groupped_train['count']

test = test.merge( groupped_train, on=features, how='left' )
test['prediction'] = test['prediction'].fillna(Mean_train)
test.head()


# In[ ]:


sample_submission.target = test.prediction.values
sample_submission.head(5)
sample_submission.to_csv( 'submission_GroupedMean.csv', index=False )

