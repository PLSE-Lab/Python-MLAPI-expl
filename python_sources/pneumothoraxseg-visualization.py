#!/usr/bin/env python
# coding: utf-8

# - Thanks @Jesper for upoading the Dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 100)

import gc

import glob

import pydicom

from matplotlib import cm
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, StratifiedShuffleSplit

import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import rle2mask #provided by competition

import os
print(os.listdir("../input"))

import random
seed = 1234
random.seed(seed)

from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

from time import time, strftime, gmtime

start = time()
print(start)

import datetime
print(str(datetime.datetime.now()))


# In[ ]:


print(os.listdir('../input/siim-acr-pneumothorax-segmentation/'), 
      os.listdir('../input/siim-acr-pneumothorax-segmentation-data/'))


# __Digital Imaging and Communications in Medicine (DICOM) standard is the de-facto solution to storing and exchanging medical image data__

# In[ ]:


#Sample Images
print(os.listdir('../input/siim-acr-pneumothorax-segmentation/sample images'))
sample_imgs = glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')
print(len(sample_imgs))


# In[ ]:


#Sample Masks
sample_masks = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv', 
                          header = None, index_col = 0)

print(sample_masks.shape)
display(sample_masks.head(10))


# In[ ]:


plt.figure(figsize = (10, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    file = sample_imgs[i]
    img = pydicom.dcmread(file)
    plt.title('Sex: {}, Age: {}, {}'.format(img.PatientSex, img.PatientAge, img.BodyPartExamined))
    plt.imshow(img.pixel_array, cmap = plt.cm.bone)
    #Getting Mask for the image
    if sample_masks.loc[file.split('/')[-1][:-4], 1] != '-1':
        mask = rle2mask(sample_masks.loc[file.split('/')[-1][:-4], 1], 1024, 1024).T
        plt.imshow(mask, alpha = 0.3, cmap = 'Blues')
    else:
        plt.text(400, 1200, 'No Anomaly', fontsize = 12)


# In[ ]:


print('Other information in a DICOM file:')
print(img.fix_meta_info)

del sample_imgs, sample_masks
gc.collect()


# __Readying the train, train masks and test sets__

# In[ ]:


print(os.listdir('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/'))
train_path = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm'
test_path = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm'
train_masks = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv'


# In[ ]:


train_files = sorted(glob.glob(train_path))
test_files = sorted(glob.glob(test_path))

print('No. of train files: {}, test files: {}'.format(len(train_files), len(test_files)))


# In[ ]:


df_rles = pd.read_csv(train_masks)
print(df_rles.shape)
display(df_rles.head())


# In[ ]:


df_rles.columns


# __Loading RLE Pixels into a Dictionary__

# In[ ]:


from collections import defaultdict

rle_dict = defaultdict(list)
for img_id, rle in zip(df_rles['ImageId'], df_rles[' EncodedPixels']):
    rle_dict[img_id].append(rle)

annotated = {k: v for k, v in rle_dict.items() if v[0] != ' -1'}
multi_annot = {k: v for k, v in rle_dict.items() if len(v) > 1}

print('No. of images with masks is {} out of {} total images'.format(len(annotated), len(rle_dict)))
print('No. of images with more than one mask: {}'.format(len(multi_annot)))
#There is a image with 10 masks 

print('No. of missing masks: {}'.format(len(train_files) - len(rle_dict)))
#This missing files can be ignored as they are not chest X-rays (per discussion 
#https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/96993#latest-560352)

del df_rles
gc.collect()


# __Visualizing Images with their masks (No mask and mask)__

# In[ ]:


files = np.random.choice(train_files, 6)

plt.figure(figsize = (12, 10))

for i, file in enumerate(files):
    plt.subplot(2, len(files) // 2, i + 1)
    img = pydicom.dcmread(file)
    img_id = file.split('/')[-1][:-4]
    plt.title('Sex: {}, Age: {}, {}'.format(img.PatientSex, img.PatientAge, img.BodyPartExamined))
    plt.imshow(img.pixel_array, cmap = plt.cm.bone)
    #print(len(rle_dict[img_id]))
    for rle in rle_dict[img_id]:
        if rle != ' -1':
            mask = rle2mask(rle, 1024, 1024).T
            plt.imshow(mask, alpha = 0.3, cmap = 'Blues')
        else:
            #print('No Anamoly')
            plt.text(400, 1200, 'No Anomaly', fontsize = 12)


# __Visualizing Images with their Masks (multiple)__

# In[ ]:


get_ipython().run_cell_magic('time', '', "file_path = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/'\n\nfiles = np.random.choice(list(multi_annot.keys()), 6)\n\nplt.figure(figsize = (12, 10))\n\nfor i, file in enumerate(files):\n    plt.subplot(2, len(files) // 2, i + 1)\n    img_path = glob.glob(file_path + file + '.dcm')\n    img = pydicom.dcmread(img_path[0])\n    plt.title('Sex: {}, Age: {}, {}'.format(img.PatientSex, img.PatientAge, img.BodyPartExamined))\n    plt.imshow(img.pixel_array, cmap = plt.cm.bone)\n    for rle in multi_annot[file]:\n        mask = rle2mask(rle, 1024, 1024).T\n        plt.imshow(mask, alpha = 0.3, cmap = 'Blues')\n\ndel file_path, files\ngc.collect()")


# In[ ]:


finish = time()
print(strftime("%H:%M:%S", gmtime(finish - start)))

