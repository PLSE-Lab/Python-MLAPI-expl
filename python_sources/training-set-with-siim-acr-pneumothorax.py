#!/usr/bin/env python
# coding: utf-8

# # Check the training data

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import sys
import glob2
import shutil

from PIL import Image
from matplotlib import pyplot as plt

sys.path.append('../input/siim-acr-pneumothorax-segmentation/')
from mask_functions import rle2mask


print(os.listdir("../input"))


# In[ ]:


data_path = '../input/siim-acr-pneumothorax-segmentation-data'
train_path = os.path.join(data_path, 'pneumothorax/dicom-images-train')
test_path = os.path.join(data_path, 'pneumothorax/dicom-images-test')

train_rle_path = os.path.join(data_path, 'pneumothorax/train-rle.csv')
annotation = pd.read_csv(train_rle_path)
annotation.head(10)


# In[ ]:


imageId = annotation['ImageId']
annotation = annotation.rename(columns={' EncodedPixels': 'EncodedPixels'})
encodedPixels = annotation['EncodedPixels']


# # Merge images and train/test split

# In[ ]:


dataset_path = './dataset'
train_new_path = os.path.join(dataset_path, 'dicom-images-train')
test_new_path = os.path.join(dataset_path, 'dicom-images-test')
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
if not os.path.isdir(train_new_path):
    os.mkdir(train_new_path)
if not os.path.isdir(test_new_path):
    os.mkdir(test_new_path)

for filename in glob2.glob('{}/**/*.dcm'.format(train_path)):
    fname = str(filename).split('/')[-1]
#     print(fname)
    shutil.copy(str(filename), os.path.join(train_new_path, fname))

for filename in glob2.glob('{}/**/*.dcm'.format(test_path)):
    fname = str(filename).split('/')[-1]
#     print(fname)
    shutil.copy(str(filename), os.path.join(test_new_path, fname))
    
print("dicom-images-train ", len(os.listdir(train_new_path)))
print("dicom-images-test  ", len(os.listdir(test_new_path)))


# # Check the duplicate data

# In[ ]:


duplicate_data = annotation[imageId.duplicated()]
unique_data = duplicate_data['ImageId'].unique()
except_duplicate_data = imageId.unique()

print("num of annotation data   :", len(annotation))
print("num of duplicate data    :", len(duplicate_data))
print("num of unique data       :", len(unique_data))
print("num of except duplicate  :", len(except_duplicate_data))


# # Merge duplicate data

# In[ ]:


start_index = 61
num_output = 10
fig, ax = plt.subplots(2, num_output, figsize=(30,10))

for i in range(num_output):
    index = i
    index += start_index
    ds = pydicom.read_file(os.path.join(train_new_path, imageId[index] + '.dcm'))
    img = ds.pixel_array
    img_men = Image.fromarray(img)
    print('Index {}, '.format(index) + imageId[index])
    
    if encodedPixels[index].strip() != '-1':
        rleToMask = rle2mask(
            rle=encodedPixels[index],
            width=img.shape[0],
            height=img.shape[1]
        )
        ax[0][i].imshow(img_men, cmap=plt.cm.bone)
        ax[0][i].imshow(rleToMask.astype(np.bool), alpha=0.4, cmap="Blues")
        ax[0][i].set_title('Index: {}'.format(index))
        
    elif encodedPixels[index].strip() == '-1':
        mask_0 = np.zeros((img.shape[:2]))
        ax[0][i].imshow(mask_0.astype(np.bool))
        ax[0][i].set_title('Index: {}'.format(index)) 
    
check_data = None
temp_mask = np.zeros((1024, 1024), dtype=np.bool)
for i in range(num_output):
    index = i
    index += start_index
    ds = pydicom.read_file(os.path.join(train_new_path, imageId[index] + '.dcm'))
    img = ds.pixel_array
    img_men = Image.fromarray(img)
    
    if encodedPixels[index].strip() != '-1':
        rleToMask = rle2mask(
            rle=encodedPixels[index],
            width=img.shape[0],
            height=img.shape[1]
        )
        if check_data == imageId[index]:
            temp_mask += rleToMask
            ax[1][i].imshow(temp_mask.astype(np.bool))
        else:
            temp_mask = rleToMask
            ax[1][i].imshow(rleToMask.astype(np.bool))
            
    elif encodedPixels[index].strip() == '-1':
        mask_0 = np.zeros((img.shape[:2]))
        temp_mask = mask_0
        ax[1][i].imshow(mask_0.astype(np.bool))
        
    check_data = imageId[index]
    
plt.show()


# # Create mask images

# In[ ]:


train_mask_path = os.path.join(dataset_path, 'mask-images-train')
test_mask_path = os.path.join(dataset_path, 'mask-images-test')
if not os.path.isdir(train_mask_path):
    os.mkdir(train_mask_path)
if not os.path.isdir(test_mask_path):
    os.mkdir(test_mask_path)

check_data = None
temp_mask = np.zeros((1024, 1024))
for i in range(len(imageId)):
    if encodedPixels[i].strip() != '-1':
        ds = pydicom.read_file(os.path.join(train_new_path, imageId[i] + '.dcm'))
        img = ds.pixel_array
        img_mem = Image.fromarray(img)
        
        rleToMask = rle2mask(
            rle=encodedPixels[i],
            width=img.shape[0],
            height=img.shape[1]
        )
        if check_data == imageId[i]:
            temp_mask += rleToMask
            cv2.imwrite(train_mask_path + '/{}_mask.png'.format(imageId[i]), temp_mask.astype('int32'))
        else:
            temp_mask = rleToMask
            cv2.imwrite(train_mask_path + '/{}_mask.png'.format(imageId[i]), rleToMask.astype('int32'))
        
    elif encodedPixels[i].strip() == '-1':
        ds = pydicom.read_file(os.path.join(train_new_path, imageId[i] + '.dcm'))
        img = ds.pixel_array
        img_mem = Image.fromarray(img)
        
        mask_0 = np.zeros((img.shape[:2]))
        temp_mask = mask_0
        cv2.imwrite(train_mask_path + '/{}_mask.png'.format(imageId[i]), mask_0.astype('int32'))
        
    check_data = imageId[i]
        
print("mask-images-train", len(os.listdir(train_mask_path)))


# # Training set visualization

# In[ ]:


print(os.listdir(dataset_path))
print('dicom-images-train ', len(os.listdir(train_new_path)))
print('dicom-images-test  ', len(os.listdir(test_new_path)))
print('mask-images-train  ', len(os.listdir(train_mask_path)))


# In[ ]:


start_index = 0
num_output = 10
fig, ax = plt.subplots(3, num_output, figsize=(30,10))

for i in range(num_output):
    index = i
    index += start_index
    
    ds = pydicom.read_file(os.path.join(train_new_path, imageId[index] + '.dcm'))
    img = ds.pixel_array
    ori_img = Image.fromarray(img)
    ax[0][i].imshow(ori_img)
    ax[0][i].set_title('Index: {}'.format(index))
    
for i in range(num_output):
    index = i
    index += start_index
    
    ds = pydicom.read_file(os.path.join(train_new_path, imageId[index] + '.dcm'))
    img = ds.pixel_array
    img_men = Image.fromarray(img)
    if encodedPixels[index].strip() != '-1':
        rleToMask = rle2mask(
            rle=encodedPixels[index],
            width=img.shape[0],
            height=img.shape[1]
        )
        ax[1][i].imshow(img_men, cmap=plt.cm.bone)
        ax[1][i].imshow(rleToMask, alpha=0.4, cmap="Blues")
        ax[1][i].set_title('Index: {}'.format(index)) 
        
    elif encodedPixels[index].strip() == '-1':
        mask_0 = np.zeros((img.shape[:2]))
        mask_0 = mask_0
        ax[1][i].imshow(mask_0.astype(np.bool))
        ax[1][i].set_title('Index: {}'.format(index)) 
    
for i in range(num_output):
    index = i
    index += start_index
    
    mask_img = cv2.imread(os.path.join(train_mask_path, imageId[index] + '_mask.png'))
    ax[2][i].imshow(mask_img)
    ax[2][i].set_title('Index: {}'.format(index))
    
print(annotation[start_index : start_index + num_output])
    
plt.show()


# In[ ]:


get_ipython().system(' rm -rf dataset')

