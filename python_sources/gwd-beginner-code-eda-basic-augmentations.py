#!/usr/bin/env python
# coding: utf-8

# This is the first competition that I have begun my Kaggle journey with. Also, the first time that I am working on an object-detection problem statement. 
# I present here a very basic EDA for the given wheat head dataset hoping that other beginners like me would find it easy to understand the data and get started with this intriguing competition.
# In addition to the EDA, I have also added a few augmentations that I tried on the images in the second half of the kernel.

# Import dependencies.

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import re
import albumentations as A
get_ipython().run_line_magic('matplotlib', 'inline')


# Save necessary paths to the data.

# In[ ]:


INPUT_DIR = '../input/global-wheat-detection'
TRAIN_DIR = f'{INPUT_DIR}/train'


# Understand the basic image characteristics.

# In[ ]:


sample = cv2.imread(TRAIN_DIR+'/01189a3c3.jpg', cv2.IMREAD_UNCHANGED)

dimensions = sample.shape
height = sample.shape[0]
width = sample.shape[1]
n_of_channels = sample.shape[2]

print('Image characteristics:\n')
print('Dimensions: {}\nHeight: {}, Width:{}, Number of channels: {}'.format(dimensions, height, width, n_of_channels))


# Load the train csv file.

# In[ ]:


train_df = pd.read_csv(INPUT_DIR+'/train.csv')
train_df.head()


# Check if all the image dimensions are same.

# In[ ]:


print(train_df['height'].unique(), train_df['width'].unique())


# Understanding the csv file.

# In[ ]:


# No of unique images in the csv file

print('Unique images: ',len(train_df['image_id'].unique()))

# Different regions for which data is collected

print('Regions: ',train_df['source'].unique())

# No of unique images for each region
region_list = []
unique_images = []
for region in train_df['source'].unique():
    region_list.append(region)
    unique_images.append(len(train_df[train_df['source']== region]['image_id'].unique()))
    print('Region: {}, Number of Images: {}'.format(str(region), len(train_df[train_df['source']== region]['image_id'].unique())))


# In[ ]:


fig, ax = plt.subplots()
ax.pie(unique_images, labels = region_list, autopct='%1.1f%%')
ax.axis('equal')
plt.show()


# Extracting bounding boxes data from the csv file. Function taken from Peter's [notebook](http://www.kaggle.com/pestipeti/global-wheat-detection-eda).

# The 'coco' format for the bounding boxes is (x_min, y_min, width, height) which has been originally provided in the dataframe. However, for my convenience, I have converted this to the 'pascal_voc' format i.e. (x_min, y_min, x_max, y_max). Note: all the functions can be modified using minor changes if you wish to use the 'coco' format.

# In[ ]:


train_df['x_min'] = -1
train_df['y_min'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall('([0-9]+[.]?[0-9]*)', x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x_min', 'y_min', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns = ['bbox'], inplace = True)
train_df['x_min'] = train_df['x_min'].astype(np.float)
train_df['y_min'] = train_df['y_min'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

train_df['x_max'] = train_df['x_min'] + train_df['w']
train_df['y_max'] = train_df['y_min'] + train_df['h']

train_df.head()


# Drop the unecessary columns.

# In[ ]:


train_df.drop(columns = ['width', 'height'], inplace=True)
train_df.head()


# Display an image with and without the bounding boxes.

# In[ ]:


def show_sample_images(image_data):
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    ax = ax.flatten()
    
    image = cv2.imread(os.path.join(TRAIN_DIR + '/{}.jpg').format(image_data.iloc[0]['image_id']), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    ax[0].set_title('Original Image')
    ax[0].imshow(image)
    
    for i, row in image_data.iterrows():
        cv2.rectangle(image,
                      (int(row['x_min']), int(row['y_min'])),
                      (int(row['x_max']), int(row['y_max'])),
                      (220, 0, 0), 3)
    
    ax[1].set_title('Image with Bounding Boxes')
    ax[1].imshow(image)
    
    plt.show()
        


# In[ ]:


show_sample_images(train_df[train_df['image_id'] == 'b6ab77fd7'])


# Apply a few basic augmentations.

# In[ ]:


def get_bboxes(bboxes, col, bbox_format = 'pascal_voc', color='white'):
    for i in range(len(bboxes)):
        x_min = bboxes[i][0]
        y_min = bboxes[i][1]
        x_max = bboxes[i][2]
        y_max = bboxes[i][3]
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        col.add_patch(rect)


# In[ ]:


def show_augmented_images(aug_result, image_data):
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    ax = ax.flatten()
    
    image = cv2.imread(os.path.join(TRAIN_DIR + '/{}.jpg').format(image_data.iloc[0]['image_id']), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    aug_image = aug_result['image']

    get_bboxes(pascal_voc_boxes, ax[0], color='red')
    orig_bboxes = pascal_voc_boxes
    ax[0].set_title('Original Image with Bounding Boxes')
    ax[0].imshow(image)

    get_bboxes(aug_result['bboxes'], ax[1], color='red')
    aug_bboxes = aug_result['bboxes']
    ax[1].set_title('Augmented Image with Bounding Boxes')
    ax[1].imshow(aug_image)
    
    plt.show()


# In[ ]:


image = cv2.imread(os.path.join(TRAIN_DIR + '/b6ab77fd7.jpg'), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
image_id = 'b6ab77fd7'


# In[ ]:


pascal_voc_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values
labels = np.ones((len(pascal_voc_boxes), ))


# a. CLAHE augmentation

# In[ ]:


aug = A.Compose([
    A.CLAHE(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# b. EQUALIZE augmentation

# In[ ]:


aug = A.Compose([
    A.Equalize(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# c. BLUR augmentation

# In[ ]:


aug = A.Compose([
    A.Blur(blur_limit=15, p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# d. RANDOM CROP augmentation

# In[ ]:


aug = A.Compose([
    A.RandomCrop(512, 512, p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# e. RESIZE augmentation

# In[ ]:


aug = A.Compose([
    A.Resize(512,512, p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# f. RANDOM GAMMA augmentation

# In[ ]:


aug = A.Compose([
    A.RandomGamma(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# g. RANDOM AFFINE augmentation

# In[ ]:


aug = A.Compose([
    A.ShiftScaleRotate(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# h. RANDOM BRIGHTNESS & CONTRAST augmentation

# In[ ]:


aug = A.Compose([
    A.RandomBrightnessContrast(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# i. RANDOM SIZED BBOX SAFE CROP augmentation

# In[ ]:


aug = A.Compose([
    A.RandomSizedBBoxSafeCrop(height=512, width = 512, p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# j. RANDOM RAIN augmentation

# In[ ]:


aug = A.Compose([
    A.RandomRain(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# k. RANDOM FOG augmentation

# In[ ]:


aug = A.Compose([
    A.RandomFog(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# l. RANDOM SOLAR FLARE augmentation

# In[ ]:


aug = A.Compose([
    A.RandomSunFlare(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# m. ISO NOISE augmentation

# In[ ]:


aug = A.Compose([
    A.ISONoise(p=1)
], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)

show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])


# I would highly appreciate if you could let me know how I can improve my kernels with the objective of making them really simple and easy to understand.

# Also, if you liked the kernel or found it useful, please upvote it so that I will be motivated further to share my work with everyone. Cheers!
