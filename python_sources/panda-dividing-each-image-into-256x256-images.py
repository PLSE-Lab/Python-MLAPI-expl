#!/usr/bin/env python
# coding: utf-8

# ## Idea
# The purpose of this notebook is to explain the procedure of dividing the given large image with any shape, into multiple 256x256 fixed sized images, which can be further used to train a model with fixed input size. 
# 
# After dividing the given image into multiple smaller images, there will be many images which has very small details or in many cases no details (fully white image), we'll discard those images as they are not useful for training models.

# In[ ]:


get_ipython().system('pip install iterative-stratification')


# In[ ]:


import os

import math
import openslide
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from PIL import Image, ImageChops

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[ ]:


BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment'
DATA_DIR = os.path.join(BASE_DIR, 'train_images')


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))\ntest_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))\nsample_sub_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))")


# In[ ]:


train_df.head()


# Let's first read and visualize the training image which we are going to use as an example in this notebook.

# In[ ]:


img = os.path.join(DATA_DIR, f'{train_df["image_id"].iloc[5]}.tiff')
img = openslide.OpenSlide(img)
patch = img.read_region((0, 0), 2, img.level_dimensions[-1])
img.close()
patch


# Here are some constants' definition, which you can modify if you want higher dimensional images instead of 256x256 or you want to use higher resolution images.

# In[ ]:


crop_size = 256  # Size of resultant images
crop_level = 2  # The level of slide used to get the images (you can use 0 to get very high resolution images)
down_samples = [1, 4, 16]  # List of down samples available in any tiff image file


# The below code will crop the given image and will store the result in an array

# In[ ]:


def split_image(openslide_image):
    """
    Splits the given image into multiple images if 256x256
    """
    
    # Get the size of the given image
    width, height = openslide_image.level_dimensions[crop_level]

    # Get the dimensions of level 0 resolution, as it's required in "read_region()" function
    base_height = down_samples[crop_level] * height  # height of level 0
    base_width = down_samples[crop_level] * width  # width of level 0

    # Get the number of smaller images 
    h_crops = math.ceil(width / crop_size)
    v_crops = math.ceil(height / crop_size)

    splits = []
    for v in range(v_crops):
        for h in range(h_crops): 
            x_location = h*crop_size*down_samples[crop_level]
            y_location = v*crop_size*down_samples[crop_level]

            patch = openslide_image.read_region((x_location, y_location), crop_level, (crop_size, crop_size))

            splits.append(patch)
    return splits, h_crops, v_crops


# In[ ]:


img = os.path.join(DATA_DIR, f'{train_df["image_id"].iloc[5]}.tiff')
img = openslide.OpenSlide(img)
crops, h_crops, v_crops = split_image(img)
img.close()


# Now as we have all the smaller images available. Let's plot all of them and verify out result by comparing it to original image.

# In[ ]:


fig, ax = plt.subplots(nrows=v_crops, ncols=h_crops, figsize=(12, 12))
count=0
for row in ax:
    for col in row:
        patch = crops[count]
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(patch)
        count += 1
plt.show()


# This looks promising. The only thing remaining is to discard the 'white' images. Let's do it.

# In[ ]:


def get_emptiness(arr):
    total_ele = arr.size
    white_ele = np.count_nonzero(arr == 255) + np.count_nonzero(arr == 0)
    return white_ele / total_ele


# In[ ]:


ignore_threshold = 0.95  # If the image is more than 95% empty, consider it as white and ignore


# In[ ]:


def filter_white_images(images):
    non_empty_crops = []
    for image in images:
        image_arr = np.array(image)[...,:3]  # Discard the alpha channel
        emptiness = get_emptiness(image_arr)
        if emptiness < ignore_threshold:
            non_empty_crops.append(image)
    return non_empty_crops


# In[ ]:


non_empty_crops = filter_white_images(crops)


# In[ ]:


len(non_empty_crops)


# Let's plot all the crops separately and verify that all the crops have meaningful details and are not empty.

# In[ ]:


for f in non_empty_crops:
    display(f)


# Cheers! We divided a large image into multiple images of size 256x256. Repeat this procedure on each of the training image to get the fixed sized images without lossing any data. 

# Now let's do the above procedure for entire training dataset and save the resultant dataset so that it can be used further in training the models.

# In[ ]:


# train_df = train_df.loc[:20]


# In[ ]:


get_ipython().system('mkdir train_images')


# In[ ]:


dataset = []
def create_dataset(count):
    img = os.path.join(DATA_DIR, f'{train_df["image_id"].iloc[count]}.tiff')
    img = openslide.OpenSlide(img)
    crops, _, _ = split_image(img)
    img.close()

    non_empty_crops = filter_white_images(crops)
    image_id = train_df['image_id'].iloc[count]

    for index, img in enumerate(non_empty_crops):
        img_metadata = {}
        img = img.convert('RGB')

        img_metadata['image_id'] = f'{image_id}_{index}'
        img_metadata['data_provider'] = train_df['data_provider'].iloc[count]
        img_metadata['isup_grade'] = train_df['isup_grade'].iloc[count]
        img_metadata['gleason_score'] = train_df['gleason_score'].iloc[count]

        img.save(f'train_images/{image_id}_{index}.jpg', 'JPEG', quality=100, optimize=True, progressive=True)
        dataset.append(img_metadata)
    return dataset


# In[ ]:


dataset = Parallel(n_jobs=8)(delayed(create_dataset)(count) for count in tqdm(range(len(train_df))))
dataset = [item for sublist in dataset for item in sublist]

dataset = pd.DataFrame(dataset)


# ## Creating k-folds

# In[ ]:


dataset.loc[:, 'kfold'] = -1

# Randomly shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset[['image_id', 'data_provider']].values
y = dataset[['isup_grade', 'gleason_score']].values

mskf = MultilabelStratifiedKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
    dataset.loc[val_idx, 'kfold'] = fold

print(dataset.kfold.value_counts())


# In[ ]:


dataset


# In[ ]:


dataset.to_csv('train.csv', index=False)


# In[ ]:


import zipfile
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('train.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/kaggle/working/train_images', zipf)
zipf.close()


# In[ ]:


get_ipython().system('mv train.zip train')


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/train_images')


# In[ ]:


get_ipython().system('ls')

