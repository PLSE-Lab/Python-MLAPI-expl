#!/usr/bin/env python
# coding: utf-8

# # Prostate cANcer graDe Assessment (PANDA) Challenge
# ## Prostate cancer diagnosis using the Gleason grading system

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/18647/logos/header.png)

# In[ ]:


import os
import cv2
import sys
import random
import openslide
import matplotlib
import skimage.io
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from IPython.display import Image, display
from skimage.transform import resize, rescale
sns.set_style("darkgrid")


# In[ ]:


ROOT = "/kaggle/input/prostate-cancer-grade-assessment/"
get_ipython().system('ls {ROOT}')


# In[ ]:


train = pd.read_csv(ROOT+"train.csv")
test = pd.read_csv(ROOT+"test.csv")
sub = pd.read_csv(ROOT+"sample_submission.csv")


# In[ ]:


display(train.head())
print("shape : ", train.shape)
print("unique ids : ", len(train.image_id.unique()))
print("unique data provider : ", len(train.data_provider.unique()))
print("unique isup_grade(target) : ", len(train.isup_grade.unique()))
print("unique gleason_score : ", len(train.gleason_score.unique()))


# In[ ]:


files = os.listdir(ROOT+"train_images/")
print(f"there are {len(files)} tiff files in train_images folder")
for i in train.image_id:
    assert i+".tiff" in files
print("all training image_ids have their files in train_images folder")


# In[ ]:


display(test.head())
print("shape : ", test.shape)
print("unique ids : ", len(test.image_id.unique()))
print("unique data provider : ", len(test.data_provider.unique()))


# In[ ]:


test['image_id'][0]+".tiff" in os.listdir(ROOT+"train_images/")


# test images are not present in given folder. I think test_images folder is missing

# In[ ]:


fig= plt.figure(figsize=(10,6))
ax = sns.countplot(x="isup_grade", data=train)
plt.title("target distribution")
plt.show()


# In[ ]:


fig= plt.figure(figsize=(10,6))
ax = sns.countplot(x="isup_grade", hue="data_provider", data=train)
plt.show()


# target variable shows different distribution depending on data provider

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
sns.countplot(ax=ax1, x="data_provider", data=train)
ax1.set_title("data_provider distribution in training data")
sns.countplot(ax=ax2, x="data_provider", data=test)
ax2.set_title("data_provider distribution in test data")
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("counts of different gleason_score")
sns.countplot(ax=ax1, y="gleason_score", data=train)
sns.countplot(ax=ax2, y="gleason_score",hue='data_provider', data=train)
plt.show()


# In[ ]:


print("isup_score ->  ", end='')
for j in train['isup_grade'].unique():
    print(j, end="\t")
print("\n", "gleason_score \n", "-"*60, sep='')

for i in train['gleason_score'].unique():
    print(f"{i:>10} |   ", end="")
    for j in train['isup_grade'].unique():
        print(len(train[(train['gleason_score']==i) & (train['isup_grade']==j)]), end="\t")
    print("")


# **negative and 0+0 are same just different names from different center**

# In[ ]:


train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("counts of different gleason_score after correction")
sns.countplot(ax=ax1, y="gleason_score", data=train)
sns.countplot(ax=ax2, y="gleason_score",hue='data_provider', data=train)
plt.show()


# In[ ]:


slide = openslide.OpenSlide(ROOT+"train_images/"+files[0])
spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
print(f"File id: {slide}")
print(f"Dimensions: {slide.dimensions}")
print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
print(f"Number of levels in the image: {slide.level_count}")
print(f"Downsample factor per level: {slide.level_downsamples}")
print(f"Dimensions of levels: {slide.level_dimensions}")
patch = slide.read_region((1780,1950), 0, (256, 256))
display(patch) # Display the image
slide.close()


# In[ ]:


dims, spacings, level_counts = [], [], []
down_levels, level_dims = [], []

# train = train.sample(300)
for i in train.image_id:
    slide = openslide.OpenSlide(ROOT+"train_images/"+i+".tiff")
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    dims.append(slide.dimensions)
    spacings.append(spacing)
    level_counts.append(slide.level_count)
    down_levels.append(slide.level_downsamples)
    level_dims.append(slide.level_dimensions)
    slide.close()
    del slide

train['width']  = [i[0] for i in dims]
train['height'] = [i[1] for i in dims]
train['spacing'] = spacings
train['level_count'] = level_counts


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax = sns.scatterplot(x='width', y='height', data=train, alpha=0.3)
plt.title("height(y) width(x) scatter plot")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12, 6))
ax = sns.scatterplot(x='width', y='height', hue='isup_grade', data=train, alpha=0.6)
plt.title("height(y) width(x) scatter plot with target")
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
sns.distplot(ax=ax1, a=train['width'])
ax1.set_title("width distribution")
sns.distplot(ax=ax2, a=train['height'])
ax2.set_title("height distribution")
plt.show()


# In[ ]:


shapes = [j for i in level_dims for j in i]
level  = np.array([j for i in level_dims for j in range(len(i))])
widths  = np.array([i[0] for i in shapes])
heights = np.array([i[1] for i in shapes])
fig, axes = plt.subplots(1, 3 ,figsize=(10,4))
for i in range(3):
    ax = sns.scatterplot(ax=axes[i], x=widths[level==i], y=heights[level==i], alpha=0.9)
    axes[i].set_title(f"level {i}")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12,6))
sns.scatterplot(x=widths, y=heights,hue=level, alpha=0.9)
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("available different levels")
sns.distplot(ax=ax1, a=widths)
ax1.set_title("width distribution")
sns.distplot(ax=ax2, a=heights)
ax2.set_title("height distribution")
plt.show()


# In[ ]:


print(f"unique level counts : {train['level_count'].unique()}")
print(f"unique spacings     : {train['spacing'].unique()}")
print(f"unique down levels  : {pd.Series([round(j) for i in down_levels for j in i]).unique()}")


# In[ ]:


masks = os.listdir(ROOT+'train_label_masks/')
print(f"masks available for {len(masks)} out of {len(files)}")


# In[ ]:


def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):
    """Print some basic information about a slide"""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        display(mask_data)

    # Compute microns per pixel (openslide gives resolution in centimeters)
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


# In[ ]:


mask = openslide.OpenSlide(os.path.join(ROOT+"train_label_masks", '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
print_mask_details(mask, center='radboud')
mask.close()


# In[ ]:


mask = openslide.OpenSlide(os.path.join(ROOT+"train_label_masks", '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
print_mask_details(mask, center='karolinska')
mask.close()


# In[ ]:


count = 5
def plot_with_images(masks):
    fig, axes = plt.subplots(2, count, figsize=(4*count, 8))
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    for i, j in enumerate(masks):
        mask = openslide.OpenSlide(os.path.join(ROOT+"train_label_masks", j))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        image = openslide.OpenSlide(os.path.join(ROOT+"train_images", j[:-10]+".tiff"))
        patch = image.read_region((0, 0), image.level_count-1, image.level_dimensions[-1]) 
        axes[0, i].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        axes[1, i].imshow(patch)
        mask.close()
        image.close()
    plt.tight_layout()
    plt.show()

for i in range(7):
    plot_with_images(masks[i*count: (i+1)*count])


# ## Background

# In[ ]:


def get_image(id, level):
    im = skimage.io.MultiImage(ROOT+"train_images/"+id+".tiff")
    return im[level]

def get_mask(id, level):
    mask = skimage.io.MultiImage(ROOT+"train_label_masks/"+id+"_mask.tiff")
    return mask[level]


# In[ ]:


ids_with_mask = [i[:-10] for i in os.listdir(ROOT+"train_label_masks/")]
ids = [(i, j) for i,j in zip(train.image_id, train.data_provider) if i in ids_with_mask]


# In[ ]:


# for t, loc in tqdm(ids):
#     image = get_image(t, 2)
#     mask = get_mask(t, 2)
#     assert np.sum(mask[:, :, 1:]) == 0
#     assert(image.shape == mask.shape)
#     if loc == "karolinska":
#         assert np.max(mask[:, :, 0]) <= 2


# **Last 2 channels of masks are empty also for karolinska values are 0, 1, 2 not 1, 2, 3 **

# In[ ]:


sns.set_style("whitegrid")

def get_mask(id, level):
    """
    updated mask to return only first channel
    """
    mask = skimage.io.MultiImage(ROOT+"train_label_masks/"+id+"_mask.tiff")
    assert np.sum(mask[level][:, :, 1:]) == 0
    return mask[level][:, :, 0]

def get_background_mask(id, level):
    mask = get_mask(id, level)
    return (mask == 0)


# In[ ]:


for i in range(4):
    t = random.randint(0, len(ids))
    mask = get_background_mask(ids[t][0], 2)
    image = get_image(ids[t][0], 2)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(mask)
    axes[1].imshow(image)
    plt.tight_layout()
    plt.show()


# In[ ]:


def pred_background(image, low=240, high=255):
    """
    predicting background with threshold
    """
    (r, g, b) = cv2.split(image)
    ret0, thresh0 = cv2.threshold(r, low, high, cv2.THRESH_BINARY)
    ret1, thresh1 = cv2.threshold(g, low, high, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(b, low, high, cv2.THRESH_BINARY)
    return (thresh0 & thresh1 & thresh2) == 255


# In[ ]:


for i in range(5):
    t = random.randint(0, len(ids))
    mask = get_background_mask(ids[t][0], 2)
    image = get_image(ids[t][0], 2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(mask)
    axes[1].imshow(image)
    axes[2].imshow(pred_background(image))
    plt.show()


# **We can predict background using thresholding as most of background is white **

# ### do upvote if it helped :)
