#!/usr/bin/env python
# coding: utf-8

# ## Airbus Ship Detection Project

# Dataset:
# 
# - Load CSV files
#     - Label Dataset: find how many unique images with and without ships
#     - Count Ships: count how many ships in images
# - Show sample images
#     - show area given by EncodedPixels in a bounding box
# 
# Things considered:
# 
# - check if there are duplicates of 'ImageId'
# - do not use 'ImageId' as index, it is not unique
# - image may have several copies and each one may have a partial set of ship pixels
# - an image copy may show one or more ships
# - check if there are duplicates of 'EncodedPixels', whether the whole string or a tuple

# ## Data Analysis and Visualization

# ### Load CSV Files

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import csv
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


basedir = '../input/train_v2/'


# ### TRAIN IMAGES

# In[ ]:


train_df = pd.read_csv("../input/train_ship_segmentations_v2.csv")
train_df.head()


# ### TRAIN IMAGES WITH LABELS

# In[ ]:


train_df["GotShip"] = 0
train_df.loc[train_df["EncodedPixels"].notnull(), "GotShip"] = 1
# train_df['GotShips'] = np.where(train_df['EncodedPixels'].isnull(), 0, 1)
train_df.head()


# ### TRAIN IMAGES WITHOUT SHIPS

# In[ ]:


print('Number of images without ships in train log: ', train_df.ImageId[train_df['GotShip'] == 0].nunique())

# train_df.to_csv("./dataset/train/ships.csv")


# In[ ]:


noship = train_df[train_df['GotShip'] == 0]
noship.head()


# ### SAMPLE IMAGES WITHOUT SHIPS

# In[ ]:


def show_samples(imagedata, no_of_images, no_of_rows=4, no_of_cols=4):
    i = 0
    ship_sx = random.sample(range(0, len(imagedata)), no_of_images)
    samples = imagedata.iloc[ship_sx]
    fig = plt.figure(1, figsize = (20,20))
    for index, row in samples.iterrows():
        i = i + 1
        image = mpimg.imread(basedir + row['ImageId'])
        img = image.copy()
        rszImg = cv2.resize(img, (200, 200), cv2.INTER_AREA)

        ax = fig.add_subplot(no_of_rows, no_of_cols, i)
        ax.set_title(index)
        ax.imshow(rszImg)
        fig.tight_layout()  


# In[ ]:


show_samples(noship, 16)


# ### TEST IMAGES WITH SHIPS 

# In[ ]:


print('Number of images with ships in train log: ', train_df.ImageId[train_df['GotShip'] != 0].size)
print('Number of unique images with ships in train log: ', train_df.ImageId[train_df['GotShip'] != 0].nunique())

# train_df.to_csv("./dataset/train/ships.csv")


# In[ ]:


ship = train_df[train_df['GotShip'] != 0]
ship.head(10)


# ### SAMPLE IMAGES WITH SHIPS

# In[ ]:


show_samples(ship, 15)


# Several images with ships have similar ImageId but different EncodedPixels. An example is given below.

# In[ ]:


x = train_df[train_df["ImageId"] == "000194a2d.jpg"]
x


# ### SAMPLE IMAGES WITH SHIPS, SIMILAR IMAGEID

# In[ ]:


show_samples(x, 5)


# In[ ]:


# CHECK THAT NO DUPLICATE ENCODEDPIXELS ARE LISTED
duped_ship = ship.drop_duplicates("EncodedPixels")
print (len(duped_ship))


# ## Count Ships

# ### IMAGES WITH/WITHOUT SHIPS DISTRIBUTION

# In[ ]:


df1 = pd.DataFrame({'':['Ship', 'No Ship'], 'Image Count':[len(ship), len(noship)]})
df1


# In[ ]:


df1.plot.bar(x='', y='Image Count', rot=0, color='b', legend=None, title="Ship Count Distribution")


# ### IMAGES WITH DUPLICATES

# In[ ]:


# COUNT THE NUMBER OF DUPLICATES EACH IMAGE HAS
unique_ship = ship['ImageId'].value_counts().reset_index()
unique_ship.columns = ['ImageId', 'NumberOfDuplicates']
unique_ship.head()


# In[ ]:


# COUNT THE NUMBER OF IMAGES vs NUMBER OF DUPLICATES 
dupeship = unique_ship.groupby('NumberOfDuplicates').count()
dupeship


# In[ ]:


plt.figure()
df2 = pd.DataFrame(dupeship, columns=['NumberOfDuplicates', 'ImageId'])
ax = df2.plot.bar(color='r', legend=None, title="Ship Duplicates Distribution")
ax.set_xlabel("Number of Duplicates")
ax.set_ylabel("Number of Images")


# ### NUMBER OF SHIPS PER IMAGE DISTRIBUTION

# In[ ]:


# SAMPLE
idx = random.sample(range(0, len(ship)), 1)
sx_one = ship.iloc[idx]
encodedpixels = sx_one['EncodedPixels'].values
sx_image = sx_one['ImageId'].values


# In[ ]:


sx = sx_image[0]
sx_base = basedir + sx
sx_base


# In[ ]:


sample_data = ship[ship['ImageId'] == sx]
sample_data


# In[ ]:


unique_ship.NumberOfDuplicates[unique_ship['ImageId'] == sx_image[0]]


# ##### THE MASK

# In[ ]:


# CREATE AN IMAGE MASK
mask = np.zeros((768, 768))

# UNRAVEL MASK INTO ARRAY
mask = mask.ravel()
mask


# ##### THE ENCODED PIXELS

# In[ ]:


# CREATE SHIP MASK
def encode_rle(encodedpixels, n=2):
    # SPLIT ENCODED PIXELS STRING
    shipmask = encodedpixels.split()
    # CONVERT LIST TO TUPLES
    shipmask = zip(*[iter(shipmask)]*n)
    # CONVERT STRING TO INT
    rle = [(int(start), int(start) + int(length)) for start, length in shipmask]
    return rle


# In[ ]:


rle_data = sample_data['EncodedPixels'].apply(encode_rle)
rle_data


# ##### IMAGE MASK AND ENCODEDPIXELS COMBINED

# In[ ]:


def total_mask(rle_data, mask):
    for rle in rle_data:
        for start,end in rle:
            print (start, end)
            mask[start:end] = 1
    mask = mask.reshape(768,768).T
    return mask


# In[ ]:


mask = total_mask(rle_data, mask)


# In[ ]:


img_mask = np.dstack((mask, mask, mask))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img_mask)


# In[ ]:


# SHOW MASK IMAGE
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
ax.set_title("RLE Masking")
ax.imshow(img_mask)

orig_image = mpimg.imread(sx_base)
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Orig Image")
ax.imshow(orig_image)
fig.tight_layout()


# In[ ]:


x = range(1200)
fig, ax = plt.subplots(1, figsize = (50,50))
ax.imshow(orig_image, extent=[0, 1200, 0, 1200])


# In[ ]:


poly = np.ascontiguousarray(mask, dtype=np.uint8)
(flags, contours, h) = cv2.findContours(poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# In[ ]:


contour_image = orig_image.copy()
cv2.drawContours(contour_image, contours, -1, (0,255,0), 1)


# In[ ]:


x = range(1200)
fig, ax = plt.subplots(1, figsize = (50,50))
ax.imshow(contour_image, extent=[0, 1200, 0, 1200])

