#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, ast
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # plotting
import matplotlib.patches as patches
import seaborn as sns # plotting


# In[ ]:


get_ipython().system('ls /kaggle/input/global-wheat-detection')


# ## Reading the given dataset

# In[ ]:


# Some constants
BASE_DIR = '/kaggle/input/global-wheat-detection'


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))\nsample_sub_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))")


# In[ ]:


train_df.head()


# In[ ]:


sample_sub_df.head()


# Let's see the size of the dataset

# In[ ]:


print(f'Shape of training data: {train_df.shape}')
print(f'Shape of given test data: {sample_sub_df.shape}')


# Let's find out number of unique images in training dataset

# In[ ]:


print(f'# of unique images: {train_df["image_id"].nunique()}')


# ## Distribution of the size of training images

# In[ ]:


print(f'Unique heights and widths: {train_df["width"].unique()}, {train_df["height"].unique()}')


# ##  Min, Max and Average number of wheat heads

# In[ ]:


print(f'Minimum number of wheat heads: {max(train_df["image_id"].value_counts())}')
print(f'Minimum number of wheat heads: {len(train_df)/train_df["image_id"].nunique()}')


# In[ ]:


print(f'Total number of images: {len(os.listdir(os.path.join(BASE_DIR, "train")))}')


# So there are 3422 images in traing directory and 3373 images are annotated, that means there are 49 images which do not have any wheat heads in it (without annotation)

# Summary of data:
#  * Number of images: 3373
#  * Avg. # of wheat heads per image: ~44
#  * All images have same dimension: 1024 * 1024
#  * Number of minimum bounding boxes (Wheat heads): 0 (Some of the images are without annotation)
#  * Number of maximum bounding boxes (Wheat heads): 116

# ## Distribution of number of bounding boxes

# In[ ]:


sns.distplot(train_df['image_id'].value_counts(), kde=False)
plt.xlabel('# of wheat heads')
plt.ylabel('# of images')
plt.title('# of wheat heads vs. # of images')
plt.show()


# In[ ]:


train_df[['x_min','y_min', 'width', 'height']] = pd.DataFrame([ast.literal_eval(x) for x in train_df.bbox.tolist()], index= train_df.index)
train_df = train_df[['image_id', 'bbox', 'source', 'x_min', 'y_min', 'width', 'height']]
train_df


# ## Distribution of size of bounding boxes

# In[ ]:


sns.distplot(train_df['width'] * train_df['height'], kde=False)
plt.xlabel('Area of bbox')
plt.ylabel('# of images')
plt.title('Area of bbox vs. # of images')
plt.show()


# ## Visualizing a few training images without bounding boxes

# In[ ]:


# Visualize few samples of current training dataset
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
count=0
for row in ax:
    for col in row:
        img = plt.imread(f'{os.path.join(BASE_DIR, "train", train_df["image_id"].unique()[count])}.jpg')
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(img)
        count += 1
plt.show()


# In[ ]:


# Visualize few samples of current training dataset
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
count=0
for row in ax:
    for col in row:
        img = plt.imread(f'{os.path.join(BASE_DIR, "train", train_df["image_id"].unique()[-count])}.jpg')
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(img)
        count += 1
plt.show()


# ## Now let's visualize a few training images with bounding boxes

# In[ ]:


def get_bbox(image_id, df, col, color='white'):
    bboxes = df[df['image_id'] == image_id]
    
    for i in range(len(bboxes)):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bboxes['x_min'].iloc[i], bboxes['y_min'].iloc[i]),
            bboxes['width'].iloc[i], 
            bboxes['height'].iloc[i], 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none')

        # Add the patch to the Axes
        col.add_patch(rect)


# In[ ]:


# Visualize few samples of current training dataset
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
count=0
for row in ax:
    for col in row:
        img_id = train_df["image_id"].unique()[count]
        img = plt.imread(f'{os.path.join(BASE_DIR, "train", img_id)}.jpg')
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        get_bbox(img_id, train_df, col, color='red')
        col.imshow(img)
        count += 1
plt.show()


# ## few more...

# In[ ]:


# Visualize few samples of current training dataset
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
count=0
for row in ax:
    for col in row:
        img_id = train_df["image_id"].unique()[-count]
        img = plt.imread(f'{os.path.join(BASE_DIR, "train", img_id)}.jpg')
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        get_bbox(img_id, train_df, col)
        col.imshow(img)
        count += 1
plt.show()


# ## Image with highest number of wheat heads

# In[ ]:


image_id = (train_df['image_id'].value_counts() == 116).index[0]
img = plt.imread(f'{os.path.join(BASE_DIR, "train", image_id)}.jpg')

fig, ax = plt.subplots(1, figsize=(12, 12))

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
get_bbox(image_id, train_df, ax, color='red')
ax.imshow(img)
plt.plot()


# ## Some of the images without any wheat head

# In[ ]:


all_images = os.listdir(os.path.join(BASE_DIR, 'train'))
all_images = set([x[:-4] for x in all_images])

images_with_bbox = set(list(train_df['image_id']))

images_without_bbox = list(all_images - images_with_bbox)


# In[ ]:


print(f'Total number of images without wheat heads: {len(images_without_bbox)}')


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
count=20
for row in ax:
    for col in row:
        img_id = images_without_bbox[count]
        img = plt.imread(f'{os.path.join(BASE_DIR, "train", img_id)}.jpg')
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(img)
        count += 1
plt.show()


# In[ ]:




