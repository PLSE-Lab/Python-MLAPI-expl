#!/usr/bin/env python
# coding: utf-8

# # Train Dataset visualization
# 
# Idea of the notebook is to:
# - visualize train images with masks to have an overview of the training dataset
# - check md5 hashes between train/test datasets
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt


# In[ ]:


from pathlib import Path

input_path = Path(".").absolute().parent / "input"
train_path = input_path / "train"
test_path = input_path / "test"

train_df = pd.read_csv(input_path / "train.csv")


# In[ ]:


train_df.head()


# ## Train dataset visualization

# Let's display a single image and mask

# In[ ]:


from PIL import Image
# pip install image-dataset-viz
from image_dataset_viz import render_datapoint


# In[ ]:


def read_image(data_id, is_train=True):    
    path = train_path if is_train else test_path
    path = (path / "images" / "{}.png".format(data_id))
    img = Image.open(path)
    img = img.convert('RGB')
    return img
    
def read_mask(data_id, is_train=True):
    path = train_path if is_train else test_path
    path = (path / "masks" / "{}.png".format(data_id))    
    img = Image.open(path)
    bk = Image.new('L', size=img.size)
    g = Image.merge('RGB', (bk, img.convert('L'), bk))
    return g

img = read_image("34e51dba6a")
mask = read_mask("34e51dba6a")


# In[ ]:


rimg = render_datapoint(img, mask, blend_alpha=0.3)
print(rimg.size)
rimg


# Now let's 'export' whole train dataset in a few images. So that you can download images and explore them using your favorite image viewer.

# In[ ]:


data_ids = train_df['id'].values.tolist()


from image_dataset_viz import DatasetExporter


de = DatasetExporter(read_image, read_mask, blend_alpha=0.3, n_cols=20, max_output_img_size=(100, 100))
de.export(data_ids, data_ids, "train_dataset_viz")


# In[ ]:


get_ipython().system('ls train_dataset_viz')


# Let's open one output image with PIL. But again, it's better to download these images from the output folder and explore locally with an image viewer.

# In[ ]:


ds_image = Image.open("train_dataset_viz/dataset_part_0.png")


# In[ ]:


ds_image


# ## Check same images between train and test

# In[ ]:


depths_df = pd.read_csv(input_path / "depths.csv")


# Compute hashes for train/test datasets

# In[ ]:


import tqdm
import hashlib

md5_df = pd.DataFrame(data=depths_df['id'], columns=['id'], index=depths_df.index)
data = []
is_train = []
for data_id in tqdm.tqdm(md5_df['id'].values):    
    p = (train_path / "images" / "{}.png".format(data_id))
    b = True
    if not p.exists():
        p = (test_path / "images" / "{}.png".format(data_id))    
        b = False
    image_file = p.open('rb').read()
    data.append(hashlib.md5(image_file).hexdigest())
    is_train.append(b)    
md5_df['hash'] = data
md5_df['is_train'] = is_train


# In[ ]:


md5_df.head()


# ### Let's check duplicates in the training dataset

# In[ ]:


train_duplicated_mask = md5_df[md5_df['is_train'] == True]['hash'].duplicated()
train_duplicates = md5_df[(md5_df['is_train'] == True) & train_duplicated_mask]
train_duplicates['hash'].unique(), train_duplicates['id'].values[:3], len(train_duplicates['id'])


# In[ ]:


read_image('e82421363e', is_train=True)


# All duplicated images in the training dataset are just black images. Do they have non-null masks ?

# In[ ]:


rle_mask_for_black_img = train_df[train_df['id'].isin(train_duplicates['id'])]['rle_mask']
rle_mask_for_black_img.isnull().all()


# Yes, they all have NaN mask

# ### Let's check duplicates in the test dataset

# In[ ]:


test_duplicated_mask = md5_df[md5_df['is_train'] == False]['hash'].duplicated()
test_duplicates = md5_df[(md5_df['is_train'] == False) & test_duplicated_mask]
test_duplicates['hash'].unique(), test_duplicates['id'].values[:3], len(test_duplicates['id'])


# In[ ]:


read_image('5e52f098d9', is_train=False)


# All duplicated images in the test dataset are just black images

# ### Let's check duplicates between training and test datasets

# In[ ]:


train_mask = md5_df['is_train'] == True
train_md5_df = md5_df[train_mask]
test_md5_df = md5_df[~train_mask]
same_hash_mask = test_md5_df['hash'].isin(train_md5_df['hash'])
test_md5_df[same_hash_mask]['hash'].unique(), test_md5_df[same_hash_mask]['id'].values[:3]


# In[ ]:


read_image('353e010b7b', is_train=False)


# All duplicated images between train and test dataset are just black images

# In[ ]:





# ## Rectangular masks
# 
# There masks defined as rectangles and could look weird on the background image. More information can be found in the [following thread](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61720).

# In[ ]:


depth_df = pd.read_csv(input_path / "depths.csv", index_col='id')
depth_df.head()


# In[ ]:


non_nan_mask = ~train_df['rle_mask'].isnull()

def rle_to_len(mask_str):
    mask = mask_str.split(' ')
    return len(mask)

train_df.loc[non_nan_mask, 'rle_mask_len'] = train_df.loc[non_nan_mask, 'rle_mask'].apply(rle_to_len)


# In[ ]:


def get_depth(data_id):
    return depth_df.loc[data_id, 'z']

train_df.loc[non_nan_mask, 'z'] = train_df.loc[non_nan_mask, 'id'].apply(get_depth)


# In[ ]:


vertical_masks = train_df[train_df['rle_mask_len'] < 3]
vertical_masks.head()


# However as it stated in [the comment](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61720#362630) :
# 
# > Salt is very difficult to locate accurately, and if faced with ambiguity, a geophysicist will draw the simplest possible polygon to define it. Since salt can play havoc on time to depth conversions (velocity models), you can get some really weird vertical features in the seismic
# 
# We should carefully select the vertical masks:

# In[ ]:


data_id = "d4d34af4f7"
img = read_image(data_id)
mask = read_mask(data_id)

plt.figure(figsize=(7, 7))
plt.subplot(121)
plt.title("Depth = {}".format(depth_df.loc[data_id, 'z']))
plt.imshow(img)
plt.subplot(122)
plt.imshow(mask)


# In[ ]:


data_id = "7845115d01"
img = read_image(data_id)
mask = read_mask(data_id)

plt.figure(figsize=(7, 7))
plt.subplot(121)
plt.title("Depth = {}".format(depth_df.loc[data_id, 'z']))
plt.imshow(img)
plt.subplot(122)
plt.imshow(mask)


# In[ ]:


data_id = "b525824dfc"
img = read_image(data_id)
mask = read_mask(data_id)

plt.figure(figsize=(7, 7))
plt.subplot(121)
plt.title("Depth = {}".format(depth_df.loc[data_id, 'z']))
plt.imshow(img)
plt.subplot(122)
plt.imshow(mask)


# In[ ]:





# In[ ]:




