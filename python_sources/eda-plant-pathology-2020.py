#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import plotly.graph_objects as go
import hashlib

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from PIL import Image

"""
/kaggle/input/plant-pathology-2020-fgvc7/train.csv
/kaggle/input/plant-pathology-2020-fgvc7/test.csv
/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv
/kaggle/input/plant-pathology-2020-fgvc7/images/Test_956.jpg
"""

DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'


# In[ ]:


train_df = pd.read_csv(DIR_INPUT + '/train.csv')
train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


test_df = pd.read_csv(DIR_INPUT + '/test.csv')
test_df.shape


# In[ ]:


test_df.head()


# # Image metadata

# In[ ]:


def calculate_hash(im):
    md5 = hashlib.md5()
    md5.update(np.array(im).tostring())
    
    return md5.hexdigest()
    
def get_image_meta(image_id, image_src, dataset='train'):
    im = Image.open(image_src)
    extrema = im.getextrema()

    meta = {
        'image_id': image_id,
        'dataset': dataset,
        'hash': calculate_hash(im),
        'r_min': extrema[0][0],
        'r_max': extrema[0][1],
        'g_min': extrema[1][0],
        'g_max': extrema[1][1],
        'b_min': extrema[2][0],
        'b_max': extrema[2][1],
        'height': im.size[0],
        'width': im.size[1],
        'format': im.format,
        'mode': im.mode
    }
    return meta


# In[ ]:


data = []

for i, image_id in enumerate(tqdm(train_df['image_id'].values, total=train_df.shape[0])):
    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id)))


# In[ ]:


for i, image_id in enumerate(tqdm(test_df['image_id'].values, total=test_df.shape[0])):
    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id), 'test'))


# In[ ]:


meta_df = pd.DataFrame(data)
meta_df.head()


# ## Image orientations
# 
# It looks like we have both portrait and landscape modes in the train and the test set as well. The image size is always 2048x1368px (or 1365x2048).

# In[ ]:


meta_df.groupby(by='dataset')[['width', 'height']].aggregate(['min', 'max'])


# ## Duplications
# We have a few duplications:
# 
# - **train** `Train_379` and `Train_1173`
# - **test** `Test_683` and `Test_1691`
# - **test** `Test_570` and `Test_1212`
# - **mixed** `Train_1703` and `Test_1407`
# - **mixed** `Train_1505` and `Test_829`

# In[ ]:


duplicates = meta_df.groupby(by='hash')[['image_id']].count().reset_index()
duplicates = duplicates[duplicates['image_id'] > 1]
duplicates.reset_index(drop=True, inplace=True)

duplicates = duplicates.merge(meta_df[['image_id', 'hash']], on='hash')

duplicates.head(20)


# In[ ]:


fig, ax = plt.subplots(5, 2, figsize=(8, 16))
ax = ax.flatten()

for i in range(0, min(duplicates.shape[0], 10), 2):
    image_i = cv2.imread(DIR_INPUT + '/images/{}.jpg'.format(duplicates.iloc[i, 2]), cv2.IMREAD_COLOR)
    image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB)
    ax[i].set_axis_off()
    ax[i].imshow(image_i)
    ax[i].set_title(duplicates.iloc[i, 2])
    
    image_i_1 = cv2.imread(DIR_INPUT + '/images/{}.jpg'.format(duplicates.iloc[i + 1, 2]), cv2.IMREAD_COLOR)
    image_i_1 = cv2.cvtColor(image_i_1, cv2.COLOR_BGR2RGB)
    ax[i + 1].set_axis_off()
    ax[i + 1].imshow(image_i_1)
    ax[i + 1].set_title(duplicates.iloc[i + 1, 2])


# # Target distribution

# In[ ]:


def show_images(image_ids):
    
    col = 5
    row = min(len(image_ids) // col, 5)
    
    fig, ax = plt.subplots(row, col, figsize=(16, 8))
    ax = ax.flatten()

    for i, image_id in enumerate(image_ids):
        image = cv2.imread(DIR_INPUT + '/images/{}.jpg'.format(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(image_id)


# In[ ]:


fig = go.Figure(data=[
    go.Pie(labels=train_df.columns[1:],
           values=train_df.iloc[:, 1:].sum().values)
])
fig.show()


# ## Random images

# In[ ]:


show_images(train_df.sample(n=15)['image_id'].values)


# In[ ]:


show_images(test_df.sample(n=15)['image_id'].values)


# ## Healthy images

# In[ ]:


show_images(train_df[train_df['healthy'] == 1].sample(n=15)['image_id'].values)


# ## Infected with apple rust

# In[ ]:


show_images(train_df[train_df['rust'] == 1].sample(n=15)['image_id'].values)


# ## Have apple scab

# In[ ]:


show_images(train_df[train_df['scab'] == 1].sample(n=15)['image_id'].values)


# ## Multiple diseases

# In[ ]:


show_images(train_df[train_df['multiple_diseases'] == 1].sample(n=15)['image_id'].values)


# In[ ]:




