#!/usr/bin/env python
# coding: utf-8

# # <div class="h2">Data Overview</div>

# In[ ]:


from PIL import Image, ImageDraw
import glob 
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv("../input/imaterialist-fashion-2020-fgvc7/train.csv")


# In[ ]:


train_df.keys()


# In[ ]:


train_df.loc[:30]


# In[ ]:


Unique_ImageId = set(train_df["ImageId"])


# In[ ]:


print(f"There are {len(train_df)} unique record in train.csv." )
print(f"There are {len(Unique_ImageId)} unique data." )


# ### Train data

# In[ ]:


data_path = [ "../input/imaterialist-fashion-2020-fgvc7/train/" + Id + ".jpg" for Id in Unique_ImageId]


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i,im_path in enumerate(data_path[:16]):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    im = Image.open(im_path)
    im = im.resize((350,480))
    plt.imshow(im)


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i,im_path in enumerate(data_path[32:48]):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    im = Image.open(im_path)
    im = im.resize((350,480))
    plt.imshow(im)


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i,im_path in enumerate(data_path[16:32]):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    im = Image.open(im_path)
    im = im.resize((350,480))
    plt.imshow(im)


# ### Test data

# In[ ]:


test_jpeg = glob.glob('../input/imaterialist-fashion-2020-fgvc7/test/*')


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i,im_path in enumerate(test_jpeg[:16]):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    im = Image.open(im_path)
    im = im.resize((350,480))
    plt.imshow(im)


#  Thank you for your reading!
