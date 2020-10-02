#!/usr/bin/env python
# coding: utf-8

# # Turning Image Tiles into Pandas DataFrame + Tiling the Images Together
# 
# Credit to the tile image dataset creator: lafoss [link](https://www.kaggle.com/iafoss/panda-16x128x128-tiles-data)
# 
# ![img](https://i.ibb.co/hF6LRVm/TILE.png)
# 
# Before that, why would anyone need this?
# I am creating this for those who will be using **Keras FlowFromDataFrame method** ([documentation](https://keras.io/api/preprocessing/image/#flowfromdataframe-method))
# 
# And also for those who want to combine these tiles into **1 single image**.
# 
# 
# Lets get it done! 

# First lets import all the things we need.

# In[ ]:


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd


# # Loading the data csv
# 
# We use pandas read csv method.

# In[ ]:


df_train = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
df_train.head()


# # Getting image files list
# 
# For this, I used Glob.
# The images are formatted in the following manner:
# 
# * 0005f7aaab2800f6170c399693a96917_0.png
# * 0005f7aaab2800f6170c399693a96917_1.png
# * 0005f7aaab2800f6170c399693a96917_2.png
# * 0005f7aaab2800f6170c399693a96917_3.png
# 
# ... and so on.
# 

# In[ ]:


img_path = "../input/panda-16x128x128-tiles-data/train"
img_id = list(df_train["image_id"])
img_files = glob.glob(img_path + f"/{img_id[1]}" + "*")


# # Get data function
# 
# This is just to get the datas

# In[ ]:


def get_df():
    data = {"image_id": [], "isup_grade": []}
    img_path = "../input/panda-16x128x128-tiles-data/train"
    img_ids = list(df_train["image_id"])
    labels = list(df_train["isup_grade"])
    for i in tqdm(range(len(img_ids))):
        img_id = img_ids[i]
        img_files = []
        label = [labels[i]] * 16
        for i in range(16):
            img_files.append(f"{img_id}"+f"_{i}"+".png")
        data["image_id"].extend(img_files)
        data["isup_grade"].extend(label)
        
    return data


# # Get our pandas dataframe!

# In[ ]:


def to_pandas(data):
    df = pd.DataFrame(data, columns = ["image_id", 'isup_grade'])
    return df


# In[ ]:


data = get_df()
df_new = to_pandas(data)

df_new.head()


# # Formatting our dataframe.
# 
# FlowFromDataFrame method can be used in 2 ways, either specifying the directory of the image or making sure each data in the dataframe has absolute paths to the images.

# In[ ]:


example = df_new["image_id"].iloc[:16]
example = list(example.map(lambda x: os.path.join("../input/panda-16x128x128-tiles-data/train", x)))


# # Example without combining the images together

# In[ ]:



w = 10
h = 10
fig = plt.figure(figsize=(9, 13))
columns = 4
rows = 4     

ax = []

for i in range(columns*rows):
    img = cv2.imread(example[i], cv2.COLOR_BGR2RGB)
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("tile:"+str(i)) 
    plt.imshow(img)

plt.show()  


# # Example on how to combine the image tiles into one single image.

# In[ ]:


def get_img_tiles(image_list):
    img_rows = []
    rc = 0
    for i in range(rows):
        img1 = cv2.imread(image_list[rc + 1], cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(image_list[rc + 2], cv2.COLOR_BGR2RGB)
        img3 = cv2.imread(image_list[rc + 3], cv2.COLOR_BGR2RGB)
        img4 = cv2.imread(image_list[rc + 4], cv2.COLOR_BGR2RGB)
        img_row = np.concatenate((img1, img2, img3, img4), axis = 1)
        if rc == 0:
            rc += 3
        elif rc == 3 or rc == 7:
            rc += 4
        else:
            rc += 0
        img_rows.append(img_row)
    img_stacked = img_row = np.concatenate((img_rows[0], img_rows[1], img_rows[2], img_rows[3]), axis = 0)
    return img_stacked


# In[ ]:


example_tile = get_img_tiles(example)
plt.imshow(example_tile)
plt.show()


# In[ ]:




