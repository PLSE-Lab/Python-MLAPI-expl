#!/usr/bin/env python
# coding: utf-8

# # Resizing & cropping images
# 
# Multiple people have pointed out that saving the daaframes to .feather made loading times much faster. There are already multiple datasets and notebooks for this competitions and I mostly based myself on [Hanjoon Choe's very nice notebook](https://www.kaggle.com/hanjoonchoe/resize-and-load-with-feather-format-much-faster) for this one.
# 
# Their are two objectives here:
# - make loading times faster with feather
# - resizing images to reduce the size of the dataset.
# 
# On that second point however, I've seen some people cropping images or resizing the images, but what I saw so far seemd to be rather random.
# 
# In this notebook, using OpenCV2, I'm making a bounding bow around each character to crop only the character, but still have all of it. However, there is a tricky part as these characters are composed on multiple strokes that sometimes do not touch each other. This is taken into account, by simply making a bigger bounding box englobing all the different ones.
# 
# There probably are faster ways of doing this, however since this is only supposed to be done once to prepare and save the data, I went for fastest implementation rather than run time.
# 
# I hope you might find this notebook and it's outputs useful!
# 
# 
# **The dataset can be found [here](https://www.kaggle.com/maxlenormand/cropped-resized-bengaliai-images)**

# In[ ]:


import numpy as np
import pandas as pd
import cv2

import time
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


start_time = time.time()
df_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
current_time = time.time()
print(f"Shape: {df_0.shape} (took {time.time() - start_time}sec to load)")

df_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')
current_time = time.time()
print(f"Shape: {df_1.shape} (took {time.time() - current_time}sec to load)")

df_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')
current_time = time.time()
print(f"Shape: {df_2.shape} (took {time.time() - current_time}sec to load)")

df_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

print(f"It took: {time.time() - start_time} to load all 4 datasets")


# In[ ]:


HEIGHT = 137
WIDTH = 236

CROP_SIZE = 100


# In[ ]:


original_img_size = HEIGHT * WIDTH

cropped_img_size = CROP_SIZE * CROP_SIZE

print(f"Original shape of images: {original_img_size}\nCropped & resized shape of images: {cropped_img_size}")
print(f"Reduction fatio: {np.round(original_img_size/cropped_img_size, 3)}")


# We reduced the image size by **3.23** times.
# 
# That's not that bad. Especially if, like me, you don't have access to any good GPUs. Reducing the size efficiently while keeping the most information is particularly important and interesting.

# In[ ]:


def crop_and_resize_images(df, resized_df, resize_size = CROP_SIZE):
    cropped_imgs = {}
    for img_id in tqdm(range(df.shape[0])):
        img = resized_df[img_id]
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        idx = 0 
        ls_xmin = []
        ls_ymin = []
        ls_xmax = []
        ls_ymax = []
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)

        roi = img[ymin:ymax,xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size))
        cropped_imgs[df.image_id[img_id]] = resized_roi.reshape(-1)
        
    resized = pd.DataFrame(cropped_imgs).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized #out_df


# In[ ]:


resized = df_0.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)


# In[ ]:


cropped_df = crop_and_resize_images(df_0, resized, CROP_SIZE)


# We get a dataframe just like the original ones, but with a CROP_SIZE * CROP_SIZE number of columns + the image_id column. This significantly reduced the image size

# In[ ]:


cropped_df.head()


# In[ ]:


cropped_df.to_feather("train_data_0.feather")


# We can now save it to feather for later

# In[ ]:


sample_df = pd.read_feather("train_data_0.feather")


# In[ ]:


resized_sample = sample_df.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)


# If we simply takes the bounding box, we would end up with multiple different ones for each independant symbol in the image. This isn't what we want. To show that this notebook takes this into account, here is one resized & cropped example with multiple independant symbols in it:

# In[ ]:


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 15))
ax0.imshow(resized[329], cmap='Greys')
ax0.set_title('Original image')
ax1.imshow(resized_sample[329], cmap='Greys')
ax1.set_title('Resized & cropped image')
plt.show()


# Now we can do all three other datasets

# In[ ]:


del resized
del cropped_df


# In[ ]:


start = time.time()

# dataset 1
resized_1 = df_1.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
cropped_df_1 = crop_and_resize_images(df_1, resized_1, CROP_SIZE)
cropped_df_1.to_feather("train_data_1.feather")
del resized_1
del cropped_df_1
print(f"Saved cropped & resized df_1 to feather in {time.time() - start}sec")
current = time.time()

# dataset 2
resized_2 = df_2.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
cropped_df_2 = crop_and_resize_images(df_2, resized_2, CROP_SIZE)
cropped_df_2.to_feather("train_data_2.feather")
del resized_2
del cropped_df_2
print(f"Saved cropped & resized df_2 to feather in {time.time() - current}sec")
current = time.time()

# dataset 3
resized_3 = df_3.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
cropped_df_3 = crop_and_resize_images(df_3, resized_3, CROP_SIZE)
cropped_df_3.to_feather("train_data_3.feather")
del resized_3
del cropped_df_3
print(f"Saved cropped & resized df_3 to feather in {time.time() - current}sec")


# In[ ]:




