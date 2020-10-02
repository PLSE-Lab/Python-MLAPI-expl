#!/usr/bin/env python
# coding: utf-8

# # From the great work of Tilii and Brian
# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534
# 
# ### This kernel unite and check their findings
# ### It produces an unique list of duplicates from their lists

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
sns.set()
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data_train_dir = r'../input/human-protein-atlas-image-classification/train'
answer_file_path = r'../input/human-protein-atlas-image-classification/train.csv'

channels = ['_yellow', '_red', '_green', '_blue']


# # Functions for visualization

# In[ ]:


def make_rgb_image_from_four_channels(channels: list, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels, 
    where yellow image will be yellow color, red will be red and so on  
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(channels[0]))
    # yellow is not added as red + bleu
    rgb_image[:, :, 0] += yellow/2   
    rgb_image[:, :, 2] += yellow/2
    # loop for R,G and B channels
    for index, channel in enumerate(channels[1:]):
        current_image = Image.open(channel)
        rgb_image[:, :, index] += current_image
    
    rgb_image = np.clip(rgb_image,0,255)
    return rgb_image.astype(np.uint8)


# In[ ]:


df_duplicates = pd.read_csv("../input/duplicatesproposal/duplicates-personnal.csv")
df_duplicates


# # Remove duplicates in duplicate list 

# In[ ]:


one_time_list = []
df_duplicate_uniq = pd.DataFrame(columns=['Keep','Remove'])
for i in range(len(df_duplicates)):
    if not ( df_duplicates.iloc[i,0] in one_time_list or df_duplicates.iloc[i,1] in one_time_list):
        one_time_list = one_time_list + [df_duplicates.iloc[i,0]] + [df_duplicates.iloc[i,1]]
        df_duplicate_uniq = df_duplicate_uniq.append({'Keep':df_duplicates.iloc[i,0],'Remove':df_duplicates.iloc[i,1]}, ignore_index=True)
        
# df_duplicate_uniq.iloc[99,0] and [129,0] are malformed image
df_duplicate_uniq.iloc[99,0], df_duplicate_uniq.iloc[99,1] = df_duplicate_uniq.iloc[99,1], df_duplicate_uniq.iloc[99,0] 
df_duplicate_uniq.iloc[129,0], df_duplicate_uniq.iloc[129,1] = df_duplicate_uniq.iloc[129,1], df_duplicate_uniq.iloc[129,0] 

print(df_duplicate_uniq.head())
print("len(df_duplicates)",len(df_duplicates))
print("len(df_duplicate_uniq)",len(df_duplicate_uniq))


# # Displaying

# In[ ]:


rg = np.arange(95,115,1)
for i in rg:
    current_line = df_duplicate_uniq.loc[i]
    image_names_1 = [os.path.join(data_train_dir, current_line[0]) 
                   + x + '.png' for x in channels]
    image_names_2 = [os.path.join(data_train_dir, current_line[1]) 
                   + x + '.png' for x in channels]
    rgb_image_1 = make_rgb_image_from_four_channels(image_names_1)
    rgb_image_2 = make_rgb_image_from_four_channels(image_names_2)
    fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(29,29))
    ax[0].imshow(rgb_image_1)
    ax[1].imshow(rgb_image_2)


# # Save

# In[ ]:


df_duplicate_uniq.to_csv("duplicates_final_list.csv")

