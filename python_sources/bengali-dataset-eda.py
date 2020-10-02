#!/usr/bin/env python
# coding: utf-8

# ### If you find this kernel usefull, Do upvote.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Train File

# In[ ]:


df_train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
print('Train Data Shape: ', df_train.shape)
df_train.head()


# ## Test File

# In[ ]:


df_test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
print('Test Data Shape: ', df_test.shape)
df_test.head()


# ## Class-Map File

# In[ ]:


class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
print('Test Data Shape: ', class_map.shape)
class_map.head()


# # Image Data
# Image Data in in parquet files and contrain grayscale images of below mentioned dimentions. If you want to read more about this file format then try: 
# https://acadgild.com/blog/parquet-file-format-hadoop
# Note that the file it self conatins values of all the 32332 pixels (137*236) in each row coresponding to a image.  
# ###### Image Height = 137
# ###### Image Width = 236

# ### Image Utils

# In[ ]:


HEIGHT = 137
WIDTH = 236

def load_npa(file):
    df = pd.read_parquet(file)
    return df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)


# In[ ]:


## loading one of the parquest file for analysis
dummy_images = load_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
print("Shape of loaded files: ", dummy_images.shape)
print("Number of images in loaded files: ", dummy_images.shape[0])
print("Shape of first loaded image: ", dummy_images[0].shape)
print("\n\nFirst image looks like:\n\n", dummy_images[0])


# ### Must say hard to tell anything by looking at this image... Better Way to look at it

# In[ ]:


## View the pixel values as image
plt.imshow(dummy_images[0], cmap='Greys')


# ##### Some More images from loaded data

# In[ ]:


f, ax = plt.subplots(5, 5, figsize=(16, 8))
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(dummy_images[i*5+j], cmap='Greys')


# ## Train Data Analysis
# ### Number of Samples: 200840
# #### File Contains:
# * image_id: the foreign key for the parquet files
# * grapheme_root: the first of the three target classes
# * vowel_diacritic: the second target class
# * consonant_diacritic: the third target class
# * grapheme: the complete character. Provided for informational purposes only, you should not need to use this.

# In[ ]:


df_train.head()


# In[ ]:


print("Unique Grapheme-Root in train data: ", df_train.grapheme_root.nunique())
print("Unique Vowel-Diacritic in train data: ", df_train.vowel_diacritic.nunique())
print("Unique Consonant-Diacritic in train data: ", df_train.consonant_diacritic.nunique())
print("Unique Grapheme (Combination of three) in train data: ", df_train.grapheme.nunique())


# In[ ]:


### Majority of Images per Grapheme count is below 180. Only 1 grapheme has 283 images in it.
images_per_grapheme = df_train.groupby('grapheme')[['image_id']].count().reset_index().reset_index()
sb.catplot(x='index', y='image_id', data=images_per_grapheme)


# In[ ]:


images_per_grapheme_root = df_train.groupby('grapheme_root')[['image_id']].count().reset_index().reset_index()
sb.catplot(x='index', y='image_id', data=images_per_grapheme_root)


# In[ ]:


images_per_grapheme_diacritic = df_train.groupby('vowel_diacritic')[['image_id']].count().reset_index()
sb.catplot(x='vowel_diacritic', y='image_id', data=images_per_grapheme_diacritic)


# In[ ]:


images_per_grapheme_diacritic = df_train.groupby('consonant_diacritic')[['image_id']].count().reset_index()
sb.catplot(x='consonant_diacritic', y='image_id', data=images_per_grapheme_diacritic)


# # Above Plots can be used to observe class imbalance at different levels

# ### Thanks for reading.
