#!/usr/bin/env python
# coding: utf-8

# **<h1>TGS - Reading data and storing in HF5</h1>
# 
# Having to read from the file systems all the images and masks of this data set every time you run a notebook is very inefficient. For this reason I've created an notebook that does all the loading and some preprocessing of this information into a Pandas data frame and stores it in the more convenient format HF5, much more efficient (it just takes a few seconds to load all the data in this format).
# 
# I'm sure all experienced kagglers have come up with some form of efficiently loading competition data, and I'd be glad to have some comments of better ways to do this than the one that I'm proposing, but I hope this suggestions might be useful for more inexperienced kaggles.

# In[ ]:


# Just the needed imports for the task at hand
# tpqdm_notebook will allow us to see the progress of image loading, which is the slowest one

import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook


# In[ ]:


# As in your computer data might be in a different place than in the Kaggle platform, defining this variable and appending it
# to all paths is a convenient way to use (almost) the same notebooks in Kaggle and your platform
DATA_DIR = '../input/'

# Load the train and depths csvs and join 
train_df = pd.read_csv(DATA_DIR + 'train.csv', index_col='id')
depths_df = pd.read_csv(DATA_DIR + 'depths.csv', index_col='id')
train_df = train_df.join(depths_df)

# A test.csv doesn't exist as such, so we use the submission information and join it with the depths information
submission_df = pd.read_csv(DATA_DIR + 'sample_submission.csv', index_col="id")
test_df = submission_df.copy()
test_df = test_df.join(depths_df)


# In[ ]:


print("**** Train ****")
print(train_df.head())
print()
print('**** Depths ****')
print(depths_df.head())
print()
print("**** Test ****")
print(test_df.head())
print()
print("**** Sample submission ****")
print(submission_df.head())
print()


# <h2>Load images and masks</h2>

# In[ ]:


train_df["images"] = [np.array(load_img(DATA_DIR + "train/images/{}.png".format(idx), grayscale=True), dtype=np.int16) for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img(DATA_DIR + "train/masks/{}.png".format(idx), grayscale=True), dtype=np.int16) for idx in tqdm_notebook(train_df.index)]
test_df['images'] = [np.array(load_img(DATA_DIR + "test/images/{}.png".format(idx), grayscale=True), dtype=np.int16) for idx in tqdm_notebook(test_df.index)]


# In[ ]:


test_df.head()


# In[ ]:


# Adjust the type of the z (depth) feature to make the objects smaller
train_df['z'] = train_df['z'].astype(np.uint16)
test_df['z'] = test_df['z'].astype(np.uint16)

train_df.info()
test_df.info()


# In[ ]:


# Add a boolean feature to indicate if the image has or not salt at all
train_df['has_salt'] = train_df['masks'].apply(lambda x: x.sum() > 0)
train_df.head()


# <h2>Save the data</h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Now let's save all the data in the same HF5 file\ntrain_df.to_hdf('tgs_salt.h5', key='train', mode='w')\ntest_df.to_hdf('tgs_salt.h5', key='test', mode='a')\nsubmission_df.to_hdf('tgs_salt.h5', key='submission', mode='a')")

