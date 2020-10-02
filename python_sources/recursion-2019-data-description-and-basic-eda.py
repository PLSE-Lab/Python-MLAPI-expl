#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# **Welcome to Recursion Cellular Image Classification competition**.
# <br>This kernel is a basic idea about the data and we will do a basic EDA.

# ## Loading the metadata with Description of metadata

# In[ ]:


BASE_DIR = '../input'


# In[ ]:


os.listdir(BASE_DIR)


# So, we basically have train.csv and test.csv files which contains the major details about our image data, other than these, we also have train_controls.csv, test_controls.csv, pixel_stats.csv, sample_submission.csv files and train and test folder which contains our image data.

# ### Train metadata

# Let's start by reading train.csv:

# In[ ]:


df_train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
print(df_train.sample(5))
print("*"*40)
print(df_train.info())


# So, we can see that in our training data, there are five columns in it, among which "sirna" will be our label which we will need to predict. So we will talk about "sirna" later. For now, we can go to the info section of our data, according to it, our training metadata contains 36515 different rows and there are no missing values, due to which we can consider our data to be already cleaned.

# Also, from the dataset info available at [RxRx.ai](https://www.rxrx.ai/#the-data),
# * Total of 308 wells on which the experiments are performed(277 wells in our training data)
# * Each having 4 plates(namely 1, 2, 3 and 4)
# * Total of 51 experiments(33 experiments in our training data)
# * Each sample having a unique id_code. 
# * Also, each well has two different sites both marked with same sirna, details about which can be found in the pixel data.
# 
# All above points can be seen easily from the following code cell. 

# In[ ]:


df_train.nunique()


# Now, our target value, i.e. "sirna" has 1108 different values, that basically means that we will need to classify our images into any of the 1108 classes. More details about our target value can be found in the above mentioned link.

# ### Test metadata

# Let's now come to our test data:

# In[ ]:


df_test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
print(df_test.sample(5))
print("*"*40)
print(df_test.info())


# Our test dataset is defined in the similar fashion excepting the "sirna" column as expected. Also this dataset too do not contain any missing values.

# ### Train/Test controls metadata

# Let's talk a bit about train_controls and test_controls dataset:

# In[ ]:


df_train_controls = pd.read_csv(os.path.join(BASE_DIR, 'train_controls.csv'))
print(df_train_controls.sample(5))
print("*"*40)
print(df_train_controls.info())
print("*"*80)
df_test_controls = pd.read_csv(os.path.join(BASE_DIR, 'test_controls.csv'))
print(df_test_controls.sample(5))
print("*"*40)
print(df_test_controls.info())


# So, according to the data description given with the problem, in each experiment, the same 30 siRNAs appear on every plate as positive controls. In addition, there is at least one well per plate with untreated cells as a negative control. It has the same schema as [train/test].csv, plus a well_type field denoting the type of control. In our [train/test]_controls.csv file, we can see that there are two unique well_types given as can be seen from below code cell:

# In[ ]:


print(df_train_controls["well_type"].unique())
print("*"*40)
print(df_test_controls["well_type"].unique())


# ### Pixel metadata

# We are provided with statistics on all the images in the pixel_stats.csv file. This information will allow us to normalize the data, for example by reducing the mean and dividing by the standard deviation.

# In[ ]:


df_pixel_stats = pd.read_csv(os.path.join(BASE_DIR, 'pixel_stats.csv'))
df_pixel_stats_updated = df_pixel_stats.set_index(['id_code','site', 'channel'])
print(df_pixel_stats.sample(5))
print("*"*40)
print(df_pixel_stats_updated.sample(5))
print("*"*40)
print(df_pixel_stats.info())


# ## Basic EDA

# #### Distribution of "plate" attribute among the train dataset

# In[ ]:


df_train["plate"].value_counts().plot(kind = "bar")


# * So, this shows us that each experiment has a equal number of different plates and dataset is stable.

# #### Distribution of "experiment" attribute among train dataset

# In[ ]:


plt.rcParams['figure.figsize'] = [15, 5]
df_train["experiment"].value_counts().plot(kind = "bar")


# * Similarly, the number of samples from each experiment is also very stable.

# #### Distribution of "well_type" among train/test control dataset

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
print(df_train_controls["well_type"].value_counts())
df_train_controls["well_type"].value_counts().plot(kind = "bar")


# * So as already told in the metadata description, the same 30 siRNAs appear on every plate as positive controls. In addition, there is at least one well per plate with untreated cells as a negative control. That's why there is a high number of positive_controls in comparison to negative_controls in our train_controls.csv metadata.
# * Similar fashion can be seen in test_controls.csv metadata as follows:

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
print(df_test_controls["well_type"].value_counts())
df_test_controls["well_type"].value_counts().plot(kind = "bar")


# #### Distribution of "site" among pixel stats dataset

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
print(df_pixel_stats["site"].value_counts())
df_pixel_stats["site"].value_counts().plot( kind = "bar")


# * So in above figure, we can see that an experiment is performed on two different sites(namely 1 and 2) in a same well.

# #### Distribution of "channel" among pixel stats dataset

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 8]
print(df_pixel_stats["channel"].value_counts())
df_pixel_stats["channel"].value_counts().plot(kind = "pie")


# * Above figure here shows that every image is having 6 different channels, actually all the images are taken with adjusting the wavelength of the apparatus and then can be combined if one wants to visualise them as a single image.

# ### Visualising image data

# * At the very last of this notebook, we would like to conclude this very basic EDA with a visualisation of the image data provided to us.

# In[ ]:


path = []
for index in range(6):
    path.append(os.path.join("../input/train/HUVEC-01/Plate1", os.listdir("../input/train/HUVEC-01/Plate1")[index]))


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(24, 16))
for index, ax in enumerate(axes.flatten()):
    img = plt.imread(path[index])
    ax.axis('off')
    ax.set_title(os.listdir("../input/train/HUVEC-01/Plate1")[index])
    _ = ax.imshow(img, cmap='gray')


# ### Thanks!
