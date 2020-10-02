#!/usr/bin/env python
# coding: utf-8

# # 1. Load libraries and data

# In[ ]:


# Import libraries
import random
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()


# In[ ]:


PATH_to_images = '../input/siim-isic-melanoma-classification/jpeg/'
PATH_to_dataframes = '../input/siim-isic-melanoma-classification/'


# In[ ]:


# Import data
train = pd.read_csv(PATH_to_dataframes + "train.csv")


# # 2. Clean data

# In[ ]:


train.head()


# In[ ]:


print(f'{train.shape[0]} observations, {train.shape[1]} columns')


# In[ ]:


# Missing values per column
train.isna().sum()


# In[ ]:


# Drop missing values
train = train.dropna()


# # 3. EDA

# In[ ]:


# Display proportion of benign and malignant melanomas
train.benign_malignant.value_counts(normalize = True)


# Facing a imbalanced classification problem.

# In[ ]:


fig = px.histogram(train, x="benign_malignant",
                   hover_data=train.columns)
fig.update_layout(title_text='Count of benign/malignant')
fig.show()


# In[ ]:


fig = px.histogram(train, x="anatom_site_general_challenge",
                   hover_data=train.columns)
fig.update_layout(title_text='Anatom sites')
fig.show()


# In[ ]:


# Categorizing age
train['age_cat'] = '0 / 20 years'
train.loc[(train['age_approx'] > 20) & (train['age_approx'] <= 40), 'age_cat'] = '20 / 40 years'
train.loc[(train['age_approx'] > 40) & (train['age_approx'] <= 60), 'age_cat'] = '40 / 60 years'
train.loc[(train['age_approx'] > 60) & (train['age_approx'] <= 80), 'age_cat'] = '60 / 80 years'
train.loc[(train['age_approx'] > 80), 'age_approx'] = '80+ years'


# In[ ]:


# Separate minority and majority class
train_no_target = train[train['target']==0]
train_target = train[train['target']==1]


# In[ ]:


def display_image(df):
    
    random_sampling = [random.randint(0, len(df)) for i in range(9)]
    image_indexes = [list(df.index)[random_sampling[i]] for i in range(len(random_sampling))]
    
    i = 0
    
    # plot first few images
    plt.figure(figsize=(12,12))
    for index in image_indexes:
        
        # Get corresponding label
        image_name = df.loc[index, 'image_name']
        site = df.loc[index, "anatom_site_general_challenge"]
        target = df.loc[index, "target"]        
        
        # define subplot
        plt.subplot(330 + 1 + i)
        plt.title('Target: %s \n'%target+                  'Site: %s\n'%site,
                  fontsize=18)
        
        # plot raw pixel data
        numpy_image = cv2.imread(PATH_to_images + "train/" + image_name + ".jpg")
        plt.imshow(numpy_image)
        i+=1
        
    plt.subplots_adjust(bottom = 0.001)  # the bottom of the subplots of the figure
    plt.subplots_adjust(top = 0.99)
    # show the figure
    plt.show()


# In[ ]:


display_image(train_no_target)


# In[ ]:


display_image(train_target)


# I would like to represent the target through all columns in the dataframe. To do so, I will use a parallel category diagram.

# In[ ]:


train_parallel = train[['image_name', 'patient_id', 'sex', 'age_cat', 'anatom_site_general_challenge', 'diagnosis', 'target']]

fig = px.parallel_categories(train_parallel, color="target", color_continuous_scale=px.colors.sequential.algae)
fig.update_layout(title='Parallel category diagram on trainset')
fig.show()


# The problem here is that it is barely readable as our data are imbalanced. I will perform downsampling on the data and check the changes.

# In[ ]:


train.target.value_counts()


# In[ ]:


# Downsampling majority class
df_majority_downsampled = resample(train_no_target, 
                                   replace=False, # sample without replacement
                                   n_samples=584, # to match minority class
                                   random_state=42)
 
# Combine minority class with downsampled majority class
train_downsampled = pd.concat([df_majority_downsampled, train_target])


# In[ ]:


train_parallel_downsampled = train_downsampled[['image_name', 'patient_id', 'sex', 'age_cat', 'anatom_site_general_challenge', 'diagnosis', 'target']]

fig = px.parallel_categories(train_parallel_downsampled, color="target", color_continuous_scale=px.colors.sequential.algae)
fig.update_layout(title='Parallel category diagram on downsampled trainset')
fig.show()


# This visualization is also quite useful to check undesirable categories in categorical columns that don't count as missing value. Here I can see that I didn't clean the sex column.

# **<font color="red" size="4">Thank you for taking the time to read this notebook. I hope that I was able to answer your questions or your curiosity and that it was quite understandable. If you liked this text, <u>please upvote it</u>. I will really appreciate and this will motivate me to make more and better content !</font>**
