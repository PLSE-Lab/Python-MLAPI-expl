#!/usr/bin/env python
# coding: utf-8

# ## Disclaimer: I have **NO** knowledge in art in any way.
# ## Partially Inspired By [iMet Data Analysis, Plus!](https://www.kaggle.com/hsakizero/imet-data-analysis-plus)

# # Prerequsites
# Import necessary libraries and configure custom plot style 

# In[ ]:


import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# custom styling
sns.set_style('dark')
sns.set(font_scale=1.75)

# For reproducibility purpose
def seed_everything(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Read Data
training_data = pd.read_csv('../input/train.csv')
attribute_name = pd.read_csv('../input/labels.csv')


# # Most Frequent Attrbutes
# Display top 30 most frequent attributes in training data  

# In[ ]:


# put all training labels to a list
attributes = training_data.attribute_ids.values.tolist()
# count frequency of each unique label
counter = pd.Series(' '.join(attributes).split()).value_counts().to_frame().reset_index()
# reset column name
counter.columns = ['attribute_id', 'frequency']
# cast id into type int
counter['attribute_id'] = counter['attribute_id'].astype(int)
# merge with attribute names
counter = counter.merge(attribute_name)
# plot top 30
plt.figure(figsize=(18, 12))
sns.barplot(data=counter.head(30), x='frequency', y='attribute_name');


# # How Much Do They Weight?
# For any label $k$, define its frequency as $f_k$ and weight as $w_k$:
# \begin{align} w_k = {f_k \over \sum_{i=1}^n f_i}\text{ , n = total number of classes} \end{align}
# 
# Let's take a look at how much do top 30 most frequent attributes weight in total

# In[ ]:


# sum of frequency of all labels
sum_of_frequency = counter['frequency'].values.sum()
# calculate weight of each label
counter['weight'] = counter['frequency'] / sum_of_frequency

def show_top_n(n):
    assert isinstance(n, int) and n > 0
    top_n = counter['weight'].head(n).values.sum()
    plt.figure(figsize=(7, 7))
    plt.pie(x=[top_n, 1.0 - top_n], 
            explode=[0.1, 0.0], 
            labels=['Top %s Most Frequent Attributes' %n, 'Other Attributes'], 
            autopct='%.2f%%', 
            textprops={'size':'larger'})

show_top_n(30)


# # Observation 1
# **30 Attributes Account For Over 50% of Total Attribute Counts!!**
# 
# Remember we have in total 1103 classes, this implies serious class imbalance in the training data.

# # What Do Arts About French Men Look Like?
# Let's look at some potentially interesting combination of frequent attributes.
# 
# We will start off wth `culture::french` and `tag::men`.

# In[ ]:


labels = pd.read_csv('../input/labels.csv')
french = labels.loc[labels['attribute_name'] == 'culture::french']['attribute_id'].item()
men = labels.loc[labels['attribute_name'] == 'tag::men']['attribute_id'].item()

def is_french_men(row):
    attr = row['attribute_ids']
    return len(attr.split()) == 2 and str(french) in attr and str(men) in attr

# Source: https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    fig.set_size_inches(20, 18)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    
def sample(data, n):
    result = {}
    samples = data.sample(n, random_state=2019)
    for i, info in enumerate(samples.values):
        filename, _ = info
        title = 'sample_%s' % str(i+1)
        img_path = ''.join(('../input/train/', filename,'.png'))
        result[title] = cv2.imread(img_path)
    return result

samples = sample(training_data[training_data.apply(is_french_men, axis=1)], 20)
plot_figures(samples, 4, 5)


# # What About British Women?
# Let's see if there are obvious visual differences between the two

# In[ ]:


british = labels.loc[labels['attribute_name'] == 'culture::british']['attribute_id'].item()
women = labels.loc[labels['attribute_name'] == 'tag::women']['attribute_id'].item()
def is_british_women(row):
    attr = row['attribute_ids']
    return len(attr.split()) == 2 and str(british) in attr and str(women) in attr

samples = sample(training_data[training_data.apply(is_british_women, axis=1)], 20)
plot_figures(samples, 4, 5)


# # Observation 2
# Notice how **sample_19** in both groups are very noisy. I suspect that even though these two images have completely different attributes, there is still a certain level of unnecessary similarity between the two due to all those noisy black and white pixels. **This certainly makes it harder for models to capture distinctive features.**  

# # Observation 3
# Notice many samples drawn look very sketchy, i.e. **sample_14** in group 2. I suspect that **slightly increasing contrast or strengthen edges during training may help our model understand these images better.**

# ## To Be Continued...
