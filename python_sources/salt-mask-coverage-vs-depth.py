#!/usr/bin/env python
# coding: utf-8

# I've just started to look at this dataset and even tough I work in the oil and gas industry, I just had to ask this simple question: Is there a correlation between the size of the salt mask and the depth?
# 
# Well that is really simple to answer with a few lines of code. This has probably been done already by someone else, but it such an easy thing to do, that I might as well do it.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook


# ## Read the data into a Pandas dataframe.
# Create a train_df as everyone else, containing the depths (z).

# In[ ]:


img_size_ori = 101
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)


# ## Read the masks and add them to the dataframe and calculate coverage.

# In[ ]:


train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / (img_size_ori * img_size_ori)


# ## Make a scatter plot.

# In[ ]:


plt.figure(figsize=(15,12))
sns.scatterplot(x="z", y="coverage", data=train_df)


# .... guess the correlation is pretty low. I guess the answer to the question is "no". Just for completeness, let's calculate a numeric value:

# In[ ]:


np.corrcoef(train_df.z , train_df.coverage)[0,1]

