#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Reading input data 
# In the first step, we need to read our data from the input the directory (`/kaggle/input/Kannada-MNIST/`).  
# We are interested in two files:
# * `train.csv` containing samples for training model,
# * `test.csv` containing records that need to be classified in this competition.

# In[ ]:


df_test = pd.read_csv(os.path.join('/kaggle/input/Kannada-MNIST/test.csv'))
df_test.sample(n=5)


# In[ ]:


df_train = pd.read_csv(os.path.join('/kaggle/input/Kannada-MNIST/train.csv'))
df_train.sample(n=5)


# We can see that each row in `train.csv` file consists of a `label` and 784 (28x28) pixels. Pixel values are our training data (called `X`) and labels are our target (classification result, called `y`). In `test.csv` on the other hand, `label` column is missing (because that's what we need to find) but we've got `id` column which is just row index - not really useful so we will get rid of that. Let's now split dataframes into X and y arrays for both training and test sets.

# In[ ]:


X_train = np.array(df_train.loc[:, df_train.columns != 'label'])
X_test  = np.array(df_test.loc[:, df_test.columns != 'id'])

y_train = np.array(df_train['label'])

print(f"X_train: {X_train.shape}\nX_test: {X_test.shape}\ny_train: {y_train.shape}")


# ## Visualization
# 
# #### Sample images
# 
# We start with displaying sample images for each of ten labels which is quite simple task but it's good to know what data we are dealing with.

# In[ ]:


rows = 5
cols = 10
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))

for label in range(cols):
    digits = df_train.loc[df_train['label'] == label]
    digits = digits.drop('label', axis=1)
    ax[0][label].set_title(label)
    for j in range(rows):
        ax[j][label].axis('off')
        ax[j][label].imshow(digits.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')


# #### Distribution
# 
# Let's display number of samples representing each label (0-9). You can clearly see that in the training set, every label has 6000 samples. It means we don't have to do anything specific to reduce data imbalance which is a good sign.

# In[ ]:


sns.distplot(y_train, kde=False)


# #### Pixel heatmap
# 
# The next (actually interesting) thing is a pixel heatmap. It shows which pixels are usually used in our data images i.e. aren't black pixels. If you take a look at the image below, you can see that approx. 2-3 pixels from each side of images (and a bit more in the top-left corner) are never used in the training set.

# In[ ]:


pixel_counts = (df_train.loc[:, df_train.columns != 'label'] / 255).astype(int)
pixel_counts = pixel_counts.sum(axis=0).values
pixel_counts = pixel_counts.reshape((28, 28))
sns.heatmap(pixel_counts)


# We can also plot heatmaps for each label separately, as below.

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 5))
ax = ax.flatten()

for label in range(10):
    pixel_counts = (df_train.loc[:, df_train.columns != 'label'] / 255).astype(int)
    pixel_counts = pixel_counts.loc[df_train['label'] == label]
    pixel_counts = pixel_counts.sum(axis=0).values
    pixel_counts = pixel_counts.reshape((28, 28))
    ax[label].axis('off')
    sns.heatmap(pixel_counts, ax=ax[label])


# ## Further steps
# 
# When you already know what kind of data you are dealing with, you can start creating and training your model. You can refer to many existing tutorials and kernels regarding MNIST dataset which is also available on Kaggle as [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition. There you can find exhausting list of visualizations, models and general advices.
