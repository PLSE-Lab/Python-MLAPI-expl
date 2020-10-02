#!/usr/bin/env python
# coding: utf-8

# ### The aim of this project is to build a multi-class classifier for 10 Impressionist painters, namely:
# * Camille Pisarro
# * Childe Hassam
# * Claude Monet
# * Edgar Degas
# * Henri Matisse
# * John Singer-Sargent
# * Paul Cezanne
# * Paul Gauguin
# * Pierre-Auguste Renoir
# * Vincent van Gogh
# 
# ### This notebook plots sample images for each painter from the dataset we have previously created. 

# In[ ]:


import os
import random

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Root directory

root = '/kaggle/input/impressionist-classifier-data/'


# In[ ]:


# Creates artist-specific directory names, and files storing the image names for each artist

artist_lst = ['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 
              'Monet', 'Pissarro', 'Renoir', 'Sargent', 'VanGogh']

for artist in artist_lst:
    exec(f"train_{artist}_dir = os.path.join(root, 'training/training', '{artist}')")
    exec(f"train_{artist}_filenames = os.listdir(train_{artist}_dir)")
    exec(f"valid_{artist}_dir = os.path.join(root, 'validation/validation', '{artist}')")
    exec(f"valid_{artist}_filenames = os.listdir(valid_{artist}_dir)")


# In[ ]:


def plot_imgs(artist, nrows=4, ncols=4, num_imgs=8):
    """
    Function to plot random sample images for each artist in a num_rows x num_cols grid
    :param artist: Artist name
    :type artist: str
    :param nrows: Number of rows in grid
    :type nrows: int
    :param ncols: Number of columns
    :type ncols: int
    :param num_imgs: Number of sample images to plot
    :type num_imgs: int
    :return: None
    """
    
    pic_idx = 0
    
    fig = plt.gcf()
    fig.set_size_inches(ncols * 6, nrows * 6)

    pic_idx += num_imgs

    train_dir = eval(f"train_{artist}_dir")
    filenames = eval(f"train_{artist}_filenames")
    filenames = random.sample(filenames, len(filenames))
    
    next_pix = [os.path.join(train_dir, fname) 
                    for fname in filenames[pic_idx-num_imgs: pic_idx]]


    for i, img_path in enumerate(next_pix):
        plt.suptitle(f"{artist}", fontsize=24)
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)


# ## Plot 8 sample images for each painter

# ### 1. Paul Cezanne

# In[ ]:


plot_imgs(artist_lst[0])


# ### 2. Edgar Degas

# In[ ]:


plot_imgs(artist_lst[1])


# ### 3. Paul Gauguin

# In[ ]:


plot_imgs(artist_lst[2])


# ### 4. Childe Hassam

# In[ ]:


plot_imgs(artist_lst[3])


# ### 5. Henri Matisse

# In[ ]:


plot_imgs(artist_lst[4])


# ### 6. Claude Monet

# In[ ]:


plot_imgs(artist_lst[5])


# ### 7. Camille Pissarro

# In[ ]:


plot_imgs(artist_lst[6])


# ### 8. Pierre-Auguste Renoir

# In[ ]:


plot_imgs(artist_lst[7])


# ### 9. John Singer-Sargent

# In[ ]:


plot_imgs(artist_lst[8])


# ### 10. Vincent van Gogh

# In[ ]:


plot_imgs(artist_lst[9])


# In[ ]:




