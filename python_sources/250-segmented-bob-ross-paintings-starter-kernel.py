#!/usr/bin/env python
# coding: utf-8

# This dataset consists of ~250 Bob Ross images, their segmentation maps, and a label key. First we'll take a look at the images:

# ## Data exploration

# In[ ]:


import os
os.listdir('../input/segmented-bob-ross-images/train/images/')[:5]


# In[ ]:


from PIL import Image


# In[ ]:


imgs_dir = '../input/segmented-bob-ross-images/train/images/'
Image.open(imgs_dir + 'painting53.png')


# In[ ]:


Image.open(imgs_dir + 'painting224.png')


# Each image is available in the `/train/images/` folder under a name like `paintingN.png`, where `N` is the painting's number. Paintings are organized in the order in which they appeared on Bob Ross's "The Joy of Painting". E.g. `painting220.png` is the 220th painting he created in the series.
# 
# Here's an interesting bit of trivia for you: Bob Ross actually painted three versions of each painting that appeared on the show. The first is a preliminary version that we painted while preparing for the episode. The second is the version he paints onscreen. The third is a more detailed version that he painted for inclusion in his art book. Bob Ross painted a total of. Many of these paintings were given away.
# 
# Only a subset of ~250 of the ~400 paintings created on the show are included in this dataset. The paintings included are those which have a square or rectangular frame (Bob Ross sometimes used other types of frames, including once painting inside of an outline of the state of Florida) and which don't have any man-made features.

# In[ ]:


len(os.listdir('../input/segmented-bob-ross-images/train/images/'))


# To download all of the original paintings (without segmentation maps) see the [jwilber/Bob_Ross_Paintings](https://github.com/jwilber/Bob_Ross_Paintings) repo on GitHub.

# In[ ]:


Image.open(imgs_dir + 'painting336.png')


# As you can see in the case above, the images were created using a screen capture of some kind, and are of sometimes uneven quality. Some images have cropped-out watermarks on them or appear to be zoomed-in versions of the originals.
# 
# The segmentation maps are in the `labels/` folder.

# In[ ]:


import numpy as np
segmaps_dir = '../input/segmented-bob-ross-images/train/labels/'
segmap = np.array(Image.open(segmaps_dir + 'painting224.png'))
segmap


# In[ ]:


np.unique(segmap)


# The label key is available in the `labels.csv` file.

# In[ ]:


import pandas as pd
pd.read_csv(
    "../input/segmented-bob-ross-images/labels.csv", header=None,
    names=['Class']
)


# The class labels used are borrowed from the [ADE20K ontology](https://groups.csail.mit.edu/vision/datasets/ADE20K/). The list of labels is kept very simple, with just nine classes total, because the dataset is very small, with just 250 images.

# In[ ]:


np.array(Image.open(imgs_dir + 'painting224.png')).shape


# In[ ]:


np.array(Image.open(imgs_dir + 'painting20.png')).shape


# All of the images have been resized so that they (and the segmentation maps) are the same `337x450px` size.
# 
# Some example segmentation maps follow:

# In[ ]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting224.png')))
axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting224.png')))


# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting20.png')))
axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting20.png')))


# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting10.png')))
axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting10.png')))


# Unfortunately one segmentation map ended up on the wrong filepath for unclear reasons:

# In[ ]:


get_ipython().run_line_magic('ls', '../input/segmented-bob-ross-images/')


# In[ ]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting74.png')))
axarr[1].imshow(np.array(Image.open('../input/segmented-bob-ross-images/painting74.png')))


# ## Modeling
# 
# This data was used to train GauGAN, an image-to-image GAN model, that is able to take segmentation maps as input and produces "Bob Ross like" paintings as output. See [the forthcoming article describing the project](https://medium.com/@aleksey.bilogur/building-a-guagan-app-on-spell-article-d206f1c3c5f3) for more details on this model and how it works.
# 
# ## Your challenge
# 
# Your challenge, should you choose to accept it: can you train a machine learning model on this data that learns to output paintings in Bob Ross's distinct style?
