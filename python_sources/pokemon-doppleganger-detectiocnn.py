#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# please note this package - imagededup was custom-installed in this kernel 
from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)


# ### Initialize the algorithm that you'd like to use like PHash(), CNN()

# ### PHash 
# 
# - Perceptual Hashing - https://en.wikipedia.org/wiki/Perceptual_hashing
# 
# ### WHash
# 
# - Wavelet Hashing - https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
# 
# ### CNN
# 
# - Convoluted Neural Network - MobileNet pretrained on imagenet dataset. 
# 

# In[ ]:


cnn = CNN()


# ### Define the image directory from where we have to find duplicates

# In[ ]:


image_dir = "../input/pokemon-images-and-types/images/images/"


# ### Encoding the images using the initialized Algorithm

# The last Global Average Pooling layer to generate encodings. 

# In[ ]:


encodings = cnn.encode_images(image_dir=image_dir)


# ### Finding Duplicates

# ### Duplicates

# In[ ]:


duplicates = cnn.find_duplicates(encoding_map=encodings, scores = True)


# In[ ]:


for key, value in duplicates.items():
   if len(value) > 0:
    print(key + ",")
    


# ### Plotting the duplicates

# In[ ]:


plot_duplicates(image_dir=image_dir, 
                duplicate_map=duplicates, 
                filename='cascoon.png')


# In[ ]:


plot_duplicates(image_dir=image_dir, 
                duplicate_map=duplicates, 
                filename='manaphy.png')


# In[ ]:


plot_duplicates(image_dir=image_dir, 
                duplicate_map=duplicates, 
                filename='plusle.png')


# Thanks to [idealo](https://github.com/idealo) for opensourcing this library. `imagededup` is definitely worth trying out : 
# 
# 
# **Github** - [`imagededup`](https://github.com/idealo/imagededup)

# Let me know your thoughts, feedback and upvote!

# In[ ]:




