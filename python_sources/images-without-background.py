#!/usr/bin/env python
# coding: utf-8

# I created dataset, wich images from train and test (not all) without background. I think it can be used to more clearly extract features from images.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


im1 = mpimg.imread("../input/petfinder-adoption-prediction/test_images/000c21f80-1.jpg")
plt.imshow(im1)


# In[ ]:


im2 = mpimg.imread("../input/petfinder-images-no-backg/petfinder_images_no_backg/Test/000c21f80-1.jpg")
plt.imshow(im2)

