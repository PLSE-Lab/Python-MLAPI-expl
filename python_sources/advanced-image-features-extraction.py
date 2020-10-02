#!/usr/bin/env python
# coding: utf-8

# Hi All,
# 
# I am a new Kaggler. I just joined Kaggle about 2 months ago. Now I am extremely interesting to earn my first Kaggle medal in this very first competition of mine.
# 
# However, as my study deadline is now approaching too fast, I hardly have time to dancing more with ensemble model. Moreover, I only have a poor laptop so for NN or LGBM stuff it's too arduous for me. Hence I decide to write a code routine to extract all advanced image features, mostly based on these two papers: 
# 1) http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.648.4682&rep=rep1&type=pdf
# 2) https://storage.googleapis.com/kaggle-forum-message-attachments/328059/9411/dimitri-clickadvert.pdf
# 
# Some aspects of my code:
# - Extract all features for each image at once, rather than extract each feature for all images at once. This aspect will save a lot of time for unnecessarily reading the input images multiple times.
# - Can specify image indices to extract (i.e., 100k-200k, 1M1-1M5,...). This will help us to divide the task to several computers.
# - Can handle and ignore unexpected errors, leave it as blank. The extracted results for NaN images are also blanked.
# - Speed (i7 CPU): 10k images/hour. This is the most frustrating. However I think 3-4 laptops might help to completely process ~2M train and test images in 2-3 days.
# - Output indicator for ongoing process, i.e., "Processed: 5124...", this to keep track of the task.
# - Periodically saved the results to disk for after a specified range, i.e., after each 1000 images, or 10000 images...
# - Can remove some features to speed up, if those features were already extracted.
# 
# I believe a lot of Kagglers here already have extracted their advanced image features, but if some of you have interests in this, please leave me a comment or personal message. At this stage I already extracted 100k training images' features. I will help you to do this work also.
# 
# I need to be a member of a team who needs advanced image features to strengthen their model. Per my readings and experiment  on the Avito images, I strongly believe these advanced image features could improve generalisation. My humble experiment show that, for 5000 images, the vadiation performance increases 0.003 (from 0.225 to 0.222) using exactly the same LGBM settings. Although 5000 images are not probabilistic enough, I still think these features have great potential.

# In[13]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.

# Read the features of the first 5000 images in the training pool
df = pd.read_csv("../input/img-features-avito-first5000train/img_features.csv", low_memory=False)
df.head(20)


# There are 26 different image features, listed as below:
# 
# 1) [wh]: the sum of width and height (width and height are equivalent, no need to split them).
# 
# 2-4) [bri_avg, bri_std, bri_min]: average, standard deviation and minimum of Brightness in HSV color space.
# 
# 5-6) [sat_avg, sat_std]: average and standard deviation of Saturation in HSV color space. Minimum of saturation seems not working as minimum of Brightness (which is reported in the mentioned paper).
# 
# 7-9) [lum_avg, lum_std, lum_min]: average, standard deviation and minimum of Luminance in YUV color space.
# 
# 10) [apw]: average pixel width, get from https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality.
# 
# 11) [blur]: Blurness,  get from https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality.
# 
# 12) [colorful]: Colorfulness, implemented based on the paper.
# 
# 13-15) [dom_rgb_color, dom_rgb_color_ratio, sim_rgb]: dominant color (categorical), its ratio (in terms of pixels count) over the number of image pixels, and Simplicity in RGB color space. The original image pixels are binned into 8 bins for each channel, resulting in 512 bins for the RGB histogram to compute these features.
# 
# 16-18) [dom_hsv_color, dom_hsv_color_ratio, sim_hsv]: dominant color (categorical), its ratio (in terms of pixels count) over the number of image pixels, and Simplicity in HSV color space. The original image pixels are binned into 8 bins for each channel, resulting in 512 bins for the HSV histogram to compute these features.
# 
# 19-21) [dom_gray, sim_gray, std_gray]:  dominant color (categorical), its ratio (in terms of pixels count) over the number of image pixels, and Simplicity in grayscale. The single channel grayscaled image is kept original (256 bins).
# 
# 22) [no_kp]: number of key points (mentioned in paper). This is a great feature which has strong correlation to deal probabilities. Images with less key points tend to have higher deal probabilities.
# 
# 23) [obj_ratio]: object ratio (the ratio of pixel counts of the main object detected in the image). This is implemented using saliency map mentioned in the paper.
# 
# 24) [no_faces]: the number of human faces in the image.
# 
# 25-26) [img_class, img_probability]: image class (categorical) and its probability classified by InceptionV3 model.
# 
# So much thanks if you don't feel uncomfortable for this kernel. 
# I will edit this kernel to display some results. Please keep updated.
# 
# However, I am still processing gradually all images. I will offer the full image features set when complete (though it may cost at least 1 week by only my own).

# 

# In[ ]:




