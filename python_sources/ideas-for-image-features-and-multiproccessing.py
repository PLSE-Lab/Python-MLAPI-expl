#!/usr/bin/env python
# coding: utf-8

# 
# # Ideas for Generating Image Features and Measuring Image Quality - with Multi-Processing and Data Partitioning
# 
# <br>
# 
# ![](https://i.imgur.com/84TEdoa.png)
# 
# <br>
# 
# [Avito](https://www.kaggle.com/c/avito-demand-prediction) is Russia's largest Advertisment firm. The quality of the advertisement image significantly affects the demand volume on an item. For both advertisers and Avito, it is important to use authentic high quality images. In this kernel, I have implemented some ideas which can be used to create new features related to images. These features are an indicatory factors about the Image Quality. Following is the list of feature ideas:  
# 
# 
# ### 1. Dullness : Is the Image Very Dull ?   
#     
#    1.1 Image Dullness Score
#   
# 
# <br>
# 

# In[33]:


from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

# add multiprocessing library
import multiprocessing
from multiprocessing import Pool

from IPython.core.display import HTML 
from IPython.display import Image

images_path = '../input/sampleavitoimages/sample_avito_images/'
imgs = os.listdir(images_path)

features = pd.DataFrame()
features['image'] = imgs


# In[34]:


features.head(5)


# In[37]:


num_partitions = 2
num_cores = multiprocessing.cpu_count()

def parallelize_dataframe(df, func):
    a,b = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, [a,b]))
    pool.close()
    pool.join()
    return df


# ## 1. Is the image Very Dull 
# 
# ### Feature 1 : Dullness
# 
# Dull Images may not be good for the advirtisment purposes. The analysis of prominent colors present in the images can indicate a lot about if the image is dull or not. In the following cell, I have added a code to measure the dullness score of the image which can be used as one of the feature in the model. 
# 
# 

# In[28]:


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent


# Lets compute the dull score for the sample images from Avito's dataset 

# In[29]:


def perform_color_analysis(img, flag):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None


# In[39]:


def score_dullness(data):
    data['dullness'] = data['image'].apply(lambda x : perform_color_analysis(x, 'black'))   
    return data

features = parallelize_dataframe(features, score_dullness)
features.head()


# In[40]:


topdull = features.sort_values('dullness', ascending = False)
topdull.head(5)


# Lets plot some of the images with very high dullness

# In[41]:


for j,x in topdull.head(2).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Dullness : " + str(x['dullness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

