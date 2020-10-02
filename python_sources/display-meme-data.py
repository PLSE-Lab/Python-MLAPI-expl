#!/usr/bin/env python
# coding: utf-8

# **Display Meme Data**

# [Reddit](http://www.reddit.com) is an American social news aggregation, web content rating, and discussion website that contains a large collection of internet memes. 
# 
# An internet meme is an activity, concept, catchphrase, or piece of media that spreads, often as mimicry or for humorous purposes, from person to person via the Internet. An Internet meme may take the form of an image, hyperlink, video, website, or hashtag. It may be just a word or phrase, sometimes including an intentional misspelling. These small movements tend to spread from person to person via social networks, blogs, direct email, or news sources. They may relate to various existing Internet cultures or subcultures, often created or spread on various websites, or by Usenet boards and other such early-Internet communications facilities. Fads and sensations tend to grow rapidly on the Internet because the instant communication facilitates word-of-mouth transmission. Some examples include posting a photo of people lying down in public places (called "planking") and uploading a short video of people dancing to the Harlem Shake.  
# 
# Citations: 
# 
# * https://en.wikipedia.org/wiki/Reddit
# * https://en.wikipedia.org/wiki/Internet_meme
# 

# *Step 1: Import Python Packages*

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import random
# print(os.listdir("../input"))
# print(os.listdir("../input/memes"))
# print(os.listdir("../input/memes/memes"))


# *Step 2: Define Functions for Plotting Images*

# In[ ]:


multipleImages = glob('../input/memes/memes/**')
def plotImages1(indexStart,indexEnd):
    i_ = 0
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for l in multipleImages[indexStart:indexEnd]:
        im = cv2.imread(l)
        #im = cv2.resize(im, (512, 512)) 
        plt.subplot(5, 5, i_+1) #.set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
        i_ += 1

multipleImages = glob('../input/memes/memes/**')
def plotImages2():
    r = random.sample(multipleImages, 9)
    plt.figure(figsize=(20,20))
    plt.subplot(331)
    plt.imshow(cv2.imread(r[0])); plt.axis('off')
    plt.subplot(332)
    plt.imshow(cv2.imread(r[1])); plt.axis('off')
    plt.subplot(333)
    plt.imshow(cv2.imread(r[2])); plt.axis('off')
    plt.subplot(334)
    plt.imshow(cv2.imread(r[3])); plt.axis('off')
    plt.subplot(335)
    plt.imshow(cv2.imread(r[4])); plt.axis('off')
    plt.subplot(336)
    plt.imshow(cv2.imread(r[5])); plt.axis('off')
    plt.subplot(337)
    plt.imshow(cv2.imread(r[6])); plt.axis('off')
    plt.subplot(338)
    plt.imshow(cv2.imread(r[7])); plt.axis('off')
    plt.subplot(339)
    plt.imshow(cv2.imread(r[8])); plt.axis('off')


# *Step 3: Display Meme Data*

# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()


# In[ ]:


plotImages2()

