#!/usr/bin/env python
# coding: utf-8

# ## Part 3 - This is officially hard...

# ### I'm following my plan!
# 
# In the last kernel I indicated that I was going to try an approach where I used Transfer Learning to get started.  
# 
# I did exactly that and pretty quickly was able to use Keras Applications to create ResNet50 / InceptionV3 / Xception networks modified to train on the data provided for this challenge.
# 
# I tried limiting the categories to 1024 then 2048 then 4096 and even 8192...
# 
# With each option the training took longer, but the validation accuracy got better (as high ar 85%).
# 
# My gap score also increased from essentially 0 to 0.039 (yay - good enough to reach #94).
# 
# However, when I was reviewing my results to get ideas how to improve further, I think I may be on the complete wrong path!    

# ### Look at the data
# 
# I wrote a simple script to make a directory structure filled with my predicted results

# In[7]:


import sys, os, csv
from shutil import copy2
from IPython.display import Image
from IPython.core.display import HTML 
from urllib import request
import pandas as pd


# In[8]:


def get_image_list(image_list_file_name):

    csvfile = open(image_list_file_name,'r')
    csvreader = csv.reader(csvfile)
    image_data_list = [line for line in csvreader]


    return image_data_list[1:]  # get rid of header


# In[9]:


def copy_to_category(image_data,base_dir):
    
    src_file = image_data[0]
    category = image_data[1]

    dest_dir = base_dir + str(category)

    if os.path.exists(src_file):
        if os.path.exists(dest_dir):
            copy2(src_file,dest_dir)
        else:
            os.makedirs(dest_dir)
            copy2(src_file,dest_dir)


# In[10]:


# copy the results to their direcories
# this won't run in this notebook

'''results_file = 'Xception_results4096.csv'

image_list = get_image_list(results_file)

base_dir = 'results4096/'

for i in range(len(image_list)):
    if image_list[i][1] != '':
        copy_to_category(image_list[i],base_dir)'''


# #### Looking at the results data...
# 
# Here is an example of my "results" vs the actual landmarks

# In[11]:


# A utility for displaaying thumbnails of images
# Taken from the very nice Kernel by Gabriel Preda
# "Google Landmark Recogn. Challenge Data Exploration"

def displayLandmarkImagesLarge(urls):
    
    imageStyle = "height: 300px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))


# In[12]:


# This is an example from the training set for landmark id 113

urls = [] # start with an empty list...

urls.append('https://lh6.googleusercontent.com/-mCwmM3A2ERA/TFLmuXWSi4I/AAAAAAAADMA/G71O3nrFG98/')
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls) # Thanks Gabriel!


# This is the landmark we are looking for...

# In[13]:


# These are some examples of images from the test set classified as landmark 113...

urls = [] # start with an empty list...

urls.append('https://lh3.googleusercontent.com/-H7ANxV_pIJA/WMGQhwlztAI/AAAAAAAAAC0/ptNEb6sTWksaf7j3MefgXhBAD2d5xY8oACOcB/')
urls.append('https://lh3.googleusercontent.com/-MxdgZ7lWPFA/WOeFXXMggfI/AAAAAAAAABI/rvdJbEIzFYc7wg4w-NrAeQUQQ_y0zClvgCJkC/')
urls.append('https://lh3.googleusercontent.com/-p0HjQL5LBGE/WMFlmBrGywI/AAAAAAAAGz4/x2Qjgjj8gKg3s3BSicvN0sKsNlqhYxDSQCOcB/')
urls.append('https://lh3.googleusercontent.com/-FuYFJZNTVKw/WMqi-muDCLI/AAAAAAAAAIE/q5y-LkQN1qMx0yXSviqsreEkZdc4GQGqACOcB/')
urls.append('https://lh3.googleusercontent.com/-UM5fv-HXThE/WI1CRj-qLyI/AAAAAAAFu-g/vTpp3Cn9ba4OvYmS96ulmFvmC3s5raxNgCOcB/')
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls)


# ### These are not the droids I'm looking for...
# 
# For most of these, I can kind of see the similarity, but they really aren't very close at all (and who knows what is going on with the bus photo!)
# 
# It's also not like the predictions for these were low confidence.  They ranged from 98% probability to 39%!

# #### Back to the drawing board to rethink my approach!

# In[ ]:




