#!/usr/bin/env python
# coding: utf-8

# This is another kernel in which I will try to explore image processing and classification with CNNs.
# I use some function from my other kernel: https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualization

# In[ ]:


import imageio

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import scipy.ndimage as ndi


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dirname_input = '/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5'
dir_input_list = os.listdir(dirname_input)
print(dir_input_list)
print("\n")
print("Number of patients: {}".format(len(dir_input_list)))


# We have to deal with 279 folders and in each of them there are two folders: "0" and "1" containing images. 
# I decided to approach this problem using dictionary. Name of the folder will be the key and the values will be 0 and 1 folder. 

# In[ ]:


#Function for creating path 
def path_img(path, folder='0'): #folder can assume string values '0' or '1'
    return os.path.join(path, folder)


# In[ ]:


#creating dictionary  for every folder, key - folder name, values - subfolders 0 and 1
dir_img_folders = []
dir_dict = {}
for folder in dir_input_list:
    dir_dict[folder] = os.path.join(path_img(dirname_input, folder), "0"), os.path.join(path_img(dirname_input, folder), "1")


# Let's see how looks structure of a random folder.

# In[ ]:


#example
print(dir_dict['13666'])
print(dir_dict['13666'][0])
print(dir_dict['13666'][1])


# Now we can see what kind of images we have in those folders.

# In[ ]:


def plot_imgs(item_dir, num_imgs=25, title=" "):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]
    img_shape = {"50,50,3": 0,
                "other": 0}
    cntr1 = 0
    cntr2 = 0
    
    plt.figure(figsize=(10, 10))
    
    for idx, img_path in enumerate(item_files):
        plt.subplot(5, 5, idx+1)
        img = plt.imread(img_path)
        if img.shape == (50,50,3):
            cntr1 = cntr1  + 1
            img_shape["50,50,3"] = cntr1
        else:
            cntr2 = cntr2 + 1
            img_shape["other"] = cntr2
            print("New shape: {}".format(img.shape))
        plt.title(title)
        plt.imshow(img)
        
    plt.tight_layout()


# In[ ]:


plot_imgs(dir_dict[dir_input_list[0]][1], 5)


# I had to see what sort of images were in each folder. Based on one folder I hoped every picture is the same size. 

# In[ ]:


for idx, _ in enumerate(dir_input_list):
    plot_imgs(dir_dict[dir_input_list[idx]][0], 5, title="Patient: {}, class: {}".format(dir_input_list[idx], "0"))
    plot_imgs(dir_dict[dir_input_list[idx]][1], 5, title="Patient: {}, class: {}".format(dir_input_list[idx], "1"))


# Unfortunetally, some images are different size than others. Let's see how many of them are there.

# In[ ]:


# def shape_counter(item_dir):
#     all_item_dirs = os.listdir(item_dir)
#     item_files = [os.path.join(item_dir, file) for file in all_item_dirs]
    
#     cntr1 = 0
#     cntr2 = 0
    
#     img_shape = {"50,50": 0}
    
#     plt.figure(figsize=(10, 10))
    
#     for idx, img_path in enumerate(item_files):
#         img = Image.open(img_path)
#         width, height = img.size
#         if '{},{}'.format(width, height) in img_shape: #checking if shape exists
#             img_shape['{},{}'.format(width, height)] = img_shape['{},{}'.format(width, height)] + 1 #if it does we update value
#         else:
#             new_shape = {'{},{}'.format(width, height): 1} #if not we create new key with value 1
#             img_shape.update(new_shape)
        
#         print(img_shape)
        
#     return img_shape


# In[ ]:


# for idx, _ in enumerate(dir_input_list):
#     shp_cnt0 = shape_counter(dir_dict[dir_input_list[idx]][0])
#     shp_cnt1 = shape_counter(dir_dict[dir_input_list[idx]][1])


# In[ ]:


# shapes = shp_cnt0.copy()
# shapes.update(shp_cnt1)
# print(shapes)


# In[ ]:


# def shapes_plot(shape_dict):
#     shape_keys = shape_dict.key() #getting keys
#     shape_num = len(shape_keys) #checking how many keys there is
#     plt.bar(range(0,shape_num+1), [shape_keys]) #plotting bars in range using len
#     plt.xticks(range(0,shape_num+1), [shape_keys])
#     plt.show()


# In[ ]:


# shape_comparison(shapes)


# In[ ]:





# In[ ]:





# In[ ]:




