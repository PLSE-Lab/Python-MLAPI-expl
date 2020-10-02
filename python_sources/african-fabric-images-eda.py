#!/usr/bin/env python
# coding: utf-8

# Hello. I decided to see what this dataset consists of and if it can be used for a little project. 
# Let's start.

# Now libraries. For start we will need only `matplotlib` and `PIL` for exploring images and `os` for working with directories. 

# In[ ]:


import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


# In[ ]:


dir_input = '/kaggle/input/african-fabric/africa_fabric'


# We can plot some images.

# In[ ]:


def plot_imgs(item_dir, title=" ", num_imgs=4):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]

    plt.figure(figsize=(10, 10))
    for idx, img_path in enumerate(item_files):
        try:
            plt.subplot(4, 4, idx+1)
            img = plt.imread(img_path, 0)
            plt.title(title)
            plt.imshow(img)
        except:
            continue
    plt.tight_layout()


# In[ ]:


plot_imgs(dir_input, num_imgs=16)


# We can see that probably all of them have the same size but we will check to be sure. 

# In[ ]:


def check_size(item_dir):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs]
    print("Number of images: {}".format(len(all_item_dirs)))
    
    for idx, img_path in enumerate(item_files):
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            if width==64 or height==64:
                continue
            else:
                print("Width: {}, height: {}".format(width, height))
        except:
            continue
        
    print("All sizes are (64,64)")


# In[ ]:


check_size(dir_input)
#53 imgs on 20 imgs


# We see that there are 1060 images and all of them are (64,64).
# We can compose them in an image grid. 

# For this I used code from 
# https://www.geeksforgeeks.org/implementing-photomosaics/

# In[ ]:


def createImageGrid(path_images): 
    
    all_item_dirs = os.listdir(path_images)
    item_files = [os.path.join(path_images, file) for file in all_item_dirs]

    m, n = 53, 20 

    width = 64
    height = 64

    # create output image 
    grid_img = Image.new('RGB', (n*width, m*height)) 

    # paste images 
    for idx, img_path in enumerate(item_files): 
        try:
            row = int(idx/n) 
            col = idx - n*row 
            img = Image.open(img_path)
            grid_img.paste(img, (col*width, row*height)) 
        except:
            continue

    return grid_img 


# In[ ]:


plt.figure(figsize=(79.5, 30))
grid = createImageGrid(dir_input)
plt.imshow(grid);

