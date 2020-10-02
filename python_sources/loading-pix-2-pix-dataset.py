#!/usr/bin/env python
# coding: utf-8

# **Loading Pix-2-Pix dataset**
# 
# This kernel aims to be a quick introduction to the data in the dataset and how to load it for your exploration and modeling tasks.

# In[ ]:


# let's first check where are our files. Don't forget to include the dataset.
import os
print(os.listdir("../input"))


# In[ ]:


# as you can see, we have 4 different sets of data that we can train our model upon, let's do a quick check to see the overall structure of these sets

def tree(rootpath, skip_file_type):
    """Takes a rootpath to draw a directory structure
    
    Args:
        rootpath: path of the directory to start drawing tree from
        skip_file_type: the type of file to skip
    Return:
        A tree that shows the directory structure
        
    NOTE: This skips the file types based on if the filename ends with the passed value in skip_file_type
    """
    for root, dirs, files in os.walk(rootpath):
        level = root.replace(rootpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        number_of_skipped_files = 0
        for f in files:
            if not f.endswith(skip_file_type):
                print('{}{}'.format(subindent, f))
            else:
                number_of_skipped_files += 1
        if number_of_skipped_files > 0:
            print('{} {} files'.format(subindent, number_of_skipped_files))


# In[ ]:


# note that we are skipping .jpg files to keep the structure clean and legible. you can try it without `.jpg` as well
tree('../input/', '.jpg')


# NOTE: Original data is in `.tar.gz` format but kaggle extracts the files for us to use. Due to that we see that the folder structure is a bit messy, `maps` inside `maps` and so on.
# 
# We also see that only `facades` dataset has `test` data available. It's the recommended dataset to use for getting started with the model. It's also comparitively smaller in size (check details on the dataset page).

# In[ ]:


# let's see a single image
import imageio
import matplotlib.pyplot as plt

def show_raw_images(file_path):
    """Shows a raw image
    
    Args:
        file_path: Path of the file
    Returns:
        The image file
    """
    img = imageio.imread(file_path)
    plt.imshow(img)


# In[ ]:


show_raw_images('../input/facades/facades/train/1.jpg')


# Each image is of size `512x256` divided into two part where first `256x256` part is the actual image and remaining `256x256` part is the trace of the original image.

# In[ ]:


# let's load images in numpy arrays and prepare training and validation data

def load_data(dataset):
    """load the data from a specific dataset
    
    Args:
        dataset: which dataset should be loaded
    Returns:
        If proper dataset, it returns a list of numpy arrays of training and validation data
    """
    possible_datasets = ['maps', 'cityscapes', 'facades']
    if dataset not in possible_datasets:
        print('Dataset not found! Please make sure that you are using either of these datasets:', possible_datasets)
        return
    else:
        print('Loading', dataset)
        train_data_path = '../input/' + dataset + '/' + dataset + '/train'
        validation_data_path = '../input/' + dataset + '/' + dataset + '/val'
        
        train_data = []
        validation_data = []
        for dataset_path in [train_data_path, validation_data_path]:
            for file in os.listdir(dataset_path):
                file_path = dataset_path + '/' + file
                img = imageio.imread(file_path)
                train_data.append(img)
        
        return train_data, validation_data


# In[ ]:


train_data, validation_data = load_data('facades')


# In[ ]:


# load data into an image
plt.imshow(train_data[0])


# In[ ]:


# check the raw data
train_data[0]


# We also see that each pixel value is in the range of 0-255 color scale and the datatype value is `uint8`. Here are some more samples from other sets in the dataset.

# In[ ]:


# maps
show_raw_images('../input/maps/maps/train/100.jpg')


# In[ ]:


# cityscapes
show_raw_images('../input/cityscapes/cityscapes/train/100.jpg')


# In[ ]:


# facades
show_raw_images('../input/facades/facades/train/100.jpg')


# In[ ]:


# edges2shoes
show_raw_images('../input/edges2shoes/edges2shoes/train/10000_AB.jpg')


# **NOTE**: The filename syntax for `edges2shoes` set is `<file_number>_AB.jpg` while all other sets have files with syntax `<file_number>.jpg`.
# 
# This concludes basic introduction to this dataset. Be sure to checkout other kernels for more inspiration!

# In[ ]:




