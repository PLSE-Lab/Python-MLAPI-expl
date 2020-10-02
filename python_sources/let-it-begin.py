#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


get_ipython().system('ls')


# peeking into the input directory

# In[ ]:


get_ipython().system('ls ../input')


# peeking into the train directory

# In[ ]:


get_ipython().system('ls ../input/train/ | head -5')


# looking into the first 2 rows of `train.csv`

# In[ ]:


get_ipython().system('head -n 2 ../input/train.csv')


# ## here we are following the PyTorch tutorial to import datasets with built in functions
# Dataset class
# -------------
# 
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
# 
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get $i$\ th sample
# 
# Let's create a dataset class for our whale identificatin dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
# 
# Sample of our dataset will be a dict
# ``{'image': image, 'id': ids}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
# 
# 
# 
# 

# In[ ]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Visualizing the image

# In[ ]:


train_data = pd.read_csv("../input/train.csv")


# In[ ]:


train_data.head()


# In[ ]:


image_name = train_data.iloc[0,0]


# In[ ]:


plt.title(str(train_data.iloc[0,1]))
plt.imshow(io.imread(os.path.join("../input/train/",image_name)))


# In[ ]:


#type of the variable that stores id
type(train_data.iloc[0,1])


# In[ ]:


class WhaleIdDataset(Dataset):
    """the whale identification dataset"""
    def __init__(self, csv, rootDir, transform=None ):
        """
        Args:
            csv (string): path to the file with Id
            rootDir (string): path to the root directory
            transform (callable,optional): optional 
                transforms to be applied on the samples.
        """
        self.data_frame = pd.read_csv(csv)
        self.root_dir = rootDir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self,idx):
        image_name = self.data_frame.iloc[idx,0]
        image_path = os.path.join(self.root_dir,
                                  image_name)
        image = io.imread(image_path)
        Id = self.data_frame.iloc[idx,1]
        sample = {"image": image,"id": Id}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample 
        


# loaded the dataset (initiated the class)

# In[ ]:


whaleDataset = WhaleIdDataset("../input/train.csv",
                              "../input/train/")


# accessing an element

# In[ ]:


whaleDataset[1]["image"].shape


# iterating over some values in the dataset

# In[ ]:


def show_image(image, id):
    plt.title(id)
    plt.imshow(image)


# In[ ]:


fig = plt.figure()
for i in range(3):
    a = fig.add_subplot(1, 3, i+1)
    a.axis("off")
    a.set_title(whaleDataset[i]["id"])
    plt.imshow(whaleDataset[i]["image"])


# In[ ]:





# In[ ]:




