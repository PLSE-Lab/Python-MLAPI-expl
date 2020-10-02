#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ls '../input/aptos2019-blindness-detection/'


# In[ ]:


DATA_DIR = '../input/aptos2019-blindness-detection/'
train_label_dir = os.path.join(DATA_DIR, 'train.csv')
test_label_dir = os.path.join(DATA_DIR, 'test.csv')

df_train = pd.read_csv(train_label_dir)
df_test = pd.read_csv(test_label_dir)

print(df_train.head())
df_test.head()


# In[ ]:


os.listdir(DATA_DIR+'train_images')


# We can see that the images are named as value in the *id_code* column of *train.csv* and '.png'

# The *DATA_DIR* folder contains:
# * train_images - images to train on or images whose category of diabetic retinopathy is provided(in train.csv)
# * test_images - images to test our model on 
# * train.csv - csv file with two columns: 1)id_code - referring to the image  2) diagnosis - category of the image
# * test.csv - csv file with one column: 1) id_code - referring to the image
# 
# The labels for test data aren't provided - we upload our model to kaggle for getting the test accuracy

# In[ ]:


import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image

DATA_DIR = '../input/aptos2019-blindness-detection/'
MODEL_DIR = '../input/'


# In[ ]:


# class that loads the Train/Validation dataset
class DRDatasetTrain(Dataset):
    def __init__(self, val_set=False, val_size=0.2, random_state=42):
        train_label_dir = os.path.join(DATA_DIR, 'train.csv')       # save path to train.csv file - DATA_DIR + 'train.csv'
        
        df = pd.read_csv(train_label_dir)                           # load train.csv to a Pandas dataframe 
        
        train_size = int((1 - val_size) * len(df)   )               # save size of training data to use - has to be an integer
        df_train = df.sample(train_size, random_state=random_state) # save a sample of size train_size from the df dataframe
                                                                    # random state allows us to get the same sample data if we run the code again later
        idx = [i for i in df.index if i not in df_train.index]      # save the indices of data not in df_train
        
        if val_set:
            df_train = df[idx]     # if we have set val_set=True 
        
        # saves the dataframe to be used later
        # drop=True avoids keeping the current index as a new column
        self.data = df_train.reset_index(drop=True) 
        
    # this function is called when len() function is called on this class' objects
    # >>> train_data = DRDatasetTrain()   >>> len(train_data) 
    def __len__(self):
        return len(self.data)
    
    # this function is called when this class' object is index
    # >>> train_data = DRDatasetTrain()   >>> train_data[2]
    def __getitem__(self, idx):
        # idx - the index of the datapoint we have to return
        id_code = str(self.data.loc[idx, 'id_code'])                      # save the 'id_code' field in the row no. 'idx' in the train.csv file
        file_name = id_code + '.png'
        
        img_file = os.path.join(DATA_DIR, 'train_images', file_name)  # save path of the image file
        img = Image.open(img_file)                                         # open the image file using PIL libraries Image class
        
        # transforms to be used on the image data
        transform = transforms.Compose([transforms.Resize((224, 224)),                    # resize images to 224x224
                                         transforms.ToTensor(),                           # convert PIL image to torch Tensor
                                         transforms.Normalize([0.485, 0.456, 0.406],      # normalize image pixels - two lists correspond to mean and std
                                                              [0.229, 0.224, 0.225])])
        img_tensor = transform(img) # transform the image to be returned
        label = self.data.loc[idx, 'diagnosis']  # save the label of the image to be returned
        
        return (img_tensor, label)# return the image and label as a dictionary


# In[ ]:


# class that loads the Test dataset
class DRDatasetTest(Dataset):
    def __init__(self, val_set=False, val_size=0.2, random_state=42):
        test_label_dir = os.path.join(DATA_DIR, 'train.csv') # save path to train.csv file - DATA_DIR + 'test.csv'
        df = pd.read_csv(test_label_dir)                     # load test.csv to a Pandas dataframe 
        self.data = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        id_code = str(self.data.loc[idx, 'id_code'])                     # save the 'id_code' field in the row no. 'idx' in the train.csv file
        file_name = id_code + '.png'
        
        img_file = os.path.join(DATA_DIR, 'test_images', file_name)  # save path of the image file
        img = Image.open(img_file)    # open the image file using PIL libraries Image class
        
        # transforms to be used on the image data
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        img_tensor = transform(img) # transform the image to be returned
        
        return {'image': img_tensor} # return the image as a dictionary


# In[ ]:


# DataLoader class is used to return data from the dataset in a controlled manner
# batch_size - how many datapoints are returned in each iteration(try reducing this if model takes too much time in each epoch)
# drop_last - should we drop the last batch of data if it's length not equal to batch_size
# shuffle - whether to shuffle the datapoints when returned in each epoch(one complete iteration over entire data)
trainloader = torch.utils.data.DataLoader(DRDatasetTrain(), batch_size=32, drop_last=False, shuffle=False) 
testloader = torch.utils.data.DataLoader(DRDatasetTest(), batch_size=32, drop_last=False, shuffle=False)


# ### Visualize some images

# In[ ]:


# make an iterator out of trainloader and use next() to get a batch(batch_size) of data
images, labels = next(iter(trainloader))
print(images.shape)
print(labels)


# In[ ]:





# In[ ]:




