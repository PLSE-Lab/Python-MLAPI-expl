#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from torchvision import datasets, transforms
from torch.utils import data
from torchvision.utils import make_grid

train_dataset = datasets.ImageFolder(root = '../input/tiny-imagenet/tiny-imagenet-200/train', 
                transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()]))

train_loader = data.DataLoader(train_dataset, batch_size = 128, shuffle = True, 
                               num_workers = 3, pin_memory = True)

val_dataset = datasets.ImageFolder(root = '../input/tiny-imagenet/tiny-imagenet-200/val', 
              transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()]))

val_loader = data.DataLoader(val_dataset, batch_size = 128, num_workers = 3, pin_memory = True)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for images, _ in train_loader:
  plt.figure(figsize = (16, 16))
  plt.axis('off')
  for image in images:
    print(images.shape)
    break
  plt.imshow(make_grid(images, nrow = 8, padding = 0).permute(2, 1, 0))
  break


# In[ ]:



