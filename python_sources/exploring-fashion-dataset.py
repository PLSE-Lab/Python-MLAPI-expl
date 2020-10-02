#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

line = "#"*140

print(plt.style.available, "\n", line) # get available plotting style

plt.style.use('classic')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Reading Dataset**

# In[ ]:


files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        files.append(path)

files.sort() # Be aware about that indexes (may change) due to os.walk is a recursive function
print(files) 


# In[ ]:


train = pd.read_csv(files[2])
train.head()


# In[ ]:


test = pd.read_csv(files[1])
test.head()

sample = pd.read_csv(files[0])
sample.head()
# # **Data Visualization & EDA**

# In[ ]:


raw_images , classes = train.drop(['class_name','class_label'] , axis = 1) , train[['class_name','class_label']]

print('Raw Pixel Data :',raw_images.shape)
print('Labels :',classes.shape)

print(line,end='\n\n')

labels = np.unique(classes.class_label.to_numpy())
names  = [classes.query(f'class_label == {l}').class_name.to_numpy()[0] for l in labels]

print('Classes\n',line)

for n,l in zip(names,labels):
    print(n,':',l)

 


# In[ ]:


def ShowImage(raw_image__,classes,i): # The function should take images raw data, classes &  i -->(index) and plot image at specific (index = i) 

    raw   = raw_image__.iloc[i].to_numpy() # Get Image at the ( index = i )

    label =  classes.iloc[i]
    class_name  = label[0]
    class_label = label[1]

    resized_pixels = raw.reshape(28 , 28) # resize (784) to (28,28) array
    
    plt.figure(figsize=(7,4))
    plt.imshow(resized_pixels) # plot the image
    plt.title(f'{class_name} --- {class_label}')
    plt.show()


# In[ ]:


print('Min pixel value :',raw_images.min().min())
print('Max pixel value :',raw_images.max().max())

ShowImage(raw_images, classes, 9)
ShowImage(raw_images, classes, 5)
ShowImage(raw_images, classes, 10)
ShowImage(raw_images, classes, 42)


# In[ ]:


print(classes['class_name'].value_counts()) # Number of each class in the dataset


# In[ ]:


binary = (raw_images < 128).astype('int8')

print('Min pixel value :',binary.min().min())
print('Max pixel value :',binary.max().max())

ShowImage(binary, classes, 9)
ShowImage(binary, classes, 5)
ShowImage(binary, classes, 10)
ShowImage(binary, classes, 42)


# In[ ]:


normlized = (raw_images - raw_images.mean()) / raw_images.std() # Z-Score

print('Min pixel value :',normlized.min().min())
print('Max pixel value :',normlized.max().max())

ShowImage(normlized, classes, 9)
ShowImage(normlized, classes, 5)
ShowImage(normlized, classes, 10)
ShowImage(normlized, classes, 42)


# In[ ]:


def get_by_level(raw_images__, high = 8, low = 0):    
    
    cmin = raw_images__.min()
    cmax = raw_images__.max()

    cscale = cmax  - cmin


    scale = (high - low) / cscale

    converted_images = (raw_images__ - cmin) * scale + low
    converted_images = (converted_images.clip(low, high) + 0.5).astype('uint8')
    
    print('Min pixel value :',converted_images.min().min())
    print('Max pixel value :',converted_images.max().max())
    
    return converted_images


# In[ ]:


level_8 = get_by_level(raw_images, high=8, low=0)

ShowImage(level_8, classes, 9)
ShowImage(level_8, classes, 5)
ShowImage(level_8, classes, 10)
ShowImage(level_8, classes, 42)


# In[ ]:


level_4 = get_by_level(raw_images, high = 4, low =0)

ShowImage(level_4, classes, 9)
ShowImage(level_4, classes, 5)
ShowImage(level_4, classes, 10)
ShowImage(level_4, classes, 42)


# In[ ]:


level_2 = get_by_level(raw_images, high = 2, low =0)

ShowImage(level_2, classes, 9)
ShowImage(level_2, classes, 5)
ShowImage(level_2, classes, 10)
ShowImage(level_2, classes, 42)


# In[ ]:


level_1 = get_by_level(raw_images, high = 1, low =0) # Binary

ShowImage(level_1, classes, 9)
ShowImage(level_1, classes, 5)
ShowImage(level_1, classes, 10)
ShowImage(level_1, classes, 42)


# In[ ]:




