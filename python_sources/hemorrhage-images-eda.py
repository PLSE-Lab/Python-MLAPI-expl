#!/usr/bin/env python
# coding: utf-8

# Here is some initial exploratory data analysis I did for [RSNA Intracranial Hemorrhage Detection competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection).
# 
# Bits and pieces were taken from [Marco Vasquez's EDA kernel](https://www.kaggle.com/marcovasquez/basic-eda-data-visualization), so give him a shoutout! That said, a vast majority of the code is my own, simply to provide a better understanding of the data. I will likely update this notebook as more insights become apparent.
# 
# As I have no medical expertise, I will avoid discussing medical aspecs of the data. A vast majority of the medical imaging I am unable to read, and I will be using strictly ML methods to detect any anomolies.

# # Training Labels
# 
# Let us take a look at our training labels.

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')


# In[ ]:


df.head(10)


# As described in the "Data" section of this competition, the column `ID` is broken up as follows:
# 
# ```
# ID_(Patient ID)_(Hemorrhage Type)
# ```
# 
# * We want to get this in a slightly different form, for ease of lookup

# In[ ]:


def get_id(s):
    s = s.split('_')
    return 'ID_' + s[1]

def get_hemorrhage_type(s):
    s = s.split('_')
    return s[2]
    
df['Image_ID'] = df.ID.apply(get_id)
df['Hemhorrhage_Type'] = df.ID.apply(get_hemorrhage_type)

df.set_index('Image_ID')


# In[ ]:


df.head(10)


# All I did here is added two new columns. `Image_ID` (which makes loading a medical image in memory relatively easy) and `Hemhorrhage_Type`, which will tell us what we're looking for.

# # Medical Images
# 
# Now let us take a look at our medical images.

# In[ ]:


from os import listdir
from os.path import isfile, join
from pathlib import Path

# To read medical images.
import pydicom

import matplotlib.pyplot as plt

train_images_dir = Path('../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/')
train_images = [str(train_images_dir / f) for f in listdir(train_images_dir) if isfile(train_images_dir / f)]


# Let us take a look at one image to see what we're dealing with.

# In[ ]:


train_images[0]


# As you can see, images are stored in the following format.
# 
# ```
# stage_1_test_images/ID_(Patient Id).dcm
# ```
# 
# This is percisely the reason why we generated the new columns in the previous section. `.dcm` files, also called DICOM images, is a [specialized format for storing medical images](https://en.wikipedia.org/wiki/DICOM).
# 
# We can load them up using the [pydicom library](https://pydicom.github.io/pydicom/stable/index.html).

# In[ ]:


ds = pydicom.dcmread(train_images[0])
im = ds.pixel_array

plt.imshow(im, cmap=plt.cm.gist_gray);


# In[ ]:


fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.gist_gray)
    fig.add_subplot


# # Statistics
# 
# Now, let us get a better idea of some of the data we're working with

# In[ ]:


import seaborn as sns

sns.countplot(df.Label)

