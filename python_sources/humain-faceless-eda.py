#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import numpy as np 
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.utils import to_categorical


# #### Paths and Files

# In[ ]:


imagesPath = '/kaggle/input/utkface-images/utkfaceimages/UTKFaceImages/'
labelsPath = '/kaggle/input/utkface-images/'


# In[ ]:


files = os.listdir(labelsPath)
labels = pd.read_csv(labelsPath+files[0])


# In[ ]:


images = os.listdir(imagesPath)


# ### Usual EDA

# In[ ]:


labels.head()


# In[ ]:


labels.sample(10)


# In[ ]:


labels.describe()


# In[ ]:


labels.dtypes


# > While building the dataset there were some error in the name of the image. Due to this few image labels for ethnicity is incorrect. `ethnicity` should be `int64` dtype but due to error in the image names it's object dtype.

# In[ ]:


print("Unique values in gender; ", labels['gender'].unique())


# In[ ]:


print("Unique values in ethnicity; ", labels['ethnicity'].unique())


# > We found the culprit. In data cleaning step we shall remove wrong data points.

# In[ ]:


ages = labels['age'].unique()
ages.sort()
print("Unique values in age; ", ages)


# > Thus age ranges from 1-116. It's continuous value thus regression should be one approach. But what's the distribution of data point? How many datapoints per age value? But before that let's clean the data.

# In[ ]:


labels = labels[labels.ethnicity != '20170109150557335.jpg.chip.jpg']
labels = labels[labels.ethnicity != '20170116174525125.jpg.chip.jpg']
labels = labels[labels.ethnicity != '20170109142408075.jpg.chip.jpg']

labels = labels.astype({'ethnicity': 'int64'})


# In[ ]:


labels.describe()


# In[ ]:


labels.dtypes


# ### Distribution of data

# In[ ]:


plt.figure(figsize=(13,8))
labels['age'].hist(bins=len(ages));


# In[ ]:


labels.groupby('age').count().sort_values('image_id', ascending=False).head(10)


# > The distribution is higly non uniform. Age 26 is dominant followed by age 1. We need to select the data acccordingly.
# 
# ##### What if we group the data

# In[ ]:


plt.figure(figsize=(13,8))
labels['age'].hist(bins=[0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)]);


# > This modified distribution looks better. If we approach age as a classification problem we can avoid the higly uneven distribution of data. The problem statement laid by TCS HumAIn ask to predict age in groups. Thus this approach is better. 
# 
# 
# #### Gender

# In[ ]:


genders = labels['gender'].unique()

plt.figure(figsize=(13,8))
labels['gender'].hist(bins=len(genders));


# > This label is good to go
# 
# #### Ethnicity

# In[ ]:


ethnicity = labels['ethnicity'].unique()

plt.figure(figsize=(13,8))
labels['ethnicity'].hist(bins=len(ethnicity));


# > Now this is not so good. Class `0` for the label `ethnicity` have more than twice the data points. 

# ### Show Training Data

# In[ ]:


def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: print('Serial title'); titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap=None)
        a.set_title(title, fontsize=50)
        a.grid(False)
        a.axis("off")
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig('faceless.png')
plt.show()


# In[ ]:


samples = np.random.choice(len(images), 16)
sample_images = []
sample_labels = []
for sample in samples:
    img = cv2.imread(imagesPath+images[sample])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    sample_images.append(img)


# In[ ]:


show_images(sample_images, 4)


# In[ ]:




