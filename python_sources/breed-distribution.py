#!/usr/bin/env python
# coding: utf-8

# **Our goal is to check how many train images we have per breed**

# Lets import some libraries first.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image


# Now we can load train data and check structure

# In[ ]:


labels = pd.read_csv("../input/labels.csv")
labels.head()


# Ok, lets check how many unique breeds we have

# In[ ]:


unique_breeds = labels.breed.unique()
len(unique_breeds)


# We have good news - all breeds are present in train dataset.
# 
# So lets calculate amount if images for every breed and check most frequent.

# In[ ]:


gr_labels = labels.groupby("breed").count()
gr_labels = gr_labels.rename(columns = {"id" : "count"})
gr_labels = gr_labels.sort_values("count", ascending=False)
gr_labels.head()


# Ok, so most popular breed in dataset is scottish deerhound with 126 images.
# Lets look on one of them :)

# In[ ]:


scottish_deerhound_id = labels.loc[labels.breed == "scottish_deerhound"].iloc[0, 0]
Image.open("../input/train/"+scottish_deerhound_id+".jpg")


# And now lets check the most critical point: is there is very rare breed in dataset.

# In[ ]:


gr_labels.tail()


# Five most rare breeds have 66-67 images. Thats very nice.
# 
# And for sure we need to check how eskimo dog looks like :)

# In[ ]:


eskimo_dog_id = labels.loc[labels.breed == "eskimo_dog"].iloc[1, 0] #0 row is too agressive, so i decided to take 1st :)
Image.open("../input/train/"+eskimo_dog_id+".jpg")


# ***Good luck!***
