#!/usr/bin/env python
# coding: utf-8

# The image data for this competition are too large to fit in memory in kernels. This kernel demonstrates how to access individual images in the zip archives without having to extract them or load the archive into memory.

# In[2]:


from PIL import Image
from zipfile import ZipFile


# First, we'll need a way to check what files are in the archive. The  zipfile module let's us look into the archive without loading the whole archive into memory or unpacking the entire archive..

# In[3]:


zip_path = '../input/train_jpg.zip'
with ZipFile(zip_path) as myzip:
    files_in_zip = myzip.namelist()


# In[12]:


files_in_zip[:5]


# In[5]:


len(files_in_zip)


# Now that we have the list of files, we can access individual files by name without loading the whole archive into memory or unpacking the entire archive..

# In[8]:


with ZipFile(zip_path) as myzip:
    with myzip.open(files_in_zip[3]) as myfile:
        img = Image.open(myfile)


# In[11]:


img.size


# In[ ]:




