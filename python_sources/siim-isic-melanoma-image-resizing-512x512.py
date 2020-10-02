#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as immg
import os
import cv2
import shutil
from PIL import Image
from fastai.vision import *
sns.set_style('darkgrid')


# In[ ]:


home = Path('/kaggle/input/melanoma-merged-external-data-512x512-jpeg');home.ls()


# In[ ]:


df = pd.read_csv('/kaggle/input/melanoma-merged-external-data-512x512-jpeg/folds.csv')
df.to_csv('new_data.csv',index=False)


# In[ ]:


path = Path('/kaggle/input/siim-isic-melanoma-classification');path.ls()


# In[ ]:


os.mkdir('/kaggle/working/train_small/')
os.mkdir('/kaggle/working/test_small/')


# In[ ]:


fnames = get_files('/kaggle/input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma')
test = get_files(path/'jpeg'/'test')


# ## Resizing train images 512x512

# In[ ]:


def image_resize(f,index):   # My custom function to process image to jpg and resize to 512x512
    name = str(f).split('/')[-1]
    img = PIL.Image.open(f)
    img = img.resize((256,256),resample=PIL.Image.BILINEAR).convert('RGB')
    loc = '/kaggle/working/train_small/'+str(name)
    img.save(loc ,quality = 85)


# In[ ]:


parallel(image_resize,fnames,max_workers=20)


# In[ ]:


shutil.make_archive('train_small','zip','/kaggle/working/train_small')


# ## Resizing test images 512x512

# In[ ]:


def image_resize_test(f,index):   # My custom function to process image to jpg and resize to 512x512
    name = str(f).split('/')[-1]
    img = PIL.Image.open(f)
    img = img.resize((256,256),resample=PIL.Image.BILINEAR).convert('RGB')
    loc = '/kaggle/working/test_small/'+str(name)
    img.save(loc ,quality = 85)


# In[ ]:


parallel(image_resize_test,test,max_workers=20)


# In[ ]:


shutil.make_archive('test_small','zip','/kaggle/working/test_small')


# In[ ]:




