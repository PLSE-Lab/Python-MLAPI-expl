#!/usr/bin/env python
# coding: utf-8

# # Notebok Info
# Hi all! This notebook creates a lighter dataset for the Human Protein Atlas Image Classification. In particular the code below reads all the images and saves them into RGBY format. This is done for a faster reading of the data in the training and testing phases.
# 
# In this example I resize the images from $[512,512,4]$ to $[128,128,4]$ and the output dataset sizes are: 1.055 GB  for the train dataset and 0.356 GB for test dataset.
# If you don't want to resize the images you have to run this kernel in you PC locally because in kaggle kernels you do not have much disk memory.

# In[ ]:


# Import Packages
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm_notebook
from IPython.display import clear_output
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED

# Variables
DATA_DIR = '../input/'


# In[ ]:


# Read Single Image
def read_img(img_id, mode='train', img_size=512):
    img_dir = 'train/'
    if mode=='test':
        img_dir = 'test/'
    
    channels = ['red','green','blue','yellow']
    img = []
    for ch in channels:
        img_ch = cv2.imread(DATA_DIR+img_dir+img_id+'_{}.png'.format(ch), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        if img_size!=512:
            img_ch = cv2.resize(img_ch, (img_size, img_size))
        img.append(img_ch)
    return np.stack(img, axis=-1)


# In[ ]:


# Compress A Folder
def zipdir(basedir, archivename):
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                absfn = os.path.join(root, fn)
                zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
                z.write(absfn, zfn)


# # Load and Save Train Images

# In[ ]:


df_tr = pd.read_csv(DATA_DIR+'train.csv')
train_generator = ([img_id, read_img(img_id, mode='train', img_size=128)] for img_id in df_tr['Id'])

# Save Train Images
os.makedirs('train/') if not os.path.exists('train/') else None
for img_id,img in tqdm_notebook(train_generator, total=df_tr.shape[0]):
    cv2.imwrite('train/{}.png'.format(img_id), img)
    
# Compress Data
zipdir('/kaggle/working/train', 'train.zip')
get_ipython().system('rm train/*')
get_ipython().system('rmdir train')


# # Load and Save Test Images

# In[ ]:


df_te = pd.read_csv(DATA_DIR+'sample_submission.csv')
test_generator = ([img_id, read_img(img_id, mode='test', img_size=128)] for img_id in df_te['Id'])

# Save Train Images
os.makedirs('test/') if not os.path.exists('test/') else None
for img_id,img in tqdm_notebook(test_generator, total=df_te.shape[0]):
    cv2.imwrite('test/{}.png'.format(img_id), img)
    
# Compress Data
zipdir('/kaggle/working/test', 'test.zip')
get_ipython().system('rm test/*')
get_ipython().system('rmdir test')


# # Check Dataset Size

# In[ ]:


get_ipython().system('ls -l ')


# In[ ]:




