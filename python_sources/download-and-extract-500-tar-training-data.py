#!/usr/bin/env python
# coding: utf-8

# This Notebook just work on a Local Machine. So you have to download this notebook.
# 
# Here's a script version: https://www.kaggle.com/arteam/download-and-extract-tar-files-train-data-script
# 
# This a notebook to download and extract the Train Data from the 500 .tar files. It is based on PC Jimmmy's [*Download 500 tar Training dataset*!](https://www.kaggle.com/pcjimmmy/download-500-tar-training-dataset) . Credits and Thanks to PC Jimmmy!!!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from urllib import request, error
from PIL import Image
from io import BytesIO
import csv
import tarfile

import tensorflow as tf
import tempfile
import urllib
import tarfile


# In[ ]:


output_dir_for_tar_files = 'download'
output_dir_for_images = 'train_data'


# Note: These Directories *don't*(!!!) have to exist. They had just in an older version with 'os.chdir(path)' to exist! Sorry!
# 
# 

# In[ ]:


data_file = []
for i in range(500):
    i_str = str(i)
    if len(i_str) < 2:
        i_str = '00'+i_str
    elif len(i_str) < 3:
        i_str = '0'+i_str
    data_file.append('https://s3.amazonaws.com/google-landmark/train/images_'+i_str+'.tar')


# In[ ]:


def download(directory, url, filename):
  """Download a tar file from the train dataset if not already done. This permits you to rerun and not download already existing tar files from previous attempts."""
  # if the file is already present we don't want to do anything but ack its presence
  filepath = directory+'/'+filename
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  print('Downloading %s to %s' % (url, filepath))  
  urllib.request.urlretrieve(url, filepath)


# In[ ]:


def extract(tar_file, path):
    opened_tar = tarfile.open(tar_file)
     
    if tarfile.is_tarfile(tar_file):
        opened_tar.extractall(path)
    else:
        print("The tar file you entered is not a tar file")


# In[ ]:


for row in data_file:
        amazon_location = row
        file_name = amazon_location[-14:]
        print(amazon_location)
        print(file_name)
        download(output_dir_for_tar_files, amazon_location, file_name)
        extract(output_dir_for_tar_files+'/'+file_name, output_dir_for_images)


# It's tested on Ubuntu, but I think it should work on Windows, too.
