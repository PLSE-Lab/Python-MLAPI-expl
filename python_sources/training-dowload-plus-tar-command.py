#!/usr/bin/env python
# coding: utf-8

# Just a compact form of original script plus tar command. Linux version. Copy and execute localy. Thanks to [PC Jimmmy](https://www.kaggle.com/pcjimmmy)

# In[ ]:


import os
from urllib import request, error
import tensorflow as tf
import tempfile
import urllib

################
# Change here
output_dir = '//d02/data/google_recog2019'
################

data_file =[]
for i in range(500):
    data_file.append('https://s3.amazonaws.com/google-landmark/train/images_%03d.tar' % (i) )


def download(directory, url, filename):
    """Download a tar file from the train dataset if not already done.
    This permits you to rerun and not download already existing tar files from previous attempts."""
    filepath = os.path.join(directory, filename)
    # if the file is already present we don't want to do anything but ack its presence
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

    _, zipped_filepath = tempfile.mkstemp(suffix='.tar')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    tf.gfile.Copy(zipped_filepath, filepath)

    os.remove(zipped_filepath)
    
    command = 'tar xvf ' + filepath + ' -C ' + output_dir
    os.system(command)

    return filepath

for row in data_file:
        amazon_location = row
        file_name = amazon_location[-14:]
        print(amazon_location)
        print(file_name)
        download(output_dir, amazon_location, file_name)

print('Downloading completed  ...')


