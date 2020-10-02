#!/usr/bin/env python
# coding: utf-8

# ## Create Yolo Labels
# 
# This is the poorly written and pretty inefficient code I used to quickly create files for YOLO. The code could definately be improved but oh well
# 
# #### ONLY CREATING LABELS FOR THE ONES WITH CHARACTERS the ones without any labels will be skipped

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm_notebook as tqdm

import os
print(os.listdir('../input'))


# Read the train set and unicode translations

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
translations_df = pd.read_csv('../input/unicode_translation.csv')


# Drop the images with no labels from train_df

# In[ ]:


train_df = train_df.dropna()


# In[ ]:


def unicode_to_num(unicode):
    '''Translates unicode ID to location from translations'''
    return translations_df[translations_df['Unicode'] == unicode].index[0]


# In[ ]:


def label_to_yolo(label, dim):
    '''Converts the given "label" to proper YOLO format by doing this:
            - Replacing unicode ID with class
            - Scaling coordinates/sizes to [0,1] 
    '''
    height = dim[0]
    width = dim[1]
    _label = label.split()
    for index in range(0, len(_label)):
        if index % 5 == 0:
            _label[index] = unicode_to_num(_label[index])
        elif (index % 5 == 1) | (index % 5 == 3):
            _label[index] = int(_label[index]) / width
        elif (index % 5 == 2) | (index % 5 == 4):
            _label[index] = int(_label[index]) / height

    return(_label)


# In[ ]:


def get_img_dimensions(path):
    '''Returns the image dimensions of an image_id'''
    img = cv2.imread('../input/train_images/'+path+'.jpg')
    return img.shape


# In[ ]:


if not os.path.exists('yolo_train_labels'):
    os.mkdir('yolo_train_labels')
def yolo_to_txt(yolo, file_name):
    '''Writes given YOLO label to a text file'''
    file = open('yolo_train_labels/'+file_name+'.txt', "w")
    for index in range(0, len(yolo), 5):
        to_write = ' '.join(str(x) for x in yolo[index:index+5])
        file.write(to_write + '\n')
    file.close()


# In[ ]:


def write_row(image_id, labels):
    img_shape = get_img_dimensions(image_id)
    if type(labels) == str:
        yolo = label_to_yolo(labels, dim=img_shape)
        yolo_to_txt(yolo, image_id)


# In[ ]:


for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    write_row(row['image_id'], row['labels'])


# Now to archive the labels in a zip

# In[ ]:


import shutil
shutil.make_archive('yolo_labels', 'zip', 'yolo_train_labels')


# In[ ]:


os.listdir('.')


# In[ ]:


get_ipython().system('rm -rf yolo_train_labels')


# In[ ]:


os.listdir('.')


# In[ ]:


class_names = open('classes.names', 'w')
for index, row in tqdm(translations_df.iterrows(), total=translations_df.shape[0]):
    class_names.write(row['Unicode']+'\n')

