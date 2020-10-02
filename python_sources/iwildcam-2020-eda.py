#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import tqdm
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from shutil import copyfile

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import cv2


# In[ ]:


train_images = glob.glob('../input/iwildcam-2020-fgvc7/train/*')
test_images = glob.glob('../input/iwildcam-2020-fgvc7/test/*')


# In[ ]:


print("number of train jpeg data:", len(train_images))
print("number of test jpeg data:", len(test_images))


# In[ ]:


with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_annotations_json = json.load(json_file)


# In[ ]:


print(train_annotations_json.keys())

for key in train_annotations_json.keys():
    if key == 'info': continue
    print(key, train_annotations_json[key][0])
    
unique_categories = set()
for cat in train_annotations_json['categories']:
    unique_categories.add(cat['id'])
print(len(unique_categories))


# In[ ]:


def PlotImageWithCategory(image_name):
    image_id = image_name.replace(".jpg","")
    image_id = image_id.replace("../input/iwildcam-2020-fgvc7/train/","")
    cat_id = [annot['category_id'] for annot in train_annotations_json['annotations'] if annot['image_id'] == image_id][0]
    im = Image.open(image_name)
    cat_name = [cat['name'] for cat in train_annotations_json['categories'] if cat['id'] == cat_id][0]
    print(cat_name)
    plt.rcParams["figure.figsize"] = (100,100)
    plt.imshow(im)


# In[ ]:


PlotImageWithCategory(train_images[22])


# In[ ]:


def PlotImagesGivenCatId(cat_id):
    cat_name = [cat['name'] for cat in train_annotations_json['categories'] if cat['id'] == cat_id][0]
    print(cat_name)
    image_ids = [annot['image_id'] for annot in train_annotations_json['annotations'] if annot['category_id'] == cat_id]
    image_files = ["../input/iwildcam-2020-fgvc7/train/"+id+".jpg" for id in image_ids]
    fig = plt.figure(figsize=(25, 16))
    for i,im_path in enumerate(image_files[:4]):
        ax = fig.add_subplot(2, 2, i+1, xticks=[], yticks=[])
        im = Image.open(im_path)
        im = im.resize((480,270))
        plt.imshow(im)


# In[ ]:


PlotImagesGivenCatId(list(unique_categories)[4])


# In[ ]:


sample_submission = pd.read_csv('/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv')
sample_submission.head()


# In[ ]:


train_df = pd.DataFrame(columns=['image_id','cat_id'])

for train_image in train_images:
    image_id = train_image.replace(".jpg","")
    image_id = image_id.replace("../input/iwildcam-2020-fgvc7/train/","")
    cat_id = [annot['category_id'] for annot in train_annotations_json['annotations'] if annot['image_id'] == image_id][0]
    
    train_df = train_df.append(pd.DataFrame({'image_id': image_id, 'cat_id': cat_id}, index=[0]))
    
train_df.to_csv('/kaggle/working/train.csv', index=False)

