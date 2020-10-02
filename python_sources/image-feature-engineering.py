#!/usr/bin/env python
# coding: utf-8

# Working off of [Wesamelshamy's Image Recognition Kernel](https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring) and using more features from OpenCV, I try to get more information from each image, like the number of colors, the kinds of colors, etc. There's a lot we can do here. Now we just have to figure out how to run it at scale for all the images and see what works in our models.

# In[1]:


import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


# Due to Kaggle's disk space restrictions, we will only extract a few images to classify here.  Keep in mind that the pretrained models take almost 650 MB disk space.

# In[2]:


"""Extract images from Avito's advertisement image zip archive.

Code adapted from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import zipfile

NUM_IMAGES_TO_EXTRACT = 10

with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])

get_ipython().system('mv *.jpg/data/competition_files/train_jpg/* ~/avito_images')
get_ipython().system('rm -rf *.jpg')


# In[3]:


import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3


# In[4]:


resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')


# In[58]:


from PIL import Image
import cv2

def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]


def classify_and_plot(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    cv_img = cv2.imread(image_path)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    return (img, cv_img, preds_arr)


# In[59]:


image_files = [x.path for x in os.scandir(images_dir)]


# In[206]:


from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def get_data_from_image(dat):
    plt.imshow(dat[0])
    img_size = [dat[0].size[0], dat[0].size[1]]
    (means, stds) = cv2.meanStdDev(dat[1])
    mean_color = np.mean(dat[1].flatten())
    std_color = np.std(dat[1].flatten())
    color_stats = np.concatenate([means, stds]).flatten()
    scores = [i[1][0][2] for i in dat[2]]
    labels = [i[1][0][1] for i in dat[2]]
    df = pd.DataFrame([img_size + [mean_color] + [std_color] + color_stats.tolist() + scores + labels],
                      columns = ['img_size_x', 'img_size_y', 'img_mean_color', 'img_std_color', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'image_green_std', 'image_red_std', 'Resnet50_score', 'xception_score', 'Inception_score', 'Resnet50_label', 'xception_label', 'Inception_label'])
    return df

dat = classify_and_plot(image_files[0])
df = get_data_from_image(dat)
print(df.head())


# In[207]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[1])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[208]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[2])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[209]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[3])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[210]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[4])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[211]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[5])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[212]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[6])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[213]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[7])\ndf = get_data_from_image(dat)\nprint(df.head())')


# In[214]:


get_ipython().run_cell_magic('time', '', 'dat = classify_and_plot(image_files[8])\ndf = get_data_from_image(dat)\nprint(df.head())')

