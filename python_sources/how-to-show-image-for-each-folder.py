#!/usr/bin/env python
# coding: utf-8

# # Train

# In[ ]:


# Ensure consistency across runs
from numpy.random import seed
import random
seed(2)
import tensorflow as tf
tf.random.set_seed(2)

# Imports to view data
import cv2
from glob import glob

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Utils
from pathlib import Path
import pandas as pd
import numpy as np


# In[ ]:


# Set global variables
TRAIN_DIR = '../input/shopee-product-detection-open/train/train/train'
TEST_DIR = '../input/shopee-product-detection-open/test/test/test'
CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
CLASSES.sort()

TARGET_SIZE = (64, 64)
TARGET_DIMS = (64, 64, 3) # add channel for RGB
N_CLASSES = 42
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64


# In[ ]:


def plot_one_sample_of_each(base_path):
    cols = 5
    rows = int(np.ceil(len(CLASSES) / cols))
    fig = plt.figure(figsize=(16, 20))
    
    for i in range(len(CLASSES)):
        cls = CLASSES[i]
        img_path = base_path + '/' + cls + '/**'
        path_contents = glob(img_path)
    
        imgs = random.sample(path_contents, 1)

        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.imread(imgs[0]))
        plt.title(cls)
        sp.axis('off')

    plt.show()
    return


# In[ ]:


plot_one_sample_of_each(TRAIN_DIR)


# # EDA TRAIN CSV

# In[ ]:


train_csv = pd.read_csv("../input/shopee-product-detection-open/train.csv")
test_csv = pd.read_csv("../input/shopee-product-detection-open/test.csv")


# In[ ]:


train_csv


# In[ ]:


each_category = train_csv.category.value_counts().sort_index()
each_category


# In[ ]:


train_csv.count()


# In[ ]:


each_category.plot(kind="bar", figsize = (20,5));


# # TEST CSV

# In[ ]:


test_csv


# In[ ]:


test_csv["category"].value_counts()

