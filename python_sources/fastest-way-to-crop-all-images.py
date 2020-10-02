#!/usr/bin/env python
# coding: utf-8

# To explore and analyze all the characters in the Kuzushiji Comeptition, you would have to cut out each character. I wanted to save you some time, so I created this notebook and found a pretty fast way to cut out each char.<br> To save you even some more time I saved the computed results of this notebooks in [this dataset](https://www.kaggle.com/christianwallenwein/kuzushiji-characters). If you want to analyze the Kuzushiji Characters, simple use the dataset.

# ## Setup

# In[ ]:


import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image
import cv2

import matplotlib.pyplot as plt

from tqdm import tnrange, tqdm_notebook

import time
import random

# hide warnings
import warnings
warnings.simplefilter('ignore')


# In[ ]:


# all important folders
INPUT = Path("../input")
TEST = INPUT/'test_images'
TRAIN = INPUT/'train_images'


# In[ ]:


CHARS = Path("../chars")
try:
    os.makedirs(CHARS)
except:
    pass


# In[ ]:


train_df = pd.read_csv(INPUT/"train.csv"); train_df.tail(3)


# ## Crop all images

# In[ ]:


def cropImage(labels, loop1_index):
    if isinstance(labels, float):
        return None

    filepath = str(f"{str(TRAIN)}/{row.image_id}.jpg")
    
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    labels = labels.split(" ")

#     change unicode char to integer representing the class
#     labels[::5] = map(unicodeToInt, labels[::5])


    # to get the highest speed, we use numpy for the cropping of the images
    # numpy doesn't support strings in their ndarrays
    # that's why we move unicodes outside of labels
    unicode = labels[::5]
    del labels[::5]
    

    labels = np.array(labels, dtype=np.int16)

    labels = labels.reshape(-1, 4)

    labels[:, 2] = np.sum(a=labels[:,[0,2]], axis=1)
    labels[:, 3] = np.sum(a=labels[:,[1,3]], axis=1)

    [Image.fromarray(img[label[1]:label[3], label[0]:label[2]]).save(f"{CHARS}/{unicode[loop2_index]}_{loop1_index}-{loop2_index}.jpg") for loop2_index, label in enumerate(labels)]


# The **first cell** will always work no matter which operating system. Use this one **on Windows**.
# <br>Linux has better support for Pythons multiprocessing library. Therefore use the **second cell** if you are **on Linux**.
# 
# btw Kaggle Kernels are based on Linux

# In[ ]:


# pbar = tqdm_notebook(total=len(os.listdir(TRAIN)))
# for loop1_index, (list_index, row) in enumerate(train_df.iterrows()):
#     cropImage(row.labels, loop1_index)
#     pbar.update(1)
# pbar.close()


# In[ ]:


from multiprocessing import Process, current_process

processes = []

pbar = tqdm_notebook(total=len(os.listdir(TRAIN)))
for loop1_index, (list_index, row) in enumerate(train_df.iterrows()):
    process = Process(target=cropImage, args=(row.labels, loop1_index))
    pbar.update(1)
    processes.append(process)
    
    process.start()
pbar.close()


# In[ ]:


len(os.listdir(CHARS))


# ## Create ZIP-file from all chars

# In[ ]:


shutil.make_archive("chars", "zip", CHARS)


# ## Useful functions

# In[ ]:


def displayRandomImagesFromFolder(directory):
    images = os.listdir(directory)

    rows = 3
    columns = 3
    fig = plt.figure(figsize=(20, 20))

    for i in range(1, rows*columns + 1):
        randomNumber = random.randint(0, len(images)-1)
        image = Image.open(directory/images[randomNumber])
        fig.add_subplot(rows, columns, i)
        plt.imshow(image, aspect='equal')
    
    plt.show()
    
    
displayRandomImagesFromFolder(CHARS)


# In[ ]:


unicode_df = pd.read_csv(INPUT/'unicode_translation.csv')
display(unicode_df.head(10))
print(len(unicode_df))


# In[ ]:


x = "U+5DDE_2802-291.jpg"
def getInfoFromFilename(filename):
    unicode, rest = filename.split("_")
    char = unicode_df[unicode_df.Unicode == unicode].char.values[0]
    rest = rest.split(".")[0]
    image_nbr = int(rest.split("-")[0])
    image_filename = os.listdir(TRAIN)[image_nbr]
    print(f"This image displays the following char: {char} \nIts unicode is: {unicode} \nThis char has been taken from {image_filename}")
    
    
getInfoFromFilename(x)

