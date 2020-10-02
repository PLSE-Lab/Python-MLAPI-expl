#!/usr/bin/env python
# coding: utf-8

# 
# **Forked from** https://www.kaggle.com/orangutan/keras-vgg19-starter
# 
# **For details**,.. https://www.kaggle.com/c/dog-breed-identification
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img
#from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
import sys
import bcolz
import random
import numpy as np
from scipy import ndimage

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First we will read in the csv's so we can see some more information on the filenames and breeds

# In[ ]:


df_train = pd.read_csv('../input/labels.csv')
df_test = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_train.head(10)


# In[ ]:


train_files = glob('../input/train/*.jpg')
test_files = glob('../input/test/*.jpg')


# In[ ]:


plt.imshow(plt.imread(train_files[100]))


# In[ ]:


targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)


# In[ ]:


im_size = 300


# In[ ]:


####  storing train data in a bcolz array with image size 300
y_train = []
x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
x_test_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)


# In[ ]:


####  storing train data in a bcolz array with image size 300
i = 0 
for f, breed in tqdm(df_train.values):
    # load an image from file
    image = load_img('../input/train/{}.jpg'.format(f), target_size=(im_size, im_size))
    image = img_to_array(image)
    # prepare the image for the VGG model
    #image = preprocess_input(image)
    label = one_hot_labels[i]
    x_train_raw.append(image)
    y_train.append(label)
    i += 1


# In[ ]:


####  storing test data in a bcolz array with image size 300
# for f in tqdm(df_test["id"]):
#     # load an image from file
#     image = load_img('../input/test/{}.jpg'.format(f), target_size=(im_size, im_size))
#     image = img_to_array(image)
#     # prepare the image for the VGG model
#     #image = preprocess_input(image)
#     x_test_raw.append(image)


# In[ ]:


print(x_train_raw.shape)
# print(x_test_raw.shape)


# In[1]:


### commentinf since not supported in kernel
# print("Enter batch size - how many images do you want to see \n")
# batch_size=input()
# batch_size=int(batch_size)
# print("Enter the resize_factor between 0.0 and 1.0 \n")
# resize_factor = input()
# resize_factor = float(resize_factor)

batch_size=2
resize_factor=0.8


# In[ ]:





# In[ ]:


def shuffle_dataset_select_indices(x_train_raw,batch_size,im_size):
    print("Let's shuffle the dataset first ")
    np.random.shuffle(x_train_raw)
    print(x_train_raw.shape)
    idx=np.random.choice(im_size, batch_size, replace=False)
    res= x_train_raw[idx]
    res=res/255.
    print("selected indices - ", idx)
    print("please note that the cropping and flipping are done with train images , later it can be extended to test easily")
    
    fig, axes = plt.subplots(1,2, figsize=(12,12))
    axes = axes.flatten()
    i=0
    for img, ax in zip( res, axes):
        ax.imshow(img)
        ax.set_xticks(())
        ax.set_yticks(())
        if i==0:
            ax.set_title("Image 1")
        elif i==1:
            ax.set_title("Image 2")
        i+=1
    plt.tight_layout()
    
    return idx


# In[ ]:


def crop_flip_images(idx,x_train_raw,resize_factor,im_size):
    
    
    temp=[]
    cropping_direction=[]
    flipping_direction=[]
    
    for i in (idx):
        x_temp= x_train_raw[i]
        print("actual image")
        plt.imshow(x_temp/255.)
        temp.append(x_train_raw[i])
        ##  After cropping , to return back the iamge to original size of IM_SIZE (300) , the original image is blurred 
        ## and kept in a IM_SIZE * IM_SIZE * 3 array. Parts of this array would be replaced by the cropped_iamge
        
        x1= np.zeros_like(x_train_raw[0])
        x1 = ndimage.uniform_filter(x_temp, size=(147, 147, 1)) 
        print("Blurred image- helps in retaining the original IM_SIZE shape .              Blurring to a good extent would  in avoding detecting more objects in non-cropped parts ")
#         plt.imshow(x1/255.)

        
        print("Lets crop first :")
              
        
        im_size_new = im_size * resize_factor
        im_size_new= int(im_size_new)

        x_starting_point = im_size-im_size_new
        x_starting_point

        i=0
        randm_number = random.randint(1,101)
        print(randm_number)
        if randm_number <= 20: 
            i=0
            print("Cropping from top left \n")
            x1[:im_size_new,:im_size_new] = x_temp[:im_size_new,:im_size_new]

        elif randm_number > 20 and randm_number <= 40:     ### cropping  bottom left
            i=1
            print("Cropping from bottom left \n")
            x1[:im_size_new,:im_size_new] = x_temp[x_starting_point:,:im_size_new]

        elif randm_number > 40 and randm_number <= 60:     ### cropping top right 
            i=2
            print("Cropping from top right \n")
            x1[:im_size_new,:im_size_new] = x_temp[:im_size_new,x_starting_point:]

        elif randm_number > 60 and randm_number <= 80:     ###cropping bottom right 
            i=3
            print("Cropping from bottom right \n")
            x1[:im_size_new,:im_size_new] = x_temp[x_starting_point:,x_starting_point:]

        elif randm_number > 80 and randm_number <= 100:        ## ccropping enter
            i=4
            print("Cropping from center \n")
            x1[:im_size_new,:im_size_new] = x_temp[(im_size//2)-(im_size_new//2): (im_size//2)+(im_size_new//2),
                                          (im_size//2)-(im_size_new//2): (im_size//2)+(im_size_new//2)]
        temp.append(x1)
        
        
        print("Now lets flip:")
        randm_number = random.randint(1,101)
        print(randm_number)
        if randm_number  <= 50:
            j=0
#             plt.imshow((x2[:,::-1]/255.)) 
            print("Flipping Horizontally")
            x1 = x1[:,::-1] ## horizontalflop
        elif randm_number > 50: 
            j=1
#             plt.imshow((x2[::-1,:]/255.))
            print("Flipping vertically")
            x1 = x1[::-1,:] ## vertical flop

        temp.append(x1)
    return temp,i,j
        
#         


# In[ ]:


def view_images(temp,batch_size,i,j):
    res = np.concatenate([arr[np.newaxis] for arr in temp])
    res.shape
    res=res/255.



    print("Lets see the final images")
    fig, axes = plt.subplots(batch_size,3, figsize=(12,12))
    axes = axes.flatten()
    cntr=0
    for img, ax in zip( res, axes):
        ax.imshow(img)
        ax.set_xticks(())
        ax.set_yticks(())
#         if cntr ==1 and i==0:
#             ax.set_title("Cropped from top left")
#         elif cntr ==1 and i==1:
#             ax.set_title("Cropped from Bottm left")
#         elif cntr ==1 and i==2:
#             ax.set_title("Cropped from top right")
#         elif cntr ==1 and  i==3:
#             ax.set_title("Cropped from bottom right")
#         elif cntr==1 and i==4:
#             ax.set_title("Cropped from centre")
#         if cntr==2 and  j==0:
#              ax.set_title("Flipped Horizontally")
#         if cntr==2 and  j==1:
#              ax.set_title("Flipped Horizontally")
        cntr+=1
    plt.tight_layout()


# In[ ]:


# flag=input("press 1 to start:")
# if flag==1:
idx=shuffle_dataset_select_indices(x_train_raw,batch_size,im_size)
temp,i,j=crop_flip_images(idx,x_train_raw,resize_factor,im_size)
view_images(temp,batch_size,i,j)

