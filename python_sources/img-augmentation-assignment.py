#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import bcolz
import random
from tqdm import tqdm
from glob import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt


# In[ ]:


df_train = pd.read_csv('../input/dog-breed-identification/labels.csv')
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
train_files = glob('../input/dog-breed-identification/train/*.jpg')
test_files = glob('../input/dog-breed-identification/test/*.jpg')


# In[ ]:


targets_series = pd.Series(df_train['breed'])
targets_names = df_train['breed'].tolist()
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
print(targets_names[0:5])


# In[ ]:


im_size = 300
x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
x_val_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
y_train = []
y_val = []
y_train_name = []
y_val_name = []


# In[ ]:


x_train_raw1 = bcolz.zeros((1,4,2,3),np.float32)
print(x_train_raw1)


# In[ ]:


i = 0 
for f in tqdm(train_files):
    image = load_img(train_files[i].format(f), target_size=(im_size, im_size))
    image = img_to_array(image)
    label = one_hot_labels[i]
    names = targets_names[i]
    if random.randint(1,101) < 80: 
        x_train_raw.append(image)
        y_train.append(label)
        y_train_name.append(names)
    else:
        x_val_raw.append(image)
        y_val.append(label)
        y_val_name.append(names)
    i += 1


# In[ ]:


print(x_train_raw.shape, x_val_raw.shape)
print(len(y_val),len(y_val), len(y_val_name))
plt.imshow(x_train_raw[2]/255)


# In[ ]:


# dummy data for testing
dummyX = x_train_raw[0:16]
dummyY = y_train[0:16]
dummyN = y_train_name[0:16]
print(dummyN)


# #### defining fucntion img_trans

# In[ ]:


def img_trans(img_data):
    im_size = img_data.shape
    nd1 = int(np.floor(im_size[0]*0.8))
    nd2 = int(np.floor(im_size[0]*0.8))
    cnd1 = int(np.floor((im_size[0]-nd1)*0.5))
    cnd2 = int(np.floor((im_size[1]-nd2)*0.5))
    k = random.randint(0,4)
    if k == 0:
        new_image = img_data
        new_image[0:cnd1*2] = [0,0,0]
        new_image[:,0:cnd2*2] = [0,0,0]
    if k == 1:
        new_image = img_data[im_size[0]-nd1:im_size[0],0:nd2]
    if k == 2:
        new_image = img_data[0:nd1,im_size[1]-nd2:im_size[1]]
    if k == 3:
        new_image = img_data[im_size[0]-nd1:im_size[0],im_size[1]-nd2:im_size[1]]
    if k == 4:
        new_image = img_data[cnd1:im_size[0]-cnd1,cnd2:im_size[1]-cnd2]
    hflip = random.randint(1,101)
    if hflip > 50:
        new_image = np.flip(new_image,1)
    vflip = random.randint(1,101)
    if vflip > 50:
        new_image = np.flip(new_image,0)
    
    return new_image


# #### function for New data generation and augmentation

# In[ ]:


def data_augmentation(image_data, image_labels, image_names):
    x_aug = bcolz.zeros((0,im_size,im_size,3),np.float32)
    y_aug = []
    yN_aug = []
    count = len(image_labels)
    i=0
    for i in range(count):
        x_aug.append(image_data[i]/255.)
        y_aug.append(image_labels[i])
        yN_aug.append(image_names[i])
        x_aug.append(img_trans(image_data[i]/255.))
        y_aug.append(image_labels[i])
        yN_aug.append(image_names[i])
    return x_aug, y_aug, yN_aug


# In[ ]:


dumX_aug, dumY_aug, dumN_aug = data_augmentation(dummyX, dummyY, dummyN)


# #### display augmented data - only for first 2 pairs of images

# In[ ]:


def display_augdata(image_data, image_names, batch_size):
    image_data1 = image_data[0:batch_size*2]
    image_names1 = image_names[0:batch_size*2]
    fig, axes = plt.subplots(batch_size, batch_size, figsize=(20,20))
    axes = axes.flatten()
    i=0
    for img, ax in zip(image_data1, axes):
        ax.set_title(image_names1[i], fontsize=36)
        ax.imshow(img)
        ax.set_xticks(())
        ax.set_yticks(())
        i=i+1
    plt.tight_layout()
display_augdata(dumX_aug, dumN_aug, 2)


# In[ ]:


def DataGen(train_data_X, train_data_Y, train_data_N, im_size, batch_size, shuffle=True, data_aug = True, test = False):
    nobs = len(train_data_N)
    iter_times = nobs/batch_size
    train_data_X.list_IDs = np.arange(0,data.shape[0])
    if shuffle=True:
        


# In[ ]:


# Parameters
params_trn = {
          'im_size': im_size,
          'batch_size': batch_size,
          'shuffle': True,
          'data_augment' : True,
          'test' : False
         }
params_val = {
          'im_size': im_size,
          'batch_size': batch_size,
          'shuffle': True,
          'data_augment' : False,
          'test' : False
         }

# Generators
training_generator = DataGenerator(x_train_raw, y_train_raw, **params_trn)
validation_generator = DataGenerator(x_val_raw, y_val_raw, **params_val)

