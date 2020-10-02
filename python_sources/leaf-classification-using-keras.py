#!/usr/bin/env python
# coding: utf-8

# **<h1>LEAF CLASSIFICATION</h1>**

# In[ ]:


# Package Imports
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Keras stuff
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

root = '../input'
np.random.seed(2016)
split_random_state = 7
split = .9
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def load_numeric_training(standardize=True):
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values
    return ID, X, y


# In[ ]:


def load_numeric_test(standardize=True):
    test_data = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = data.pop('id')

    test = StandardScaler().fit(test_data).transform(test_data) if standardize else test_data.values
    return ID, test


# In[ ]:


def resize_img(img, max_dim=96):
    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


# In[ ]:


# Original Image
original_image = load_img(os.path.join(root, 'images', '1'+'.jpg'), grayscale=True)
plt.imshow(original_image)


# In[ ]:


# Resized Image
plt.imshow(resize_img(original_image))


# In[ ]:


def load_image_data(ids, max_dim=96, center=True):
    X = np.zeros((len(ids), max_dim, max_dim, 1))
    for i, ide in enumerate(ids):
        x = resize_img(load_img(os.path.join(root, 'images', str(ide)+'.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        
        length = x.shape[0]
        width = x.shape[1]
        
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
            
        X[i, h1:h2, w1:w2, 0:1] = x
        
    return np.around(X / 255.0)


# In[ ]:


def load_train_data(split=split, random_state=None):
    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(X_num_tr, y))
    X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


# In[ ]:


def load_test_data():
    # Load the pre-extracted features
    ID, X_num_te = load_numeric_test()
    # Load the image data
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te


# In[ ]:


print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print('Training data loaded!')


# In[ ]:




