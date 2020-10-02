#!/usr/bin/env python
# coding: utf-8

# In this notebook i will show how to load, process and organize the captcha dataset images & their labels <br/>
# Additional code and notebooks are avaliable on my repository: https://github.com/Vykstorm/CaptchaDL

# ## Import statements

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
from re import match
from itertools import product, count, chain
from keras.utils import to_categorical

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get captcha texts

# 
# 
# List all files in the dataset.
# 

# In[36]:


images = os.listdir('../input/samples/samples')
images[0]


# 
# 
# Select image files
# 

# In[37]:


images = list(filter(lambda image: match('^[a-z0-9]+\..+$', image), images))


# In[38]:


len(images)


# 
# 
# Extract the captcha texts from the file names
# 

# In[39]:


texts = [match('^([a-z0-9]+)\..+$', image).group(1) for image in images]


# Make sure all captchas have a fixed size of 5 tokens

# In[40]:


all([len(text) == 5 for text in texts])


# Get all unique characters that appears once in the captchas

# In[41]:


alphabet = list(frozenset(chain.from_iterable(texts)))
alphabet.sort()
''.join(alphabet)


# Assign a unique integer id label for each character in the alphabet

# In[42]:


ids = dict([(ch, alphabet.index(ch)) for ch in alphabet])
ids['b']


# Now we are going to to create a 2D array ('y_labels') of size num images (n) x captcha text size (m)
# 
# \begin{bmatrix}
# yl_0^0 & yl_0^1 & yl_0^2 & ... & yl_0^m \\
# yl_1^0 & yl_1^1 & yl_1^2 & ... & yl_1^m \\
# & & ... & \\
# yl_n^0 & yl_n^1 & yl_n^2 & ... & yl_n^m
# \end{bmatrix}
# 
# Where the element $yl_i^j$ its the integer label for the jth character on the ith captcha image

# In[43]:


n, m = len(texts), 5
y_labels = np.zeros([n, m], dtype=np.uint8)
for i, j in product(range(0, n), range(0, m)):
    y_labels[i, j] = ids[texts[i][j]]
y_labels[0]


# Now we turn y_labels to a 3D matrix ('y') of size num images x text size x alphabet size. <br/>
# $y_i^j$ (y[i, j, :]) is a sparse vector filled by zeros except the element at kth position where k is the integer label of the jth character on the ith captcha image <br/>
# 
# $y_i^j =
# \begin{bmatrix}
# yl_i^{j,0} & yl_i^{j, 1} & yl_i^{j, 2} & ... & yl_i^{j, s}
# \end{bmatrix} =
# \begin{bmatrix}
# 0 & 0 & ... & 1 & ... & 0 & 0
# \end{bmatrix}$
# 
# $s$ is the alphabet size

# In[44]:


y = np.zeros([n, m, len(alphabet)], dtype=np.uint8)
for i, j in product(range(0, n), range(0, m)):
    y[i, j, :] = to_categorical(y_labels[i, j], len(alphabet))


# In[45]:


y[0, 0, :]


# In[46]:


y.shape


# Note that $yl_i^j = argmax_k(y_i^{j,k})$

# In[47]:


np.all((y_labels == y.argmax(axis=2)).flatten())


# ## Get captcha images

# We are going to store all the images (grayscaled) in a 4D matrix of size: num images x image height x image width x 1 with float32 dtype with values in the range [0, 1]

# In[48]:


X = np.zeros((n,) + (50, 200, 1), dtype=np.float32)
for i, filename in zip(range(0, n), images):
    img = cv.cvtColor(cv.imread('../input/samples/samples/' + filename), cv.COLOR_BGR2GRAY)
    assert img.shape == (50, 200)
    X[i, :, :, 0] = img.astype(np.float32) / 255


# In[49]:


plt.imshow(X[10, :, :, 0], cmap='gray'), plt.xticks([]), plt.yticks([]);


# ## Save data

# Save the variables we defined here to use later in other kernels

# In[50]:


np.savez_compressed('preprocessed-data.npz', X=X, y=y, y_labels=y_labels, alphabet=alphabet)

