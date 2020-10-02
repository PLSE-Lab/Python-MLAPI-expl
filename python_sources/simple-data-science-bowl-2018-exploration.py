#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This kernel is inspired by [this one](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277)
# 
# # Understanding problem, data and simple exploration
# 1. Particular cases
#     * First training image
#     * Second training image
# 2. Preprocessing all data
# 3. Run-length encoding

# ## 1. Particular cases

# In[ ]:


train_dir = '../input/stage1_train/'
test_dir = '../input/stage1_test/'
first_image_id = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
second_image_id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
first_path = train_dir + first_image_id
second_path = train_dir + second_image_id


# In[ ]:


first_img = imread(first_path + '/images/' + first_image_id + '.png')
second_img = imread(second_path + '/images/' + second_image_id + '.png')

print('first_img => (height, width, channels)=', first_img.shape)
print('second_img => (height, width, channels)=', second_img.shape)


# Grayscale images are encoded from 0 (black) to 255 (white) and between then we have intermediate colors ranging from totally black to totally white. In numpy, there are a type to hold this range of integers.
# 
# [np.unit8](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html) => Unsigned integer (0 to 255)

# ---
# Beacuse there are **4** channels in these **RGB** images, let's explore the last channel in these particular cases
# 

# In[ ]:


# show last channel of the two images
imshow(first_img[:, :, 3])
plt.show()
imshow(second_img[:, :, 3])
plt.show()


# In[ ]:


np.all(first_img[:, :, 3] / 255)


# The last channel is empty (totally white = 255), therefore we can ignore it.
# 
# Latter in this notebook, we check if last channel in all training set images are empty
# ___

# Let's define two type of mask, one (**mask_sum**) superimpose each individual mask to form one master mask and another (**mask_max**) picking the maximun element between the partial master mask and the individual mask

# In[ ]:


first_mask_sum = np.zeros((256, 320, 1), dtype=np.uint8)
first_mask_max = np.zeros((256, 320, 1), dtype=np.uint8)
second_mask_sum = np.zeros((256, 256, 1), dtype=np.uint8)
second_mask_max = np.zeros((256, 256, 1), dtype=np.uint8)

# mask for first image
for mask_file in next(os.walk(first_path + '/masks/'))[2]:
    mask = imread(first_path + '/masks/' + mask_file)
    first_mask_max = np.maximum(first_mask_max, mask[:, :, np.newaxis])
    first_mask_sum += mask[:, :, np.newaxis] # because there are no overlapping
    
# mask for second image
for mask_file in next(os.walk(second_path + '/masks/'))[2]:
    mask = imread(second_path + '/masks/' + mask_file)
    second_mask_max = np.maximum(second_mask_max, mask[:, :, np.newaxis])
    second_mask_sum += mask[:, :, np.newaxis] # because there are no overlapping


# In[ ]:


print('first_mask => (height, width, channels)=', first_mask_sum.shape)
print('second_mask => (height, width, channels)=', second_mask_sum.shape)


# 

# In[ ]:


# showing first image and masks
imshow(first_img)
plt.show()
imshow(first_mask_sum[:, :, 0])
plt.show()
imshow(first_mask_max[:, :, 0])
plt.show()


# In[ ]:


# showing second image and masks
imshow(second_img)
plt.show()
imshow(second_mask_sum[:, :, 0])
plt.show()
imshow(second_mask_max[:, :, 0])
plt.show()


# In[ ]:


# sanity check for first_mask_sum
first_mask_normalized = first_mask_sum / 255
zeros = (first_mask_normalized[first_mask_normalized == 0] + 1).sum()
ones = first_mask_normalized[first_mask_normalized == 1].sum()
zeros + ones == 256 * 320 # height x width
# i.e. there are only zeros and ones, no overlapping


# In[ ]:


# sanity check for second_mask_sum
second_mask_normalized = second_mask_sum / 255
zeros = (second_mask_normalized[second_mask_normalized == 0] + 1).sum()
ones = second_mask_normalized[second_mask_normalized == 1].sum()
zeros + ones == 256 * 256 # height x width
# i.e. there are only zeros and ones, no overlapping


# Latter in this notebook, we can check if this property is true for all training set masks
# ___

# Resizing 256x320 image to 256x256

# In[ ]:


first_img_resized = resize(first_img, (256, 256, 4), preserve_range=True, mode='constant')
first_img_resized = first_img_resized.astype(dtype=np.uint8)
imshow(first_img_resized)
plt.show()
imshow(first_img)
plt.show()


# The resized image looks good
# ___

# ## 2. Preprocessing all data

# In[ ]:


train_images_ids = os.listdir(train_dir)
train_m = len(train_images_ids) # number of training examples
test_images_ids = os.listdir(test_dir)
test_m = len(test_images_ids)
print('(Training examples, Test examples):', '(', train_m, ',', test_m, ')')


# In this part of the notebook, we'll do some sanity check to proof is all training set images have the same properties: shape, last channel, masks...

# In[ ]:


d = dict()
last_channel_empty = []
for i, ids in enumerate(train_images_ids):
    path = train_dir + ids
    img = imread(path + '/images/' + ids + '.png')
    last_channel_empty.append( np.all(img[:, :, 3] / 255) )
    shape = img.shape
    if (shape in d.keys()):
        d[shape] += 1
    else:
        d[shape] = 1

# print all images shape
for shape in d.keys():
    print(shape, ':', d[shape])
if np.all(last_channel_empty):
    print('Last channel empty in all images')
else:
    print('Last channel not empty in some images')


# Beacuse there are several shapes, we need to resize all images into one common shape.
# 
# All images have 4 channels but the last channel is empty, therefore, we can delete it.

# In[ ]:


# Let's define the common shape to resize all images
height, width, channels = 256, 256, 3
print('height, width, channels:', height, width, channels)


# In[ ]:


X_train = np.zeros((train_m, height, width, channels), dtype=np.uint8)
Y_train = np.zeros((train_m, height, width, 1), dtype=np.uint8)

print(train_m, 'examples =>', end=' ')
for i, ids in enumerate(train_images_ids):
    if (i%10==0):
        print(i, end=',')
    path = train_dir + ids
    img = imread(path + '/images/' + ids + '.png')[:, :, :channels]
    if (img.shape != (height, width, channels)): # resize it
        img = resize(img, (height, width, channels), mode='constant', preserve_range=True)
        img = img.astype(np.uint8)
    X_train[i] = img
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    for mask_file in os.listdir(path + '/masks/'):
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = mask_[:, :, np.newaxis]
        if (mask_.shape != (height, width, 1)):
            mask_ = resize(mask_, (height, width, 1), mode='constant', preserve_range=True)
            mask_ = mask_.astype(np.uint8)
        mask += mask_
    Y_train[i] = mask
print('Done')


# In[ ]:


X_test = np.zeros((test_m, height, width, channels), dtype=np.uint8)
sizes_test = [] # latter, for submit results, we need the original shapes

print(test_m, 'examples =>', end=' ')
for i, ids in enumerate(test_images_ids):
    if (i%5==0):
        print(i, end=',')
    path = test_dir + ids
    img = imread(path + '/images/' + ids + '.png')[:, :, :channels]
    sizes_test.append(img.shape)
    if (img.shape != (height, width, channels)):
        img = resize(img, (height, width, channels), mode='constant', preserve_range=True)
        img = img.astype(np.uint8)
    X_test[i] = img
print('Done')


# In[ ]:


print('Shape X_train:', X_train.shape)
print('Shape Y_train:', Y_train.shape)
print('Shape X_test:', X_test.shape)


# In[ ]:


# Normalice inputs
X_train = X_train / 255 # values ranging form 0 to 1
Y_train = Y_train / 255 # only contains 0 or 1
X_test = X_test / 255    # values ranging form 0 to 1


# Now we can train our [U-net](https://arxiv.org/pdf/1505.04597.pdf) model over X_train and test it over X_test to submit results in LB.

# ## 3. Run-length encoding
# I've used the implementation done in https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

# In[ ]:


# Run-length encoding
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# In[ ]:


#threshold = 0.5
#y_pred = model.predict(X_test)
# resize all y_pred images to its original shape.
#y_pred_resize = []
#for i in range(len(y_pred)):
#    y_pred_resize.append(resize(y_pred[i], (sizes_test[i][0], sizes_test[i][1]), 
#                                mode='constant', preserve_range=True))
#y_pred_t = y_pred_resize > threshold # only 0's and 1's


# In[ ]:


# over y_pred_t, apply run length encoding to submit results
#new_test_ids = []
#rles = []
#for n, id_ in enumerate(test_images_ids):
#    rle = list(prob_to_rles(y_pred_resize[n]))
#    rles.extend(rle)
#    new_test_ids.extend([id_] * len(rle))


# In[ ]:


# create submission DataFrame
#sub = pd.DataFrame()
#sub['ImageId'] = new_test_ids
#sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
#sub.to_csv('predictions.csv', index=False)

