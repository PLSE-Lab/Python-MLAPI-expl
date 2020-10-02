#!/usr/bin/env python
# coding: utf-8

# # Introduction
# * This is part 3 for this competitions submit

# ##  Part1 - Simple tansfer learning to detect ship exist
# * https://www.kaggle.com/super13579/simple-transfer-learning-detect-ship-exist-keras

# ## Part2 - U-net base on ResNet34 transfer learning 
# * https://www.kaggle.com/super13579/u-net-base-on-resnet34-transfer-learning-keras

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '../input/airbus-ship-detection/'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
import gc; gc.enable() # memory is tight

from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


from keras import models, layers
fullres_model = models.load_model('../input/u-net-base-on-resnet34-transfer-learning-keras/seg_model.h5', compile=False)
seg_in_shape = fullres_model.get_input_shape_at(0)[1:3]
seg_out_shape = fullres_model.get_output_shape_at(0)[1:3]
print(seg_in_shape, '->', seg_out_shape)


# ## Plot Test Image by part-2 trained model

# In[ ]:


test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')


# In[ ]:


fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    first_seg = fullres_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title(c_img_name)
    ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')


# In[ ]:


#Debug use
"""
from skimage.morphology import binary_opening, disk
c_path = os.path.join(test_image_dir, '8a56c9bdd.jpg')
c_img = imread(c_path)
first_img = np.expand_dims(c_img, 0)/255.0
first_seg = fullres_model.predict(first_img)[0]
first_seg = binary_opening(first_seg>0.5, np.expand_dims(disk(2), -1))
"""


# ## Use Part-1 result to seperate have ship and no ship data

# In[ ]:


have_ship= pd.read_csv("../input/simple-transfer-learning-detect-ship-exist-keras/Have_ship_or_not.csv")


# In[ ]:


have_ship.head()


# In[ ]:


test_names = have_ship.loc[have_ship['Have_ship'] > 0.5, ['ImageId']]['ImageId'].values.tolist()
test_names_nothing = have_ship.loc[have_ship['Have_ship'] <= 0.5, ['ImageId']]['ImageId'].values.tolist()
len(test_names), len(test_names_nothing)


# ## No ship data to submission file

# In[ ]:


ship_list_dict = []
for name in test_names_nothing:
    ship_list_dict.append({'ImageId':name,'EncodedPixels':None})


# ## Have ship data to submission file

# In[ ]:


from tqdm import tqdm_notebook
from skimage.morphology import binary_opening, disk
for c_img_name in tqdm_notebook(test_paths):
    if c_img_name in test_names:
        c_path = os.path.join(test_image_dir, c_img_name)
        c_img = imread(c_path)
        c_img = np.expand_dims(c_img, 0)/255.0
        cur_seg = fullres_model.predict(c_img)[0]
        cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
        cur_rles = multi_rle_encode(cur_seg)
        if len(cur_rles)>0:
            for c_rle in cur_rles:
                ship_list_dict += [{'ImageId': c_img_name, 'EncodedPixels': c_rle}]
        else:
            ship_list_dict += [{'ImageId': c_img_name, 'EncodedPixels': None}]
    gc.collect()
    


# In[ ]:


submission_df = pd.DataFrame(ship_list_dict)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('submission.csv', index=False)
submission_df.sample(3)

