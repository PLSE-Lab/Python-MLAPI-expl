#!/usr/bin/env python
# coding: utf-8

# We combine here the 2 best public models:
# 
# The U-Net segmentation from: https://www.kaggle.com/hmendonca/u-net-model-with-submission
# 
# and reduce the false positives with the classification CNN: https://www.kaggle.com/kmader/transfer-learning-for-boat-or-no-boat (Thanks Kevin!)
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))


# In[ ]:


boat_df = pd.read_csv("../input/transfer-learning-for-boat-or-no-boat/submission.csv")
boat_df.head()


# In[ ]:


is_boat = boat_df.EncodedPixels.notnull()
print('Found {} boats'.format(is_boat.sum()))


# In[ ]:


from keras.models import load_model
fullres_model = load_model('../input/u-net-model-with-submission/fullres_model.h5')


# In[ ]:


from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight

ship_dir = '../input/airbus-ship-detection/'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

def predict(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = fullres_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg > 0.9, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]


# In[ ]:


from tqdm import tqdm_notebook

out_pred_rows = []
for c_img_name in tqdm_notebook(boat_df.ImageId[is_boat]):
    out_pred_rows += pred_encode(c_img_name, min_max_threshold=1.0)


# In[ ]:


sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]
sub.head()


# In[ ]:


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

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks


# In[ ]:


## let's see what we got
TOP_PREDICTIONS = 6
fig, m_axs = plt.subplots(TOP_PREDICTIONS, 3, figsize = (14, TOP_PREDICTIONS*4))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, c_img[0]

for (ax1, ax2, ax3), c_img_name in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):
    pred, c_img = raw_prediction(c_img_name)
    ax1.imshow(c_img)
    ax1.set_title('Image: ' + c_img_name)
    ax2.imshow(pred[...,0], cmap=get_cmap('jet'))
    ax2.set_title('Prediction')
    ax3.imshow(masks_as_color(sub.query('ImageId==\"{}\"'.format(c_img_name))['EncodedPixels']))
    ax3.set_title('Masks')


# In[ ]:


sub1 = pd.read_csv('../input/airbus-ship-detection/sample_submission_v2.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None
print(len(sub1), len(sub))

sub = pd.concat([sub, sub1])
print(len(sub))
sub.to_csv('blended_submission.csv', index=False)
sub.head()

