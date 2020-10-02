#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import timeit
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from scipy.sparse import bsr_matrix
import dask.array as da
import dask
import dask.diagnostics as diag
from multiprocessing.pool import ThreadPool
import h5py
from bokeh.io import output_notebook
from bokeh.resources import CDN
import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import signal
import cv2
from PIL import Image
import pdb
from tqdm import tqdm
from glob import glob

import warnings
warnings.filterwarnings("ignore")

output_notebook(CDN, hide_banner=True)

ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
import gc; gc.enable() # memory is tight

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
    img = np.zeros(shape[0]*shape[1], dtype=np.bool)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    count = 1
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += count * rle_decode(mask)
            count += 1
    return np.expand_dims(all_masks, -1)

def get_file_size(is_train, id):
    if is_train: 
        return os.stat(os.path.join(train_image_dir, id)).st_size/1024
    return os.stat(os.path.join(test_image_dir, id)).st_size/1024

def get_file_path(is_train, id):
    if is_train: 
        return os.path.join('train', id)
    return os.path.join('test', id)
    
def get_ship_size_and_sparse_image(df):
    if not isinstance(df['EncodedPixels'], str):
        df['size'] = 0
        #df['bsr_matrix'] = bsr_matrix((768, 768), blocksize=[6, 6], dtype=int)
        return df
    decoded = rle_decode(df['EncodedPixels'], shape=(768, 768))
    bsr = bsr_matrix(decoded, blocksize=[6,6], shape=(768, 768), dtype=int)
    
    # maybe, sometime in the future, i want it back - but for now - this version does not store the bsr_matrix
    #df['bsr_matrix'] = bsr_matrix(decoded, blocksize=[6,6], shape=(768, 768), dtype=int)
    df['size'] = bsr.sum()
    return df

def aggregate_and_merge(df, what = 'size', do = np.sum, name = 'total'):
    temp = df.groupby('ImageId').agg({what: do}).reset_index()
    temp = pd.DataFrame(temp).rename(index=str, columns={what: name})
    df = df.merge(temp, how='left', on='ImageId')
    df[name] = df[name].fillna(0).astype(int)
    return df

def dask_read_seg(in_batches, max_items = None):
    d_mask_fun = dask.delayed(masks_as_image)
    if max_items is None:
        max_items = len(in_batches)
    lazy_images = [d_mask_fun(c_masks['EncodedPixels'].values) 
                   for _, (_, c_masks) in zip(range(max_items), in_batches)
                  ]     # Lazily evaluate on each group
    s_img = lazy_images[0].compute()
    arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                              dtype=s_img.dtype,   # for every lazy value
                              shape=s_img.shape)
              for lazy_image in lazy_images]

    return da.stack(arrays, axis=0)                # Stack all small Dask arrays into one


def get_domimant_colors(fname, top_colors=2):
    img = cv2.imread(os.path.join(ship_dir, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (32, 32))
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    
    return clt.cluster_centers_, hist

def get_colors_and_merge(df):
    details = []
    files = df['file_path'].values
    
    for imfile in files:
        dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(imfile, top_colors=1)
        dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] 
                                                          * dominant_colors_hsv.shape[1])
        details.append(dominant_colors_hsv.squeeze())
    
    kmeans = KMeans(n_clusters=10).fit(details)
    df['dominant_colors'] = details
    df['color_cluster'] = kmeans.predict(details)
    
    return df
    




# # What is this Kernal all about?
# Inspired by the Kernel from [Kevin Mader](https://www.kaggle.com/kmader/package-segmentation-images) i decided to put together an updated Version, with a few usefull additional features. 
# In addition to the repacking of the test and train dataset, i added the dominant colors of each image. I think that could help classify images into "boat" and "no-boat". For that purpose, i just rip code from the great Kernel from [Costas Voglis](https://www.kaggle.com/voglinio/airbus-ship-detection-clustering-no-ship-images/notebook)
# 
# It's still a work in Progress, but it contains the complete dataset. 
# 
# If you didn't have heard already, the annotations for the test-set got released. [Read all about the data-leak, and what happens now](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388)
# 
# Well, now get to the good stuff. I changed Kevins code a bit, so that the exported masks all have different numbers - i think its best, if you can reconstruct the original masks without some heavy postprocessing. So keep in mind that the generated masks are not binary anymore. For each mask i'll add to the array, i count one up. 
# 
# Besides that i thought it would be nice to have next to the count of found ships in the image, the sizes in pixel, and for every picture the median an std of the ship-sizes. This way it should be easy to balance the training-data for small and/or large vessels.
# 
# Because i'm lazy, if also put the image-filepath into the csv file, so that i dont have to worry anymore if an image is train or (former)-test.
# 
# Puh, some last addition: I've filtered a few corrupted images out ,-)
# But lets start.. with some code... 
# 
# 
# *please wait while loading...*

# In[ ]:


# load train data and mark them as train 
masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations.csv'))
masks['train'] = True

# load test data and tell every entry that it is not train data (at least not originaly ,-)
# in case you wonder, we need that information - so that we know where to load the images
append = pd.read_csv(os.path.join(ship_dir, 'test_ship_segmentations.csv'))
append['train'] = False

# do i need to comment this? (it gets late, and i need to finish this, i'm not THAT funny - normaly ,-)
masks = masks.append(append)

# wuhhh, nice - i don't need to check anymore where i get those images
masks['file_path'] = masks.apply(lambda x: get_file_path(x['train'], x['ImageId']), axis=1)


# Ok, **feature compiling** - I'm very eager to know how long that did take: 

# In[ ]:


get_ipython().run_cell_magic('time', '', "# here we generate ship_count\nmasks['has_ship'] = masks['EncodedPixels'].map(lambda x: 1 if isinstance(x, str) else 0)\nmasks = aggregate_and_merge(masks, what = 'has_ship', name = 'ship_count')\n\n# with the filesize, i can filter images that are corrupted\nmasks['file_size_kb'] = masks.apply(lambda x: get_file_size(x['train'], x['ImageId']), axis=1)\nmasks = masks[masks['file_size_kb'] > 50] # keep only +50kb files\n\n# shipsize - ahaaand... a blockwise sparse matrix \n# (which i used to calculate the size, a tiny little bit faster)\n# but besides that, it turned out to be useless - RLE encoding an decoding is way (not that much) - way faster\n# as i said, i'm lazy - so i keep it \nmasks = masks.apply(get_ship_size_and_sparse_image, axis=1)\n\n# here we have the total count of target-pixels, or total_ship_size for each image \nmasks = aggregate_and_merge(masks, what = 'size', name = 'total_ship_size')\n\n# with the total_ship_size, and the ship_count - its easy for you to get to the mean \n# but! sometimes, its more interesting, to look if there are outliers in size - or have a feature\n# that isn't that vulnerable for them - hence: here i present, the median_ship_size\nmasks = aggregate_and_merge(masks, what = 'size', do = np.median, name = 'median_ship_size')\n\n# did i mention that it could be interesting to know if there are outliers, to plan your training?\n# well, i think, the standard derivation can also help\nmasks = aggregate_and_merge(masks, what = 'size', do = np.std, name = 'std_ship_size')\nmasks.fillna(0)")


# So, we are done for the day. If you are interested in all the data - even if i did only use them for a short time..
# and even the *author* does not exactly know what to do with them... 
# 
# You find a generated file named **train.csv** containing at least one entry per image, but also one for every mask, if you click on the output-tab.
# 
# Well, then take a peak at some *df.sample()*

# In[ ]:


# somehow the order got mixed up
# if anyone finds where in my code that happend - i would be grateful for a hint
masks = masks.set_index('ImageId').reset_index()
masks.to_csv('train.csv', index=False)

# since EncodedPixels are so long, we change the display-setting 
pd.set_option('max_colwidth', 15)
pd.set_option('display.precision', 2)

masks.sample(5)


# Okay, i said i would keep all the features - but i dont know - RAM and diskspace - isn't *that* cheap 
# 
# So if you are interested in the good stuff, but not the strange stuff i worked with.. 
# 
# In contrast to the full train.csv, the other generated file only contains one entry per image.
# 
# Go and fetch the **reduced_train.csv** from the output-tab, right after you had a look:

# In[ ]:


# preparing a bit early, because we are going to drop EncodedPixels, which we need for Kevins code 
all_batches = list(masks.groupby('ImageId'))

# as i said, here comes the sugar 
masks.drop_duplicates(['ImageId'], inplace=True)
masks.drop(['has_ship', 'file_size_kb', 'train', 'EncodedPixels', 'size'], axis=1, inplace=True)
masks.to_csv('reduced_train.csv', index=False)

masks.sample(5)


# Did i say reduced? Lets add Color-Features... 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'masks = get_colors_and_merge(masks)')


# If you like these features, grab the file **color_train.csv**
# 
# You know the drill, lets have a look:

# In[ ]:


masks.to_csv('color_train.csv', index=False)
masks.sample(5)


# It is time to let [Kevin](https://www.kaggle.com/kmader) have the last words..  because, the last bits of this kernel are completly his work. I think it was a nice and convinient idea to package all masks into one file. 
# 
# Thank you Kevin, your Kernels are always a good and inspirational way to learn. 
# 
# As Kevin in the original Kernel said: 
# 
# 
# > **Now Package Everything**
# 
# > instead of just using a small portion of the dataset we export all the results.

# In[ ]:


# larger chunks are more efficient for writing/compressing and make the paralellization more efficient
larger_chunker = lambda x: x.rechunk({0: x.shape[0]//400, 1: -1, 2: -1, 3: -1})
all_img_ds = larger_chunker(dask_read_seg(all_batches))

with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(4)):
        all_img_ds.to_hdf5('segmentions.h5', '/image', compression = 'lzf')

