#!/usr/bin/env python
# coding: utf-8

# # PANDA 256x256 crops dataset
# As the images are quite large in the dataset for this competition, here I show a fast way to crop them to form 256x256 tiles and discard "white tiles".
# 
# **Version 1.** Provides code to create tiles from train images and saves tiles to `/kaggle/working/data256`
# 
# **Some aspects that may need to be improved:** 
# * The `crop_and_tile` function is slicing up to 255 pixels out at the end of each image dimension to make the image size multiple of 256. I didn't check if any "good" pixels are being removed in this step.
# * Some regions of interest may turn out to be in the margin or corner of the image. This may not be ideal, one possibility is to generate tiles with some overlap. 
# 
# **Note:** So far I have no idea if this approach will be any good, it's just my first guess on this competition.

# In[ ]:


from fastai.vision import *
import skimage.io
import zipfile


# In[ ]:


def crop_and_tile(fn, tiff_layer=2, empty_thr=250, tile_thr=0):
    im = skimage.io.MultiImage(str(fn))[tiff_layer]
    crop = tile_size*(im.shape[0]//tile_size), tile_size*(im.shape[1]//tile_size)
    im = im[:crop[0], :crop[1]]
    imr = im.reshape(im.shape[0]//tile_size,tile_size,im.shape[1]//tile_size, tile_size, 3)
    imr = imr.transpose(1,3,0,2,4)
    imr = imr.reshape(imr.shape[0], imr.shape[1], imr.shape[2]*imr.shape[3], imr.shape[4])
    imr = imr.transpose(2,0,1,3)
    not_empty = np.array([(im[...,0]<empty_thr).sum() for im in imr])>tile_thr
    return imr[not_empty]

def save_tiles(path:Path, filename:str, tiles):
    for i, t in enumerate(tiles):
        im = PIL.Image.fromarray(t)
        im.save(path/f'{filename}_{i}.png')


# In[ ]:


tile_size = 256
path = Path('/kaggle/input/prostate-cancer-grade-assessment')
save_path = Path(f'/kaggle/working/data{tile_size}')
save_path.mkdir(exist_ok=True)
train_folder = 'train_images'
masks_folder = 'train_label_masks'
path.ls()


# In[ ]:


# Load train.csv
train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


files = (path/train_folder).ls()

def do_one(fn, *args):
    tiles = crop_and_tile(fn)
    save_tiles(save_path, fn.stem, tiles)
    
parallel(do_one, files, max_workers=4)


# In[ ]:


# Plot some samples
saved_files = np.random.permutation(save_path.ls())
fig, axes = plt.subplots(ncols=16,nrows=16,figsize=(12,12),dpi=120,facecolor='gray')
for ax, fn in zip(axes.flat, saved_files):
    im = PIL.Image.open(fn)
    ax.imshow(np.array(im))
    ax.axis('off')

