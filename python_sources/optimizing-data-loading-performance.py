#!/usr/bin/env python
# coding: utf-8

# ## Data loading performance comparison
# 
# The massive dataset in this competition means that data loading is a significant portion of training time. In this notebook, I've compared 3 different approaches for loading the data:
# 
# * Just load the provided PNG files without any preprocessing
# * Significantly compress the data by rendering images as RGB jpeg's (while maintaining image dimensions.)
# * Converting the raw PNGs to preprocessed numpy arrays
# 
# Converting the images to JPEGs as described speeds up data loading by a factor of ~3. That's obviously great, but I'm concerned that it signficantly reduces model performance because so much data is lost by the compression. The JPEG images are 13 times smaller on disk than the originals, and while the JPEG algorithm throws out high frequencies that aren't perceptible to humans, I don't know how losing those frequencies affects a CNN.
# 
# As an alternative, we can convert the images to numpy arrays. That leads to a much smaller speed up - about 25% - but it doesn't lead to any data loss. In that sense, it's a free lunch. 25% isn't huge, but with training times on the order of 10 hours, it's also nothing to sneeze at.
# 
# As a final note, I've stitched together code from three authors with my own additions, so apologies for the variety of styles and naming conventions.

# In[ ]:


import os
from os import sys

import imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

# Create a dataframe including one experiment's worth of samples
df = pd.read_csv('../input/train.csv').head(1106)  # HEPG-01


# In[ ]:


# First case: don't do any preprocessing
# Modeled on https://www.kaggle.com/leighplt/densenet121-pytorch
# This also serves as a base class for the datasets in cases 2 and 3.
class ImageDS(D.Dataset):
    def __init__(self, df, img_dir, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.img_dir = img_dir
        self.len = df.shape[0] * 2
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, site, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,experiment,f'Plate{plate}',f'{well}_s{site}_w{channel}.png'])
    
    def _load_data(self, index, site):
        paths = [self._get_img_path(index, site, ch) for ch in self.channels]
        # Although we're normalizing here, the computational cost is insignificant
        normalize = T.Normalize(
            mean=[0.5] * 6,
            std=[0.5] * 6
        )
        return normalize(torch.cat([self._load_img_as_tensor(img_path) for img_path in paths]))
    
    def __getitem__(self, index):
        site = (index % 2) + 1
        index = index // 2
        return self._load_data(index, site)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
def loop(data_loader):
    for _ in data_loader:
        pass
    


# In[ ]:


# Note that we're loading images directly from the input folder.
ds = ImageDS(df, '../input/train')
loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
get_ipython().run_line_magic('timeit', '-r 3 loop(loader)')


# Loading and preprocessing a whole folder of images takes about a minute. That's not-so-great. Let's compare that to converting our 6-channel image sets to 3-channel JPEGs of the same dimension:

# In[ ]:


get_ipython().system('git clone https://github.com/recursionpharma/rxrx1-utils')
get_ipython().system('mkdir -p processed-data/jpg')
sys.path.append('rxrx1-utils')
import rxrx.io as rio


# In[ ]:


# Option two: Convert to an rgb jpg
# Modified version of https://www.kaggle.com/xhlulu/recursion-2019-load-size-and-resize-images

def convert_to_rgb(df, img_dir='processed-data/jpg/', resize=False, new_size=224, extension='jpeg'):
    N = df.shape[0]
    for i in tqdm(range(N)):
        code = df['id_code'][i]
        experiment = df['experiment'][i]
        plate = df['plate'][i]
        well = df['well'][i]
        for site in [1, 2]:
            save_path = f'{img_dir}{code}_s{site}.{extension}'

            im = rio.load_site_as_rgb(
                'train', experiment, plate, well, site, 
                base_path='../input/'
            )
            im = im.astype(np.uint8)
            im = Image.fromarray(im)
            
            if resize:
                im = im.resize((new_size, new_size), resample=Image.BILINEAR)
            im.save(save_path)

class JpgImageDS(ImageDS):
    def __init__(self, df, img_dir):
        super().__init__(df, img_dir)
        
    def _get_img_path(self, index, site):
        code = self.records[index].id_code
        return f'{self.img_dir}{code}_s{site}.jpeg'
    
    def _load_data(self, index, site):
        return self._load_img_as_tensor(self._get_img_path(index, site))


# In[ ]:


convert_to_rgb(df)


# In[ ]:


# Option two: Load jpegs
ds = JpgImageDS(df, 'processed-data/jpg/')
loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
get_ipython().run_line_magic('timeit', '-r 3 loop(loader)')


# As I mentioned, this loop is **a lot** faster:  ~20 seconds down from ~60, for a 3x improvement. But it's not a faster way to load data per se. If anything, the need to uncompress the jpeg makes it slower. Instead, we've just reduced the data size by a factor of ~13, from 1.6GB to 126MB:

# In[ ]:


get_ipython().system('du -sh ../input/train/HEPG2-01/')
get_ipython().system('du -sh processed-data/jpg/')


# (The 1.6GB size actually understates things: The images are ~3.6gb unzipped.)
# 
# Let's see if we can improve performance without throwing out data:

# In[ ]:


# Option 3: Create preprocessed numpy files.
# Code mostly from https://www.kaggle.com/gidutz/starter-kernel-recursion-pharmaceuticals

BASE_DIR = '../input'
OUTPUT_DIR = 'processed-data/npy/'
DATA_PATH_FORMAT = os.path.join(BASE_DIR, 'train/{experiment}/Plate{plate}/{well}_s{sample}_w{channel}.png')

df_pixel_stats = pd.read_csv(os.path.join(BASE_DIR, 'pixel_stats.csv')).set_index(['id_code','site', 'channel'])

def transform_image(sample_data, pixel_data, site):
    x=[]
    for channel in [1,2,3,4,5,6]:
        impath = DATA_PATH_FORMAT.format(experiment=sample.experiment,
                                        plate=sample_data.plate,
                                        well=sample_data.well,
                                        sample=site,
                                        channel=channel)
        # normalize the channel
        img = np.array(imageio.imread(impath)).astype(np.float64)
        img -= pixel_data.loc[channel]['mean']
        img /= pixel_data.loc[channel]['std']
        img *= 255 # To keep MSB

        x.append(img)

    return np.stack(x).T.astype(np.byte)


get_ipython().system('mkdir -p {OUTPUT_DIR}')
for _, sample in tqdm(df.iterrows(), total=len(df)):
    for site in [1, 2]:
        pixel_data = df_pixel_stats.loc[sample.id_code, site, :].reset_index().set_index('channel')
        x = transform_image(sample, pixel_data, site)
        np.save(os.path.join(OUTPUT_DIR, '{sample_id}_s{site}.npy').format(sample_id=sample.id_code, site=site), x)


# In[ ]:


class NpyImageDS(ImageDS):
    def __init__(self, df, img_dir):
        super().__init__(df, img_dir)
        
    def _get_img_path(self, index, site):
        sample_id = self.records[index].id_code
        return f'{self.img_dir}{sample_id}_s{site}.npy'
    
    def _load_data(self, index, site):
        return torch.Tensor(np.load(self._get_img_path(index, site)).astype(np.float32)/ 255.0)


# In[ ]:


ds = NpyImageDS(df, OUTPUT_DIR)
loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
get_ipython().run_line_magic('timeit', '-r 3 loop(loader)')


# Loading the .npy files is about 25% faster, although that number can vary from run to run. And although there's nothing in our code that would cause data loss, we can confirm the .npy files are the size we'd expect:

# In[ ]:


get_ipython().system('du -sh ../input/train/HEPG2-01/')
get_ipython().system('du -sh processed-data/npy/')


# So why are the numpy files faster? I was worried that the answer would be something anticlimactic, like "We've already unzipped them" (unlike the PNG files, which Kaggle stores in a zip folder.) But testing that, I saw no speed up from pre-unzipping the files:

# In[ ]:


get_ipython().system('rm -r processed-data')
get_ipython().system('mkdir -p processed-data/raw/HEPG2-01')
get_ipython().system('cp -r ../input/train/HEPG2-01/ processed-data/raw/HEPG2-01/')
ds = ImageDS(df, 'processed-data/raw/HEPG2-01')
loader = D.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
get_ipython().run_line_magic('timeit', '-r 1 loop(loader)')


# Loading numpy files appears to just be faster. I don't know why, so if anyone has insight into that please comment.

# In[ ]:


# Apparently these directories need to be removed to avoid an error.
get_ipython().system('rm -r  rxrx1-utils')
get_ipython().system('rm -r processed-data')

