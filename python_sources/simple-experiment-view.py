#!/usr/bin/env python
# coding: utf-8

# ## Save RGB Images with Pytorch
# 
# **Mostly taken from [Karl Heyer's](https://www.kaggle.com/towardsentropy) script linked [here](https://www.kaggle.com/towardsentropy/save-rgb-image-pytorch)**
# 
# This code saves RGB converted versions of all images in the train and test sets. The conversion is done using code from the [RXRX1 Utils Repo](https://github.com/recursionpharma/rxrx1-utils) adapted to Pytorch.
# 
# Note that since this code saves images, it won't work on Kaggle read only kernels

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import torch
import PIL
from pathlib import Path
from PIL import Image


# In[ ]:


path = Path('../input')


# In[ ]:


def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    #a = np.transpose(a, (1, 0, 2))
    #a = np.transpose(a, (2, 1, 0))
    a = np.transpose(a, (2, 0, 1))
    return torch.from_numpy(a.astype(dtype, copy=False) )


# In[ ]:


def _load_dataset(base_path, dataset, include_controls=True):
    df =  pd.read_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = pd.read_csv(
            os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res


# In[ ]:


def combine_metadata(base_path=path,
                     include_controls=True):
    df = pd.concat(
        [
            _load_dataset(
                base_path, dataset, include_controls=include_controls)
            for dataset in ['test', 'train']
        ],
        sort=True)
    return df


# In[ ]:


md = combine_metadata()


# In[ ]:


md.head()


# In[ ]:


def image_path(dataset, experiment, plate,
               address, site, channel, base_path=path):

    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))


# In[ ]:


def open_6_channel(dataset, experiment, plate, address, site, base_path=path):
    return torch.cat([pil2tensor(PIL.Image.open(image_path(dataset, experiment, plate, address, site, i)), np.float32) for i 
            in range(1,7)])


# In[ ]:


DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)


# In[ ]:


RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}


# In[ ]:


def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):

    t = t.permute(1,2,0).numpy()
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) /             ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) +             rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im

def save_path(dataset, experiment, plate,
               address, site, base_path=path):

    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}.png".format(address, site))

def save_rgb(dataset, experiment, plate, address, site):
    im_6 = open_6_channel(dataset, experiment, plate, address, site)
    im_rgb = convert_tensor_to_rgb(im_6)
    dest = 'train_rgb' if 'train' in dataset else 'test_rgb'
    save_file = save_path(dest, experiment, plate, address, site).replace('../input', '../working/')
    # print(save_file)
    PIL.Image.fromarray(im_rgb.astype('uint8')).save(save_file)


# In[ ]:


path = ''
for type_ in ['train', 'test']:
    os.mkdir('{}_rgb'.format(type_))
    for folder in os.listdir('../input/{}'.format(type_)):
        os.mkdir(os.path.join('{}_rgb'.format(type_), folder))
        for i in range(1,5):
            os.mkdir(os.path.join('{}_rgb'.format(type_), folder, 'Plate{}'.format(i)))


# In[ ]:


def plot_experiment(x,y,stater_pos):
    xn,yn = x,y
    f,ax = plt.subplots(xn,yn, figsize=(25, 45))
    for i in range(xn*yn):
        imgidx = mdtmp.iloc[i+stater_pos] 
        fname = '../working//{}_rgb/{}/Plate{}/{}_s{}.png'.format(imgidx['dataset'], imgidx['experiment'], imgidx['plate'], imgidx['well'], imgidx['site'])
        a = Image.open(fname) 
        os.remove(fname)
        ax[int(i/yn), i%yn].imshow(a)
        ax[int(i/yn), i%yn].title.set_text(imgidx['well_type']+'\n'+imgidx['well'])
        ax[int(i/yn), i%yn].title.set_fontsize(10)
        if imgidx['well_type']=='positive_control':
            ax[int(i/yn), i%yn].title.set_color('blue')
        if imgidx['well_type']=='negative_control':
            ax[int(i/yn), i%yn].title.set_color('red')
        ax[int(i/yn), i%yn].set_xticks([])
        ax[int(i/yn), i%yn].set_yticks([])
    plt.show() 


# In[ ]:


EXPERIMENT = 'U2OS-01'
mdtmp = md[(md['experiment']==EXPERIMENT) & (md['plate']==1) & (md['site']==1)]
_ = mdtmp.apply(lambda row: save_rgb(row['dataset'], row['experiment'], row['plate'], row['well'], row['site']), axis=1)


# In[ ]:


imgidx = mdtmp.iloc[0]
imgidx


# In[ ]:


plot_experiment(22,14,0)

