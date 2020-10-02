#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm_notebook as tqdm

from fastai.vision import *
from fastai.vision.gan import *


# # Generator and Discriminator

# ## Parameters of GAN

# In[ ]:


path ='../input/all-dogs/'


# In[ ]:


trfm = get_transforms(do_flip=False, flip_vert=False, max_rotate=8.0, 
                      max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75,
                      xtra_tfms=[contrast(scale=(1, 2), p=0.75),rand_zoom(scale =(1.0,1.2)),crop_pad(size=64, row_pct=(0,1), col_pct=(0,1))]
                )
trfm1 = get_transforms(do_flip=False, flip_vert=False,xtra_tfms=[crop_pad(size=64, row_pct=(0,1), col_pct=(0,1))])


# In[ ]:


def get_data(bs, size):
    return (GANItemList.from_folder(path, noise_sz=100)
               .split_none()
               .label_from_func(noop)
#               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .transform(trfm,size=size,tfm_y=True)
               .databunch(bs=bs)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


data = get_data(64, 64)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1).to(device)
critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1).to(device)
learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)


# In[ ]:


learn.fit(100,5e-4)


# In[ ]:


learn.gan_trainer.switch(gen_mode=True)
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(16,16))


# In[ ]:


preds,_ = learn.get_preds(ds_type=DatasetType.Train)


# In[ ]:


img1=preds.numpy()[1]


# In[ ]:


img1.shape


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


img1tran =np.transpose(img1, (2, 1, 0))


# In[ ]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000
for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
    gen_images = generator(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[ ]:


images.shape

