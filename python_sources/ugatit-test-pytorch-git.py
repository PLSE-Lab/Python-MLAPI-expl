#!/usr/bin/env python
# coding: utf-8

# ## Data

# In[ ]:


pip install Augmentor


# In[ ]:


import numpy as np # linear algebra
import os # accessing directory structure
import fastai
from fastai.vision import *
import Augmentor


# In[ ]:


def augmente(path,nbSample=1) :
    p = Augmentor.Pipeline(path,path)
    p.flip_left_right(0.5) 
    p.rotate(0.3, 25, 25) 
    p.skew(0.1, 0.1) 
    p.skew_tilt(0.1, 0.1) 
    p.shear(0.1,10,10)
    p.invert(0.333333)
    p.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=12)
    for i in range(nbSample) :
        p.process()


# outcode contain the resize function of pictures

# In[ ]:


cd /kaggle/usr/lib/outcode/


# In[ ]:


from outcode import *


# clone the repo

# In[ ]:


get_ipython().system('git clone https://github.com/taki0112/UGATIT.git /kaggle/UGATIT/')


# create directories for data and copy it into them

# In[ ]:


get_ipython().system('mkdir /kaggle/UGATIT/dataset')
get_ipython().system('mkdir /kaggle/UGATIT/dataset/selfie2anime')
get_ipython().system('mkdir /kaggle/UGATIT/dataset/selfie2anime/trainA')
get_ipython().system('mkdir /kaggle/UGATIT/dataset/selfie2anime/trainB')
get_ipython().system('mkdir /kaggle/UGATIT/dataset/selfie2anime/testA')
get_ipython().system('mkdir /kaggle/UGATIT/dataset/selfie2anime/testB')
get_ipython().system('mkdir /kaggle/UGATIT/checkpoint')


# In[ ]:


path = Path('/kaggle/input')
pathUGATIT = Path('/kaggle/UGATIT')


# In[ ]:


ls /kaggle/input


# In[ ]:


ls /kaggle/input/cs-170-save/


# In[ ]:


#resize(path/"celeba-dataset/img_align_celeba/img_align_celeba/",pathUGATIT/'dataset/selfie2anime/trainA',limit=10000,begin_file=70000)
#resize(path/"flickrfaceshq-dataset-ffhq/",pathUGATIT/'dataset/selfie2anime/trainA',limit=12000, begin_file=20000)
resize(path/"selfie2anime/trainA", pathUGATIT/'dataset/selfie2anime/trainA',limit=5000)
resize(path/'cultist-faces/blackandwhite/',pathUGATIT/'dataset/selfie2anime/trainB')
#resize(path/"celeba-dataset/img_align_celeba/img_align_celeba/",pathUGATIT/'dataset/selfie2anime/testA',limit=50,begin_file=10000)
resize(path/"flickrfaceshq-dataset-ffhq/",pathUGATIT/'dataset/selfie2anime/testA',limit=50)
resize(path/'cultist-faces/blackandwhite/',pathUGATIT/'dataset/selfie2anime/testB',limit=50)


# In[ ]:


augmente(pathUGATIT/'dataset/selfie2anime/trainB',2)


# In[ ]:


ls -1 /kaggle/UGATIT/dataset/selfie2anime/trainB | wc -l


# In[ ]:


cd /kaggle/UGATIT


# In[ ]:


rm -fr .git


# In[ ]:


ls 


# # Copy checkpoints

# In[ ]:


nbEpoch = 138
nbIter = 10000


# In[ ]:


#cp -r /kaggle/input/ugatit-selfie2anime-pretrained/checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/ /kaggle/UGATIT/checkpoint/


# In[ ]:


cp -r /kaggle/input/ugatit-test-pytorch-git/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/ /kaggle/UGATIT/checkpoint/


# In[ ]:


# cp -r /kaggle/input/cs-170-save/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/ /kaggle/UGATIT/checkpoint/


# In[ ]:


get_ipython().system('rm -fr /kaggle/UGATIT/checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/*1340000*')
get_ipython().system('rm -fr /kaggle/UGATIT/checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/*1350000*')


# In[ ]:


ls /kaggle/input/ugatit-test-pytorch-git/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/


# In[ ]:


ls /kaggle/UGATIT/checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/


# # Train Cs Face

# In[ ]:


get_ipython().system('python main.py --dataset selfie2anime --light True --iteration {nbIter} --save_freq 10000 --print_freq {nbIter} --epoch {nbEpoch}')


# In[ ]:


cp -r /kaggle/UGATIT/checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/ /kaggle/working


# In[ ]:


ls /kaggle/working/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/


# # Test checkpoints

# In[ ]:


get_ipython().system('python main.py --dataset selfie2anime --light True --phase test')


# In[ ]:


cp -r /kaggle/UGATIT/results/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing /kaggle/working/img


# In[ ]:


ls -1 /kaggle/UGATIT/results/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing | wc -l


# In[ ]:


open_image("/kaggle/input/flickrfaceshq-dataset-ffhq/01522.png")


# In[ ]:


open_image("/kaggle/UGATIT/results/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/01522.png")


# In[ ]:


open_image("/kaggle/UGATIT/results/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/BWci.jpg")


# In[ ]:


pathRes = "/kaggle/UGATIT/results/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing"

src = ImageImageList.from_folder(pathRes).split_by_rand_pct(0.1, seed=50)
data_init = (src.label_from_func(lambda x: Path(pathRes)/x.name)
           .databunch(bs=8))

data_init.show_batch(8)


# # Memory

# In[ ]:


from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

