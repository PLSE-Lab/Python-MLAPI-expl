#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


pip install Augmentor


# In[ ]:


import numpy as np # linear algebra
import os # accessing directory structure
from os import listdir
from os.path import isfile, join
import fastai
from fastai.vision import *
import Augmentor
from PIL import Image 


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


# In[ ]:


cd /kaggle/usr/lib/outcode/


# In[ ]:


from outcode import *


# In[ ]:


get_ipython().system('git clone https://github.com/giddyyupp/ganilla /kaggle/ganilla')


# # Datasets

# In[ ]:


ls /kaggle/ganilla


# In[ ]:


cd /kaggle/ganilla


# In[ ]:


get_ipython().system('mkdir /kaggle/ganilla/datasets/photo2witness')
get_ipython().system('mkdir /kaggle/ganilla/datasets/photo2witness/trainA')
get_ipython().system('mkdir /kaggle/ganilla/datasets/photo2witness/trainB')
get_ipython().system('mkdir /kaggle/ganilla/datasets/photo2witness/testA')
get_ipython().system('mkdir /kaggle/ganilla/datasets/photo2witness/testB')
get_ipython().system('mkdir /kaggle/ganilla/checkpoints/')
get_ipython().system('mkdir /kaggle/ganilla/checkpoints/photo2witness/')


# In[ ]:


path = Path('/kaggle/input')
pathUGATIT = Path('/kaggle/ganilla')


# In[ ]:


resize(path/"landscape-pictures/"  , pathUGATIT/'datasets/photo2witness/trainA',limit=500)
resize(path/'the-witness/'        , pathUGATIT/'datasets/photo2witness/trainB')

resize(path/"landscape-pictures/", pathUGATIT/'datasets/photo2witness/testA',limit=50)
resize(path/'the-witness/'       , pathUGATIT/'datasets/photo2witness/testB',limit=50)


# In[ ]:


# augmente(pathUGATIT/'datasets/photo2witness/trainB',1)


# In[ ]:


ls -1 /kaggle/ganilla/datasets/photo2witness/trainB | wc -l


# In[ ]:


ls /kaggle/input/ganilla-witness/*.pth /kaggle/input/ganilla-witness/*.txt


# In[ ]:


cp /kaggle/input/ganilla-witness/*.pth /kaggle/ganilla/checkpoints/photo2witness/


# In[ ]:


cp /kaggle/input/ganilla-witness/*.txt /kaggle/ganilla/checkpoints/photo2witness/


# In[ ]:


ls /kaggle/ganilla/checkpoints/photo2witness/


# # Training

# In[ ]:


nb_epoch = 100
nb_niter = 50
begin_epoch = 100
end_epoch = nb_epoch + nb_niter


# In[ ]:


pip install -r requirements.txt


# In[ ]:


get_ipython().system('python train.py --dataroot ./datasets/photo2witness --name photo2witness --model cycle_gan --netG resnet_fpn --niter {nb_niter} --niter_decay {nb_epoch} --batch_size 4 --print_freq 2000 --save_epoch_freq {end_epoch} --epoch {begin_epoch}  --continue_train')


# # Test

# In[ ]:


get_ipython().system('python test.py --dataroot ./datasets/photo2witness --name photo2witness --model cycle_gan --netG resnet_fpn --batch_size 4 --epoch {end_epoch}')


# In[ ]:


pathRes = "/kaggle/ganilla/results/photo2witness/test_"+str(end_epoch)+"/images/" #/kaggle/ganilla/checkpoints/photo2witness/web/images

src = ImageImageList.from_folder(pathRes).split_by_rand_pct(0.1, seed=50)
data_init = (src.label_from_func(lambda x: Path(pathRes)/x.name)
           .databunch(bs=8))

data_init.show_batch(8)


# In[ ]:


get_ipython().system('mkdir /kaggle/working/img')


# In[ ]:


cp -r /kaggle/ganilla/results/photo2witness/test_{end_epoch}/images/*real_A* /kaggle/working/img


# In[ ]:


cp -r /kaggle/ganilla/results/photo2witness/test_{end_epoch}/images/*real_B* /kaggle/working/img


# In[ ]:


cp -r /kaggle/ganilla/results/photo2witness/test_{end_epoch}/images/*fake_A* /kaggle/working/img


# In[ ]:


cp -r /kaggle/ganilla/results/photo2witness/test_{end_epoch}/images/*fake_B* /kaggle/working/img


# In[ ]:


ls -1 /kaggle/ganilla/results/photo2witness/test_{end_epoch}/images/ | wc -l


# # Output

# In[ ]:


ls /kaggle/ganilla/checkpoints/photo2witness


# In[ ]:


cp -r /kaggle/ganilla/checkpoints/photo2witness/{end_epoch}_* /kaggle/working


# In[ ]:


cp -r /kaggle/ganilla/checkpoints/photo2witness/*.txt /kaggle/working


# In[ ]:


cd /kaggle/working


# In[ ]:


ls img


# In[ ]:


ls -1 /kaggle/working/img | wc -l


# # Memory

# In[ ]:


from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

