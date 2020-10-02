#!/usr/bin/env python
# coding: utf-8

# # Plant phenotyping Training Workshop - Nanjing agricultural university - 2019
# Etienne David - UMT CAPTE - 20/10/2019
# 
# ___
# <center><img src="https://umt-capte.fr/wp-content/uploads/2019/08/logo_capte_hd_fond_sombre.png" alt="drawing" style="width:400px;"/></center>
# 
# <br>
#  ___
# 
# This notebook is intended to be used as material for the Plant phenotyping Training Workshop.
# 
# ## Goals
# - Build your first classifier and understand all the steps of training
# - Understand transfer learning strategy
# - Work on building your first segmentation model !
# 
# ## Tools
# - We will used kaggle notebook as they provide 30h of free GPU utilization and an environment with all necessary python libraries
# - The tutorial is based on Pytorch and fastai, a wrapper built for learning Deep Learning, providing tons of utilities (www.fast.ai)
# 
# ## Data
# - We will use freely available data on Kaggle

# ### Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from fastai.vision import *
import fastai
import pathlib
from fastai import *
from fastai.vision import *
from fastai.tabular import *
from torchvision.transforms import *

from itertools import product
import cv2
from tqdm import tqdm_notebook as tqdm

def dense_patchify(img, patch_size,min_overlap=0.1,scale=1 ):

    i_h, i_w, channel = np.shape(img)
    p_h, p_w = patch_size

    if i_h < p_h | i_w < p_w:
        print("image is too small for this patch size. Please add black pixels or resize images")

    if scale != 1:
        img = cv2.resize(img, (round(i_w*scale),round(i_h*scale)))



    n_h = 1
    n_w =1
    c_h, c_w = (i_h-p_h),(i_w-p_h)

    while c_h > p_h*(1-min_overlap):
        n_h += 1
        c_h = round((i_h-p_h)/n_h)

    while c_w > p_w*(1-min_overlap):
        n_w += 1
        c_w = round((i_w-p_w)/n_w)


    new_img = np.zeros(((c_h*n_h)+p_h+n_h,(c_w*n_w)+p_w+n_w,channel))
    new_img[:i_h,:i_w]=img
    
    possible_h = range(0,(c_h*(n_h+1)), c_h )
    possible_w = range(0,(c_w*(n_w+1)), c_w )

    origins_array = list(product(possible_h, possible_w))

    patches  = np.zeros((len(origins_array), p_h, p_w,3))
    
    for idx,origin in enumerate(origins_array):
        patches[idx,:,:,:] = cv2.resize(new_img[origin[0]:origin[0]+p_h,origin[1]:origin[1]+p_w,:], (p_h,p_w),  interpolation = cv2.INTER_AREA) 
            
    return patches, np.array(origins_array)


# ## Building a CNN classifier
# 
# ### Workflow
# 3 steps:
# 1. Find labeled data
# 2. Train a classifier with transfer learning. We will use a CNN classifier that has been trained on imagenet
# 3. Evaluation of performance
# 
# ### Presentation of imagenet
# 
# - ImageNet is a very large scale image. Usually people share networks which have been trained on this dataset. We usually consider that the weights learned are a better initialization than randomly initialized weights. 
# 
# 
# link: http://www.image-net.org/
# 
# 
# 
# ___ 
# 
# ### Dataset
# 
# Aarhus University: Plant Seedlings dataset. A competition was held on Kaggle few years ago. It contains seedlings RGB images.
# 
# https://vision.eng.au.dk/plant-seedlings-dataset/
# 
# 
#  
#  ___
# 
# ### Theory on transfer learning 
# 
# <center><img src="https://miro.medium.com/max/1920/1*qfQ3hmHLwApXZBN-A85r8g.png" alt="drawing"/></center>
# 
# - Using a network trained on Imagenet improve convergence, generalization and performance with small dataset
# - Libraries usually provided the needed pretrained models
# 

# #### Loading data

# In[ ]:


np.random.seed(84) # Here we fix the randomness to have replicable results


# Here are the path to dataset
data_dir = "/kaggle/input/v2-plant-seedlings-dataset/nonsegmentedv2/"
data_dir = pathlib.Path(data_dir)


data = ImageDataBunch.from_folder(data_dir, train='.', valid_pct=0.2,
                                  ds_tfms=(), size=224).normalize(imagenet_stats)



# Explications :
# - ImageDataBunch is a python object provided by fastai to facilitate the loading of data
# - data_dir is the path to our data
# - valid_pct is the percentage of images we hold for evaluation. Here we fix this percentage to 20%
# - ds_tfms is the transformation we apply to images before entering the network. Here it's empty
# - size is the target size before entering the network. Here we resize all images to 224
# - normalize: in transfer learning, the images need to be normalized with the mean and variance from the original dataset. imagenet_stats provides directly such metrics

# ### Data Augmentation
# 
# Let's take a look on our data
# Remember, we haven't apply any transformation.

# In[ ]:


data.show_batch(rows=5, figsize=(15, 15))


# ### Data augmentation
# 
# One very usual way to improve generalization of a model and increase the size of the training set is data augmentation. 
# 
# Data Augmentation consists on applying more or less realistic transformation on the image.
# 
# We will use transformations included in fastai package. 
# 
# You can always use Data Augmentation for training so never, never hesitate to use plenty of them !
# 
# <img src="https://camo.githubusercontent.com/fd2405ab170ab4739c029d7251f5f7b4fac3b41c/68747470733a2f2f686162726173746f726167652e6f72672f776562742f62642f6e652f72762f62646e6572763563746b75646d73617a6e687734637273646669772e6a706567" />

# In[ ]:


# Link to doc : https://docs.fast.ai/vision.transform.html#get_transforms

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_zoom=1.3, max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75)
data = ImageDataBunch.from_folder(data_dir, train='.', valid_pct=0.2,
                                  ds_tfms=tfms, size=224, bs=6).normalize(imagenet_stats)
data.show_batch(rows=5, figsize=(15, 15))


# ### Training
# 
# Our dataset is already ready !
# Please note that in real life it can take much more time to obtain a homogeneous dataset and it's usually the first source of error. We will speak about it later in the workshop.
# 
# Fastai provides a one-line command to:
# - download a pretrained model
# - set up loss and optimizer
# - launch training
# 
# The network we will use is resnet 18, which has 18 convolutions layers !
# 
# ![](https://www.researchgate.net/profile/Chiman_Kwan/publication/332303940/figure/fig1/AS:745864146452480@1554839278101/Architecture-of-ResNet-18-Figure-from-reference-18.jpg)
# 
# By default the cnn_learner is in a frozen mode, which mean only the last layer is learning.
# 
# #### Loss and optimizer
# The negative log likelihood loss or CrossEntropy. It is useful to train a classification problem with C classes.
# 
# ![](https://i.imgur.com/hU252jE.jpg)
# 
# The optimizer is Adam. We will pass on optimizer today.
# 
# #### Metrics
# 
# Accuracy 
# 
# ![](https://miro.medium.com/max/2868/1*WGK_3mj_KBZh9yTiLXGh-A.png)

# In[ ]:


learn = cnn_learner(data,
                    models.resnet18,
                    loss_func =CrossEntropyFlat(),
                    opt_func=optim.Adam,
                    metrics=accuracy,
                    callback_fns=ShowGraph)

learn.model_dir = "/kaggle/models"


# In[ ]:


defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(4) # Here we fit the head for 4 epochs


# #### Unfreezing
# 
# After the head is trained, we can unfreeze the whole network for better accuracy.
# 
# Here we use lr_find which help us to find the right learning rate

# In[ ]:


learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(3e-6, 3e-5))
learn.save("plant-model")


# #### Interpretation
# 
# Fastai provides useful function to explore if our model did right or not

# In[ ]:



preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10))


# In[ ]:


for i in interp.top_losses(10).indices:
    data.open(data.items[i]).show(title=data.items[i])


# ### Segmentation
# 
# The same process can be repeated for segmentation purposes. The only difference is that instead of a unique label, we expect a matrix of label.
# 
# We will use a simple dataset with three classes: background, non-hypocothyl, hypocothyl
# 
# We will use U-Net
# 
# ![](https://www.researchgate.net/profile/Alan_Jackson9/publication/323597886/figure/fig2/AS:601386504957959@1520393124691/Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al.png)

# In[ ]:



window_shape = (512, 512)
#B = extract_patches_2d(A, window_shape)



data_dir = "/kaggle/input/plant-segmentation/plant_segmentation/dataset/arabidopsis"
data_dir = pathlib.Path(data_dir)

path_img = data_dir/'images'

path_lbl = data_dir/'masks'

new_train_images = pathlib.Path() / "/kaggle/input/images"
new_train_masks = pathlib.Path() / "/kaggle/input/masks"

new_train_images.mkdir(exist_ok=True)
new_train_masks.mkdir(exist_ok=True)

for i,j  in tqdm(zip(path_img.glob("*"),path_lbl.glob("*"))):
    img = cv2.imread(str(i))
    patches,_ = dense_patchify(img,(512,512))
    
    for idx, p in enumerate(patches):
        cv2.imwrite(str( new_train_images / f"{str(idx)}_{i.name}"),p)

    img = cv2.imread(str(j))
    patches,_ = dense_patchify(img,(512,512))
    
    for idx, p in enumerate(patches):
        cv2.imwrite(str( new_train_masks / f"{str(idx)}_{j.name}"),p)


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_zoom=1.3, max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75) 

get_y_fn = lambda x: new_train_masks/f'{x.name}'
codes = ["Background","hypocotyl","non-hypocotyl"]

data = (SegmentationItemList.from_folder(new_train_images)
        .split_by_rand_pct()
        .label_from_func(get_y_fn, classes=codes)
        .transform(tfms,tfm_y=True)
        .databunch(bs=4, path=data_dir)
        .normalize(imagenet_stats))

data.show_batch(rows=2, figsize=(10, 10))

# https://towardsdatascience.com/introduction-to-image-augmentations-using-the-fastai-library-692dfaa2da42


# In[ ]:


wd=1e-2
learn = unet_learner(data, models.resnet18, metrics=dice, wd=wd)
learn.model_dir = "/kaggle/models"


# #### Can you write the rest of the code ?

# In[ ]:




