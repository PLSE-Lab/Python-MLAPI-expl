#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


from pathlib import Path
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from distutils.dir_util import copy_tree


# Get Data from dataset and create databunch

# In[3]:


input_path = Path("/kaggle/input/dance-images")
path_cl= input_path/"clear"
path_bl= input_path/"blurry"


# In[4]:


path = Path("/tmp/model")
model_path_cl = path/"clear"
model_path_bl = path/"blurry"
model_path_cl.mkdir(parents=True, exist_ok=True)
model_path_bl.mkdir(parents=True, exist_ok=True)
copy_tree(str(path_cl), str(path/"clear"))
copy_tree(str(path_bl), str(path/"blurry"))


# In[5]:


bs,size=4, 256
arch = models.resnet34
src = ImageImageList.from_folder(model_path_bl).split_by_rand_pct(0.1, seed=42)


# In[6]:


def get_data(bs,size):
    data = (src.label_from_func(lambda x: model_path_cl/x.name)
           .transform(get_transforms(max_zoom=0), size=size, tfm_y=True)
           .databunch(bs=bs, num_workers=0).normalize(imagenet_stats, do_y=True))

    #data.c = 3
    return data


# In[7]:


data_gen = get_data(bs,size)
data_gen.show_batch()


# Pre-train generator

# In[8]:


y_range = (-3.,3.)
loss_gen = MSELossFlat()


# In[9]:


learn_gen = unet_learner(data_gen, arch, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)


# In[ ]:


learn_gen.lr_find()
learn_gen.recorder.plot()


# In[ ]:


lr = 1e-2
learn_gen.fit_one_cycle(4)


# In[ ]:


learn_gen.unfreeze()
learn_gen.fit_one_cycle(5, slice(1e-4,lr))
learn_gen.show_results()


# In[ ]:


learn_gen.save('gen-pre')
torch.cuda.empty_cache()


# Save generated images

# In[ ]:


learn_gen.load('gen-pre');
name_gen = 'image_gen'
path_gen = path/name_gen
path_gen.mkdir(exist_ok=True)


# In[ ]:


def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1


# In[ ]:


save_preds(data_gen.fix_dl)


# In[ ]:


learn_gen=None
torch.cuda.empty_cache()
gc.collect()


# Pretrain critic

# In[ ]:


def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    #data.c = 3
    return data


# In[ ]:


data_crit = get_crit_data([name_gen, 'clear'], bs=bs, size=size)
data_crit.show_batch()


# In[ ]:


loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
learn_critic = Learner(data_crit, gan_critic(), metrics=accuracy_thresh_expand, loss_func=loss_critic)


# In[ ]:


learn_critic.lr_find()
learn_critic.recorder.plot()


# In[ ]:


lr = 1e-4
learn_critic.fit_one_cycle(10, lr)


# In[ ]:


learn_critic.save('critic-pre')


# In[ ]:


learn_crit=None
torch.cuda.empty_cache()
gc.collect()


# Create GAN and train

# In[ ]:


data_crit = get_crit_data(['blurry', 'clear'], bs=bs, size=size)
learn_crit = Learner(data_crit, gan_critic(), metrics=None, loss_func=loss_critic).load('critic-pre')
learn_gen = unet_learner(data_gen, arch, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen).load('gen-pre')


# In[ ]:


switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=True, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)))
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-4
learn.fit(30, lr)


# Train more with bigger size images

# In[ ]:


bs,size = 1, 512
data = get_data(bs, size)
learn.data = data
gc.collect()


# In[ ]:


learn.fit(10, lr/2)


# In[ ]:


learn.show_results()


# In[ ]:


learn.save('gan')


# Test with non-artificially blurred images

# In[ ]:


m = learn.model.eval();


# In[ ]:


fn = "/kaggle/input/non-artificial-blurred/DSCN0140.JPG"
x = open_image(fn);
xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()
xb_im


# In[ ]:


pred = m(xb)
pred_im = Image(data.denorm(pred.detach())[0])
pred_im


# Fin
