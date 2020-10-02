#!/usr/bin/env python
# coding: utf-8

# In this kernel i train a classifier using fastai. The classifieer data used for training is genereated [here](https://www.kaggle.com/meaninglesslives/make-pneumothorax-classifer-data). I use pretrained imagenet weights for seresnext50.
# 
# I use the trained classifier to modify the submission in [my efficientnet kernel](https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder). The idea is that the trained unet may predict masks even when there is no pneumothorax. Zeroing out wrongly predicted masks will help us get a better performance.

# # Loading Libraries

# In[ ]:


import torchvision
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
import cv2 as cv
import numpy as np
import pandas as pd
import fastai


# In[ ]:


import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = fastprogress.force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
fastai.basic_data.master_bar, fastai.basic_data.progress_bar = master_bar, progress_bar
dataclass.master_bar, dataclass.progress_bar = master_bar, progress_bar

fastai.core.master_bar, fastai.core.progress_bar = master_bar, progress_bar
seed = 10


# # Making the training set and dataloader

# In[ ]:


os.listdir('../input')
get_ipython().system('tar -xf /kaggle/input/make-pneumothorax-classifer-data/classifier_data.tar.gz -C .')
get_ipython().system('tar -xf /kaggle/input/make-pneumothorax-classifer-data/test_data.tar.gz -C .')


# In[ ]:


ls


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])


path = '/kaggle/working/classifier_data'
path_test = '/kaggle/working/test'

data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=seed)
        .label_from_folder()
        .transform(tfms, size=224)
        .databunch().normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


# class names and number of classes
# print(data.classes)
len(data.classes),data.c


# In[ ]:


get_ipython().system('pip install pretrainedmodels')
import pretrainedmodels


# In[ ]:


from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()


# In[ ]:


def resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))


# In[ ]:


learn = cnn_learner(data, resnext50_32x4d, pretrained=True, cut=-2,
                    split_on=lambda m: (m[0][3], m[1]), 
                    metrics=[accuracy])
learn.loss_fn = FocalLoss()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# # Stage 1 training with size 128

# In[ ]:


learn.fit_one_cycle(32, max_lr=slice(2e-2), wd=1e-5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


learn.save('resnext50_32x4d_1');
learn.unfreeze();
learn = learn.clip_grad();


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.load('resnext50_32x4d_1');
learn.unfreeze();
learn = learn.clip_grad();


# In[ ]:


lr = [3e-3/100, 3e-3/20, 3e-3/10]
learn.fit_one_cycle(36, lr, wd=1e-7)


# In[ ]:


learn.save('resnext50_32x4d_2');


# # Size 224

# In[ ]:


SZ = 224
cutout_frac = 0.20
p_cutout = 0.75
cutout_sz = round(SZ*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)

tfms = get_transforms(do_flip=True, max_rotate=15, flip_vert=False, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2)), cutout_tfm])


# In[ ]:


data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=seed)
        .label_from_folder()
        .transform(tfms, size=224)
        .databunch().normalize(imagenet_stats))

learn.data = data
learn.bs = 32
data.train_ds[0][0].shape


# In[ ]:


learn.load('resnext50_32x4d_2');
learn.freeze();
learn = learn.clip_grad();


# In[ ]:


learn.loss_func = FocalLoss()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(24, slice(3e-3), wd=5e-6)


# In[ ]:


learn.save('resnext50_32x4d_3');
learn.load('resnext50_32x4d_3');


# In[ ]:


learn.unfreeze();
learn = learn.clip_grad();


# In[ ]:


lr = [1e-3/200, 1e-3/20, 1e-3/10]
learn.fit_one_cycle(32, lr)


# In[ ]:


learn.save('resnext50_32x4d_4');
learn.load('resnext50_32x4d_4');


# In[ ]:


learn.export('/kaggle/working/fastai_resnet.pkl');


# # Predicting on the test set

# In[ ]:


learn = load_learner('/kaggle/working/','fastai_resnet.pkl', ImageList.from_folder(path_test))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
cls_pred = F.softmax(preds,1).argmax(1).cpu().numpy()


# In[ ]:


paths = list(map(str,list(learn.data.test_ds.x.items)))
all_test_paths = [p.split('/')[-1][:-4] for p in paths]

df_preds = pd.DataFrame()
df_preds['test_paths'] = all_test_paths
df_preds['class_pred'] = cls_pred

df_preds.set_index('test_paths',inplace=True)


# In[ ]:


df_preds.head()


# In[ ]:


no_dis_idx = df_preds[df_preds.class_pred==1].index
print(len(no_dis_idx))


# In[ ]:


sub = pd.read_csv('/kaggle/input/unet-plus-plus-with-efficientnet-encoder/submission.csv'
                  ,index_col='ImageId')
sub.head()


# In[ ]:


sub.loc[no_dis_idx] = ' -1'


# In[ ]:


sub.to_csv('sub_classifier_correction.csv')


# In[ ]:


get_ipython().system('rm -r */')

