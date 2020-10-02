#!/usr/bin/env python
# coding: utf-8

# # fastai training
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastai.metrics import KappaScore
torch.cuda.is_available()


# In[ ]:


from torchvision.models import *
import torch
import torch.optim as optim
from fastai.callbacks import *


# In[ ]:


print(len(os.listdir("../input/panda-challenge-resized-dataset/")))

len(os.listdir("../input/prostate-cancer-grade-assessment/train_images/"))


# In[ ]:


train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


train_df['path'] = train_df.image_id.apply(lambda  x :x +'.jpeg')
train_df['isup_grade'] = train_df.isup_grade.apply(lambda  x : float(x))
train_df.head(2)


# In[ ]:


train = ImageList.from_df(train_df, path='../input/panda-challenge-resized-dataset/',cols='path').split_by_rand_pct(0.2,seed=42).label_from_df(cols='isup_grade', label_cls=FloatList)


# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,)


# In[ ]:


data = (train.transform(tfms,size=256).databunch(bs=32).normalize(imagenet_stats))

data.c


# In[ ]:


data.show_batch(2)


# In[ ]:


from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat).cpu(), y.cpu(), weights='quadratic'),device='cuda:0')


# In[ ]:


import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path
from efficientnet_pytorch import EfficientNet


# In[ ]:


model_name = 'efficientnet-b3'
def getModel(pret):
    model = EfficientNet.from_name(model_name,override_params={'num_classes':data.c})
    if pret:
        model_state = torch.load('../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth')

        if '_fc.weight' in model_state.keys():
            model_state.pop('_fc.weight')
            model_state.pop('_fc.bias')
            res = model.load_state_dict(model_state, strict=False)
            assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        else:
            # A basic remapping is required
            from collections import OrderedDict
            mapping = { i:o for i,o in zip(model_state.keys(), model.state_dict().keys()) }
            mapped_model_state = OrderedDict([
                (mapping[k], v) for k,v in model_state.items() if not mapping[k].startswith('_fc')
            ])
            res = model.load_state_dict(mapped_model_state, strict=False)
            print(res)
    return model,model._fc.in_features
#     model._bn1 = nn.Identity()
    #model._fc = nn.Linear(64,data.c)
#     return nn.Sequential(model)


# In[ ]:


model,j=getModel(True)
model._fc= nn.Linear(j, data.c)


# In[ ]:


learn = Learner(data,model,metrics=[quadratic_kappa],)
learn.model_dir = '/kaggle/working/'
learn.split(children(learn.model))
len(learn.layer_groups)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(8,1e-4)


# In[ ]:



learn.export('/kaggle/working/model_1.pkl')


# In[ ]:


learn.freeze_to(-1)
learn.fit_one_cycle(5,1e-5)


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:



learn.export('/kaggle/working/model.pkl')


# In[ ]:




