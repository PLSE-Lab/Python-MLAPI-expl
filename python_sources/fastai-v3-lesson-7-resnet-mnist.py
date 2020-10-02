#!/usr/bin/env python
# coding: utf-8

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=7)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-7-official-resources/32553)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-7-in-class-chat/32554/118)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-7-further-discussion/32555)
# 
# Note: This is a mirror of the FastAI Lesson 7 Nb. 
# Please thank the amazing team behind fast.ai for creating these, I've merely created a mirror of the same here
# For complete info on the course, visit course.fast.ai

# ## MNIST CNN

# In[ ]:


#!conda update -c pytorch -c fastai fastai


# In[ ]:


import fastai
import fastai.utils.collect_env
fastai.utils.collect_env.show_install(1)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *


# ### Data

# In[ ]:


path = untar_data(URLs.MNIST)


# In[ ]:


path.ls()


# In[ ]:


il = ImageItemList.from_folder(path, convert_mode='L')


# In[ ]:


il.items[0]


# In[ ]:


defaults.cmap='binary'


# In[ ]:


il


# In[ ]:


il[0].show()


# In[ ]:


sd = il.split_by_folder(train='training', valid='testing')


# In[ ]:


sd


# In[ ]:


(path/'training').ls()


# In[ ]:


ll = sd.label_from_folder()


# In[ ]:


ll


# In[ ]:


x,y = ll.train[0]


# In[ ]:


x.show()
print(y,x.shape)


# In[ ]:


tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])


# In[ ]:


ll = ll.transform(tfms)


# In[ ]:


bs = 128


# In[ ]:


# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()


# In[ ]:


x,y = data.train_ds[0]


# In[ ]:


x.show()
print(y)


# In[ ]:


def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))


# In[ ]:


xb,yb = data.one_batch()
xb.shape,yb.shape


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# ### Basic CNN with batchnorm

# In[ ]:


def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


# In[ ]:


model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


learn.summary()


# In[ ]:


xb = xb.cuda()


# In[ ]:


model(xb).shape


# In[ ]:


learn.lr_find(end_lr=100)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=0.1)


# ### Refactor

# In[ ]:


def conv2(ni,nf): return conv_layer(ni,nf,stride=2)


# In[ ]:


model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(10, max_lr=0.1)


# ### Resnet-ish

# In[ ]:


from fastai.layers import *


# In[ ]:


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)
        
    def forward(self, x): return x + self.conv2(self.conv1(x))


# In[ ]:


help(res_block)


# In[ ]:


class SequentialEx(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self,i): return self.layers[i]
    def append(self,l): return self.layers.append(l)
    def extend(self,l): return self.layers.extend(l)
    def insert(self,i,l): return self.layers.insert(i,l)


# In[ ]:


class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."
    def __init__(self, dense:bool=False):
        super().__init__()
        self.dense=dense

    def forward(self, x): return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)


# In[ ]:


def res_block(nf, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, bottle:bool=False, **kwargs):
    "Resnet block of `nf` features."
    norm2 = norm_type
    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **kwargs),
                      conv_layer(nf_inner, nf, norm_type=norm2, **kwargs),
                      MergeLayer(dense))


# In[ ]:


model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)


# In[ ]:


def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


# In[ ]:


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


learn.lr_find(end_lr=100)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, max_lr=0.05)


# In[ ]:


learn.summary()


# ## fin

# In[ ]:




