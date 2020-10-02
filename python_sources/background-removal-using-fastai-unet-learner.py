#!/usr/bin/env python
# coding: utf-8

# # Background Removal using Fastai Unet learner
# ## Purpose
# The purpose of the notebook is to demonstrate Fastai library usage to perform background removal in images.
# 
# ## Methodology
# The dataset consists of:-
# 1. Train Folder: Consists of images which possess some kind of corruption such as paper wrinkles or coffee stains.
# 2. Train_cleaned Folder: Consists of cleaned images
# 3. Test Folder: Corrupt images that needs to be cleaned using the model for performance verification.
# 

# # Setup
# ## Library import

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pathlib
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn
from subprocess import check_output


# # Data import

# In[ ]:


path = pathlib.Path('/kaggle/input/denoising-dirty-documents')


# In[ ]:


items = check_output(["ls", "../input/denoising-dirty-documents"]).decode("utf8")
items = items.split('\n'); items.pop(); items


# In[ ]:


import zipfile

for item in items:
    # Will unzip the files so that you can see them..
    print(item)
    with zipfile.ZipFile(path/item,"r") as z:
        z.extractall(".")


# In[ ]:


bs,size=4,128
arch = models.resnet34
path_train = pathlib.Path("/kaggle/working/train")
path_train_cleaned = pathlib.Path("/kaggle/working/train_cleaned")
path_test = pathlib.Path("/kaggle/working/test")


# # Data processing

# In[ ]:


src = ImageImageList.from_folder(path_train).split_by_rand_pct(0.2, seed=42)
      


# In[ ]:


def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_train_cleaned/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


# In[ ]:


data = get_data(bs,size)


# In[ ]:


data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))


# ## Feature Loss

# In[ ]:


t = data.valid_ds[0][1].data
t = torch.stack([t,t])


# In[ ]:


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


# In[ ]:


gram_matrix(t)


# In[ ]:


base_loss = F.l1_loss


# In[ ]:


vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)


# In[ ]:


blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


# In[ ]:


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


# In[ ]:


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])


# ## Train

# In[ ]:


wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();


# In[ ]:


learn.model_dir  ='/kaggle/working/models'


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


len(data.valid_ds.items)


# In[ ]:


lr = 1e-3


# In[ ]:


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


# In[ ]:


do_fit('1a', slice(lr*10))


# In[ ]:


learn.unfreeze()


# In[ ]:


do_fit('1b', slice(1e-5,lr))


# In[ ]:


data = get_data(12,size*2)


# In[ ]:


learn.data = data
learn.freeze()
gc.collect()


# In[ ]:


learn.load('1b');


# In[ ]:


do_fit('2a')


# In[ ]:


learn.unfreeze()


# In[ ]:


do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)


# ## Test

# In[ ]:


fn = data.valid_ds.x.items[10]; fn


# In[ ]:


img = open_image(fn); img.shape


# In[ ]:


p,img_pred,b = learn.predict(img)


# In[ ]:


show_image(img, figsize=(8,5), interpolation='nearest');


# In[ ]:


Image(img_pred).show(figsize=(8,5))


# In[ ]:


model_path = Path("/kaggle/working/export.pkl")
learn.export(file = model_path)

