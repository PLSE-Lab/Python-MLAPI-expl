#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *


# In[ ]:


import pandas as pd
import glob


# In[ ]:


path_images = Path("../input/train")
path_lbl = path_images


# In[ ]:


fnames = glob.glob('../input/train/*[!_mask].tif')


# In[ ]:


lbl_names = glob.glob('../input/train/*_mask.tif')


# In[ ]:


def get_y_fn(x):
    x = Path(x)
    return path_lbl/f'{x.stem}_mask{x.suffix}'


# In[ ]:


filter_func = lambda x: str(x) in fnames


# In[ ]:


size = 224


# In[ ]:


from fastai.utils.mem import *
#free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
#if free > 8200: bs=8
#else:           bs=4
#print(f"using bs={bs}, have {free}MB of GPU RAM free")
bs=8


# In[ ]:


class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom


# In[ ]:


codes = ['0','1']
src = (SegItemListCustom.from_folder(path_images)
       .filter_by_func(filter_func)
       .random_split_by_pct()
       .label_from_func(get_y_fn,classes=codes))


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
data.path = Path('.')


# In[ ]:


#data.show_batch(2, figsize=(10,7))


# In[ ]:


learn = unet_learner(data, models.densenet121, metrics=[dice])


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr = 1e-5)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,max_lr = slice(1e-6,1e-4))

