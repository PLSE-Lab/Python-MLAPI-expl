#!/usr/bin/env python
# coding: utf-8

# ## Image segmentation with CamVid

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# In[ ]:


path = untar_data(URLs.CAMVID)
path.ls()


# In[ ]:


path_lbl = path/'labels'
path_img = path/'images'


# ## Data

# In[ ]:


fnames = get_image_files(path_img)
fnames[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3]


# In[ ]:


img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))


# In[ ]:


get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'


# In[ ]:


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)


# In[ ]:


src_size=np.array(mask.shape[1:])
src_size, mask.data


# In[ ]:


codes = np.loadtxt(path/'codes.txt', dtype=str); codes


# ## Dataset

# In[ ]:


size = src_size//2
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


src = (SegmentationItemList.from_folder(path_img)
      .split_by_fname_file('../valid.txt')
      .label_from_func(get_y_fn, classes=codes))


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(2, figsize=(10,7))


# In[ ]:


data.show_batch(2, figsize=(10,7), ds_type = DatasetType.Valid)


# ## Model

# In[ ]:


name2id = {v: k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[ ]:


metrics = acc_camvid


# In[ ]:


wd = 1e-2


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 3e-3


# In[ ]:


learn.fit_one_cycle(10, slice(lr), pct_start=0.9)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.show_results(rows=3, figsize=(8,9))


# In[ ]:


learn.unfreeze()


# In[ ]:


lrs = slice(lr/400, lr/4)


# In[ ]:


learn.fit_one_cycle(12, lrs, pct_start=0.8)


# In[ ]:


learn.save('stage-2');


# ## Go big

# In[ ]:


learn.destroy();


# In[ ]:


size = src_size

free = gpu_mem_get_free_no_cache()

if free > 8200:  bs=3
else:            bs=1
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
    .databunch(bs=bs)
    .normalize(imagenet_stats))


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)


# In[ ]:


learn.load('stage-2');


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(10, slice(lr), pct_start = 0.8)


# In[ ]:


learn.save('stage-1-big')


# In[ ]:


learn.load('stage-1-big');


# In[ ]:


learn.unfreeze()


# In[ ]:


lrs = slice(1e-6, lr/10)


# In[ ]:


learn.fit_one_cycle(10, lrs)


# In[ ]:


learn.save('stage-2-big')


# In[ ]:


learn.load('stage-2-big');


# In[ ]:


learn.show_results(rows=3, figsize=(10,10))


# In[ ]:




