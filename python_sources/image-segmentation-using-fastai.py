#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.callbacks.hooks import *


# In[ ]:


path = untar_data(URLs.CAMVID)
path.ls()


# In[ ]:


path_lbl = path/'labels'
path_img = path/'images'


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


src_size = np.array(mask.shape[1:])
src_size,mask.data


# In[ ]:


codes = np.loadtxt(path/'codes.txt', dtype=str); codes


# In[ ]:


size = src_size//2
bs=8


# In[ ]:


src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs, num_workers=0)
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch(2, figsize=(10,7))


# In[ ]:


data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)


# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[ ]:


metrics=acc_camvid


# In[ ]:


wd=1e-2


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


# lr = slice(1e-06,1e-03)


# In[ ]:


learn.fit_one_cycle(10, slice(1e-06,1e-03), pct_start=0.9)


# In[ ]:


path = "."


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.show_results(rows=3, figsize=(8,9))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, slice(1e-5,1e-4), pct_start=0.8)


# In[ ]:


learn.save('stage-2')


# In[ ]:




