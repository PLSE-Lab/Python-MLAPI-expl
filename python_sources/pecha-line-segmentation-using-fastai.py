#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


from fastai.vision import *
from fastai.callbacks.hooks import *


# In[ ]:


#path = untar_data(URLs.CAMVID)
#path.ls()


# In[ ]:


get_ipython().system('mkdir -p /tmp/.fastai/data/')
get_ipython().system('cp -r ../input/pecha /tmp/.fastai/data/')

path = Path('/tmp/.fastai/data/pecha/')
path.ls()


# In[ ]:


path_lbl = path/'labels'
path_img = path/'images'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3], len(lbl_names)


# In[ ]:


img_f = fnames[5]
img = open_image(img_f)
img.show(figsize=(10,10)), img.size


# In[ ]:


get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'


# In[ ]:


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(10,10), alpha=1)


# In[ ]:


src_size = np.array(mask.shape[1:])
src_size,mask.data


# In[ ]:


codes = np.loadtxt(path/'codes.txt', dtype=str); codes, len(codes)


# In[ ]:


size = src_size//2
bs=4
size, src_size


# In[ ]:


src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))


# In[ ]:


tfms = get_transforms(do_flip=False, flip_vert=False)
data = (src.transform(tfms, size=size, tfm_y=True)
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


x, y = next(iter(learn.data.train_dl))
x.shape, y.shape


# In[ ]:


y.min(), y.max()


# In[ ]:


import matplotlib.pyplot as plt

img = x.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')[0]
img = img.reshape(250, 410, 3)
mask = y.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')[0]
mask = mask.reshape(250, 410)

plt.imshow(img)
plt.show()
plt.imshow(mask, cmap='gray')
plt.show()


# In[ ]:


out = learn.model(x.data); out.shape


# In[ ]:


get_ipython().system('export CUDA_LAUNCH_BLOCKING=1')


# In[ ]:


# %%debug
learn.loss_func(input=out, target=y)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


# lr = slice(1e-06,1e-03)


# In[ ]:


learn.fit_one_cycle(10, slice(1e-06,1e-03), pct_start=0.9)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


path = "."


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.show_results(rows=3, figsize=(40,50))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, slice(1e-06,1e-05), pct_start=0.8)


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.show_results(rows=3, figsize=(40,50))

