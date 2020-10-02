#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - What's your ~~pet~~ car

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


import torch

def my_setseed(s=42): #set seed, for reproducible results
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Looking at the data

# In[ ]:


# path = untar_data(URLs.PETS); path
path = Path('/kaggle/input/carros/data/data/cars'); path


# In[ ]:


path.ls()


# In[ ]:


# path_anno = path/'annotations'
# path_img = path/'images'


# In[ ]:


# fnames = get_image_files(path_img)
fnames = get_image_files(path, recurse=True)
fnames[:5]


# In[ ]:


my_setseed()
# pat = r'/([^/]+)_\d+.jpg$'
pat = r'/([^/]+)/[^/]+$' #regex to get class from folder name


# In[ ]:


# data = ImageDataBunch.from_name_re( path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
#                                   ).normalize(imagenet_stats)
data = ImageDataBunch.from_name_re(path, fnames, pat=pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=1
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))


# ## Training: resnet34

# In[ ]:


my_setseed()
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4)


# In[ ]:


learn.save('basico')


# Now we fine tune

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, slice(3e-4,3e-3))


# Not overfitting yet, might be worth training more epochs. Save what we have so far so that we can revert more easily if needed.

# In[ ]:


learn.save('finetuned')


# In[ ]:


learn.fit_one_cycle(1)


# ## Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# ## Training: resnet50

# In[ ]:


my_setseed()
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4)


# In[ ]:


learn.save('basico-50')


# Now we fine tune

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-4,2e-3))


# Doesn't improve. Reload previous model.

# In[ ]:


learn.load('basico-50');


# ## Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

