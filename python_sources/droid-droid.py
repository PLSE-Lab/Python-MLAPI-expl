#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !kaggle competitions download -c virtual-hack


# In[ ]:


# !unzip -q car_data.zip


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# In[ ]:


bs = 64
# !ls 'car_data/train'


# In[ ]:


path = Path('../input/car_data/car_data')


# In[ ]:


path.ls()


# In[ ]:


data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.3,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(8,7))


# In[ ]:


# print(data.classes)
len(data.classes),data.c


# In[ ]:


pwd


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Resnet34 , Validation on Test Folder (Provided)

# In[ ]:


data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 
                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)


# In[ ]:


learn.load('stage-1');
# learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-3))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.load('stage-2');


# ## Unfreeze Resnet 34

# In[ ]:


learn.unfreeze()


# In[ ]:


data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.), 
                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)


# In[ ]:


get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


learn.lr_find()


# In[ ]:


# learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5, max_lr=1e-4)


# ## Training: resnet50 with Test folder as validation

# In[ ]:


data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 
                                  valid='test', size=229, bs = 64) .normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10, max_lr=1e-2)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
learn.fit_one_cycle(10, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.load('stage-1-50');

interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


# interp.plot_confusion_matrix(figsize=(12,12),cmap='viridis', dpi=60)


# ## Densenet 161

# In[ ]:


data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 
                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.densenet169, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(20, max_lr=1e-2)

