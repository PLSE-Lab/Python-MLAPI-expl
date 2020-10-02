#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64
# bs = 16


# # Looking at data

# In[ ]:


help(untar_data)


# In[ ]:


path = untar_data(URLs.PETS); path


# In[ ]:


path.ls()


# In[ ]:


path_anno = path/'annotations'
path_img = path/'images'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7, 6))


# In[ ]:


print(data.classes)
len(data.classes), data.c


# # Training: resnet34

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# # Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# # Unfreezing, fine-tuning and learning rates

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))


# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=299, bs=bs//2).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# # Other data formats

# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)


# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()


# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))
data.classes


# In[ ]:


data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


fn_paths = [path/name for name in df['name']]; fn_paths[:2]


# In[ ]:


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
                label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes


# In[ ]:


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]


# In[ ]:


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes

