#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/')
path.ls()


# In[ ]:


df = pd.read_csv(path/'train_v2.csv')
df.head()


# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


np.random.seed(42)
src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' '))


# In[ ]:


data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet50


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/tmp/models')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:




