#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from pathlib import PosixPath
path = PosixPath('../input')


# In[3]:


import pandas as pd
df = pd.read_csv(path/'train.csv')


# In[4]:


df.id = 'train/train/' + df.id


# In[5]:


df.head()


# In[6]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[7]:


src = (ImageList.from_df(df, path)
       .split_by_rand_pct(0.2)
       .label_from_df(1))


# In[8]:


tfms=get_transforms()
data = (src.transform(tfms, size=32)
        .databunch()
        .normalize(imagenet_stats))


# In[9]:


data.train_ds[0][0].shape


# In[10]:


data.show_batch(rows=3, figsize=(9,7))


# In[11]:


data.classes, data.c


# In[12]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[13]:


learn.data.train_ds[0][0].shape


# In[14]:


learn.lr_find()
learn.recorder.plot()


# In[15]:


learn.fit_one_cycle(8, slice(1e-2))


# In[16]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[17]:


interp.plot_top_losses(9, figsize=(9,7))


# In[18]:


interp.plot_confusion_matrix()


# In[19]:


learn.recorder.plot_losses()


# In[20]:


learn.recorder.plot_lr()


# In[21]:


learn.save('stage-1')


# In[23]:


learn.unfreeze()


# In[26]:


learn.lr_find()
learn.recorder.plot()


# In[27]:


learn.fit_one_cycle(4, slice(1e-5/2))


# In[28]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[29]:


interp.plot_top_losses(9, figsize=(9,7))


# In[30]:


interp.plot_confusion_matrix()


# In[31]:


learn.recorder.plot_losses()


# In[32]:


learn.recorder.plot_lr()


# In[33]:


learn.save('stage-2')


# In[43]:


tfms=get_transforms(flip_vert=True)
data64 = (src.transform(tfms, size=64)
          .databunch()
          .normalize(imagenet_stats))


# In[44]:


learn.data = data64


# In[45]:


learn.freeze()


# In[46]:


learn.lr_find()
learn.recorder.plot()


# In[47]:


learn.fit_one_cycle(4, slice(1e-2))


# In[48]:


learn.fit_one_cycle(4, slice(1e-2))


# In[49]:


learn.fit_one_cycle(4, slice(1e-2))


# In[50]:


learn.fit_one_cycle(4, slice(1e-2))


# In[51]:


learn.fit_one_cycle(4, slice(1e-2))


# In[52]:


learn.fit_one_cycle(4, slice(1e-2))


# In[53]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[54]:


interp.plot_top_losses(9, figsize=(9,7))


# In[55]:


interp.plot_confusion_matrix()


# In[56]:


learn.recorder.plot_losses()


# In[57]:


learn.recorder.plot_lr()


# In[58]:


learn.save('stage-3')


# In[67]:


learn.unfreeze()


# In[68]:


learn.lr_find()
learn.recorder.plot()


# In[69]:


learn.fit_one_cycle(4, slice(1e-5, 1e-2/5))


# In[70]:


learn.fit_one_cycle(4, slice(1e-5, 1e-2/5))


# In[71]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[72]:


interp.plot_top_losses(9, figsize=(9,7))


# In[73]:


interp.plot_confusion_matrix()


# In[74]:


learn.recorder.plot_losses()


# In[75]:


learn.recorder.plot_lr()


# In[76]:


learn.save('stage-4')


# In[77]:


learn.export("/export.pkl")


# In[78]:


predictor = load_learner("/", test=ImageList.from_df(df, path))


# In[79]:


preds_train, y_train, losses_train  = predictor.get_preds(ds_type=DatasetType.Test, with_loss=True)
preds_train[:5], y_train[:5], losses_train[:5]


# In[80]:


y_train = torch.argmax(preds_train, dim=1)


# In[81]:


interp = ClassificationInterpretation(predictor, preds_train, tensor(df.has_cactus.values), losses_train)


# In[82]:


interp.plot_confusion_matrix()


# In[83]:


from sklearn.metrics import roc_auc_score
def roc_auc(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


# In[84]:


roc_auc(y_train, df.has_cactus.values)


# In[85]:


predictor = load_learner("/", test=ImageList.from_folder(path/'test/test'))


# In[86]:


preds_test, y_test, losses_test  = predictor.get_preds(ds_type=DatasetType.Test, with_loss=True)
preds_test[:5], y_test[:5], losses_test[:5]


# In[87]:


y_test = torch.argmax(preds_test, dim=1)
y_test


# In[88]:


sub_df = pd.DataFrame({'id': os.listdir(path/'test/test'), 
                         'has_cactus': y_test})


# In[89]:


sub_df.head()


# In[90]:


sub_df.to_csv('submission-v2.csv', index=False)

