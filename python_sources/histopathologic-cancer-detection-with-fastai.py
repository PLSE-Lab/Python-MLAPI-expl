#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


path = Path('/kaggle/input/histopathologic-cancer-detection/')
path.ls()


# In[ ]:


path_img_train= path/'train'


# In[ ]:


path_img_test=path/'test'


# In[ ]:


path_label=path/'train_labels.csv'


# In[ ]:


df_label=pd.read_csv(path_label)


# In[ ]:


df_label.head()


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,
                      max_lighting=0.05, max_warp=0.)


# In[ ]:


doc(get_transforms)


# In[ ]:


data = ImageDataBunch.from_csv(path,folder='train',csv_labels=path_label,ds_tfms=tfms, size=90, suffix='.tif',test=path_img_test,bs=64);
stats=data.batch_stats()        
data.normalize(stats)


# In[ ]:


data.show_batch(rows=5, figsize=(12,9))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


data


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# **how to show document in fastai!**

# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12),dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.model_dir=Path('/kaggle/working')
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.save('stage1')


# Please train the whole model ..unfreeze!

# learn unfreeze says please train the whole model! NOT just some specific layer in the MODEL?

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# so after unfreeze it should be worst the result, because now we are training from scratch or training the whole network instead of just the last layer. Remember we are trying to use the transfer learning concept here. The network that are being trained for say eye, is kind of the same as the network trained for cat. So, we don't have to reinvent to wheel.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# this bring back the model we save earlier NOT weights!

# In[ ]:


learn.load('stage1')


# i don't get this, why the plot looks the same even after i load the old model? WHY? whatever

# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# OK let's do the prediction

# In[ ]:


preds,y=learn.get_preds()


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score


# In[ ]:


pred_score=auc_score(preds,y)
pred_score


# In[ ]:


preds,y=learn.TTA()
pred_score_tta=auc_score(preds,y)
pred_score_tta


# In[ ]:


preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')
sub.head()


# In[ ]:


clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])
fname_cleaned=clean_fname(data.test_ds.items)
fname_cleaned=fname_cleaned.astype(str)


# In[ ]:


sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])
sub.to_csv(f'submission_{pred_score}.csv')


# In[ ]:




