#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
path = "../input/picnic_hack"
os.listdir(path)

# Any results you write to the current directory are saved as output.


# In[2]:


from fastai.vision import *
import pandas as pd

tfms = get_transforms(flip_vert=True, max_lighting=0.15, max_zoom=1.1, max_warp=0.)
src = (ImageList.from_csv(path, 'train.csv', folder='train')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=','))
data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))


# In[3]:


data.show_batch(rows=3, figsize=(9,9))


# In[4]:


arch = models.resnet152
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score_01 = partial(fbeta, thresh=0.12)
f_score_02 = partial(fbeta, thresh=0.16)
f_score_03 = partial(fbeta, thresh=0.18)
f_score_04 = partial(fbeta, thresh=0.20)
learn = cnn_learner(data, arch, metrics=[f_score_01,f_score_02,f_score_03, f_score_04], model_dir="/tmp/model/")


# In[6]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 0.01
learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')
learn.export(file="/tmp/model/export.pkl")


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, lr/5))


# In[ ]:


learn.freeze()
learn.fit_one_cycle(3, slice(lr/10))


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, lr/10))


# In[ ]:


learn.freeze()
learn.fit_one_cycle(3, slice(lr/50))


# In[ ]:


learn.export(file="/tmp/model/export.pkl")


# TEST PREDICTIONS

# In[ ]:


test = ImageList.from_folder(path + '/test')
len(test)


# In[ ]:


learn_pred = load_learner("/tmp/model/", test=test)


# In[ ]:


preds, _ = learn_pred.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.12
labelled_preds = [','.join([learn_pred.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


fnames = [f.name for f in learn_pred.data.test_ds.items]
print(len(fnames))
labelled_preds[:5]


# In[ ]:


df = pd.DataFrame({'file':fnames, 'label':labelled_preds}, columns=['file', 'label'])
df.to_csv('submission.tsv', sep = '\t', index=False)
from IPython.display import FileLink
FileLink(f'submission.tsv')


# In[ ]:





# In[ ]:





# In[ ]:




