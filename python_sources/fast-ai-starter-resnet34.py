#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable

import sys
sys.path.insert(0, '../input/aptos2019-blindness-detection')


# In[ ]:


# copy pretrained weights for resnet34 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')


# In[ ]:


df = pd.read_csv(PATH/'train.csv')
df.head()


# In[ ]:


df.diagnosis.value_counts()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Try Oversampling\n\nres = None\nsample_to = df.diagnosis.value_counts().max()\n\nfor grp in df.groupby('diagnosis'):\n    n = grp[1].shape[0]\n    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)\n    rows = pd.concat((grp[1], additional_rows))\n    \n    if res is None: res = rows\n    else: res = pd.concat((res, rows))")


# In[ ]:


res.diagnosis.value_counts()


# In[ ]:


src = (
    ImageList.from_df(res,PATH,folder='train_images',suffix='.png')
#         .use_partial_data(0.1)
        .split_by_rand_pct()
        .label_from_df()
    )
src


# In[ ]:


data = (
    src.transform(get_transforms(),size=128)
    .databunch()
    .normalize()
)
data


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"
learn = cnn_learner(data,models.resnet34,metrics=[accuracy,kappa],model_dir='/kaggle')


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


learn.fit_one_cycle(4,slice(1e-6,1e-3))


# In[ ]:


# # progressive resizing
# learn.data = data = (
#     src.transform(get_transforms(),size=224)
#     .databunch()
#     .normalize()
# )
# learn.freeze()
# # learn.lr_find()
# # learn.recorder.plot()


# In[ ]:


# learn.fit_one_cycle(2,3e-4)


# In[ ]:


# learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


# learn.fit_one_cycle(2,slice(1e-6,3e-5))


# In[ ]:


sample_df = pd.read_csv(PATH/'sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

