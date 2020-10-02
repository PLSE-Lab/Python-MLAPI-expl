#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.tabular import *


# The train_df & test_df is borrowed from this wonderful [kernel](https://www.kaggle.com/gpreda/elo-world-high-score-without-blending)  by Gabriel. Thank you very much. The fastai library used here is from course V3. This complete kernel is heavily copied from ahem.. inspired from [here](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb). Special thanks to Jeremy for the wonderful course :)

# In[ ]:


train_df = pd.read_csv('../input/elo-world-high-score-without-blending/train_df.gz', compression='gzip', header=0,
                      sep=',',quotechar='"')


# In[ ]:


test_df = pd.read_csv('../input/elo-world-high-score-without-blending/test_df.gz', compression='gzip', header=0,
                      sep=',',quotechar='"')


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


df_indep = train_df.drop('target',axis=1)
n_valid = 40000 #~20% of the training set
n_trn = len(train_df)-n_valid


# In[ ]:


n_trn, n_valid


# We add embeddings for all the categorical fields. 

# In[ ]:


cat_flds = [n for n in df_indep.columns if train_df[n].nunique()<50 and n != 'outliers']
','.join(cat_flds)


# In[ ]:


len(cat_flds)


# In[ ]:


for df in [train_df, df_indep, test_df]:
    df.drop('first_active_month', axis=1, inplace=True)


# In[ ]:


for df in [train_df, df_indep, test_df]:
    for n in cat_flds: 
        df[n] = df[n].astype('category').cat.as_ordered()
    df['card_id'] = df['card_id'].astype('category').cat.as_ordered()
    df['card_id_code'] = df.card_id.cat.codes
    df.drop('card_id', axis=1, inplace=True)


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


cont_flds = [n for n in df_indep.columns if n not in cat_flds and n!= 'outliers']
','.join(cont_flds)


# In[ ]:


procs=[FillMissing, Categorify, Normalize] #self-explanatory - neural nets like normalised values


# In[ ]:


len(cont_flds), len(cat_flds)


# In[ ]:


dep_var = 'target'
df = train_df[cat_flds + cont_flds + [dep_var]].copy()


# In[ ]:


df[dep_var].head()


# In[ ]:


path = Path('../input/') #we need to give some path - doesn't matter


# In[ ]:


data = (TabularList.from_df(df, path=path, cat_names=cat_flds, cont_names=cont_flds, procs=procs)
                   .split_by_idx(range(n_valid))
                   .label_from_df(cols=dep_var, label_cls=FloatList, log=False)
                   .databunch())


# In[ ]:


min_y = np.min(train_df['target'])*1.2


# In[ ]:


max_y = np.max(train_df['target'])*1.2


# In[ ]:


y_range = torch.tensor([min_y, max_y], device=defaults.device)


# We added 2 hidden layers of size 1000,500 with dropouts .001, .01 respectively. The size of embeddings is chosen at half of the unique categories capped at 50 I afaik.

# In[ ]:


learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=rmse)


# You can see the embeddings & our added hidden input layers with dropout & batchnorm below:

# In[ ]:


learn.model


# In[ ]:


len(data.train_ds.cont_names)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# We picked 1e-2 because that's where the loss is converging fastest

# In[ ]:


learn.fit_one_cycle(5, 1e-2, wd=0.2)


# In[ ]:


learn.recorder.plot_losses()


# The loss can be consistent if we use stratified kfold instead of using first 40k rows.

# In[ ]:


# learn.fit_one_cycle(5, 3e-4)


# In[ ]:


# learn.fit_one_cycle(4, 5e-2, wd=0.2)


# In[ ]:


# learn.fit_one_cycle(3, 1e-2, wd=0.2)


# In[ ]:


learn.predict(df.iloc[0])


# In my local machine I've used older fastai version and trained using dataset split by stratified kfold. Due to some local GPU problem I wasn't able to output the predictions to csv. I think this solution might work good if you blend with your existing solutions. Do let me know the comments if you found it useful. Also let me know if you need a pytorch only version of this kernel.

# In[ ]:




