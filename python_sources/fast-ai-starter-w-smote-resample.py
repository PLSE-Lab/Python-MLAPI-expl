#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
from fastai.tabular import *
from fastai.callbacks import ReduceLROnPlateauCallback,EarlyStoppingCallback
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))


# In[ ]:


class roc(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.total = 0
        self.batch_count = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = F.softmax(last_output, dim=1)
        roc_score = roc_auc_score(to_np(last_target), to_np(preds[:,1]))
        self.total += roc_score
        self.batch_count += 1
    
    def on_epoch_end(self, num_batch, **kwargs):
        self.metric = self.total/self.batch_count


# ## Load Data

# In[ ]:


train = pd.read_csv("../input/train.csv").drop('ID_code',axis=1)
test = pd.read_csv("../input/test.csv")


# In[ ]:


sm = SMOTE(random_state = 21)
x_res, y_res = sm.fit_resample(train.iloc[:,1:], train['target'])


# In[ ]:


dep_var = 'target'
cont_names = train.iloc[:,1:].columns.tolist()


# In[ ]:


train_res = pd.DataFrame(data = x_res,columns = cont_names)
train_res['target'] = y_res


# In[ ]:


sns.set(style="darkgrid")
ax =  sns.countplot(x="target", data=train_res)


# In[ ]:


procs = [Normalize]


# In[ ]:


test = TabularList.from_df(test,cont_names=cont_names)


# In[ ]:


data = (TabularList.from_df(train_res,cont_names=cont_names, procs=procs)
                           .random_split_by_pct(0.15)
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch(bs=32))


# In[ ]:


learn = tabular_learner(data, 
                        layers=[200,100], 
                        metrics=[accuracy,roc()]
                       )


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


##ES = EarlyStoppingCallback(learn, monitor='roc',patience = 5)
##RLR = ReduceLROnPlateauCallback(learn, monitor='roc',patience = 2)


# In[ ]:


learn.fit(1, .01,)


# In[ ]:


preds,_ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds = preds.tolist()


# In[ ]:


preds_nn = []
for i in range(len(preds)):
    preds_nn.append(preds[i][0])


# In[ ]:


subby = pd.read_csv("../input/test.csv")
subby['target'] = preds_nn


# In[ ]:


subby = subby[['ID_code','target']]


# In[ ]:


subby.to_csv('subby.csv',index=False)


# In[ ]:


subby.head()


# In[ ]:




