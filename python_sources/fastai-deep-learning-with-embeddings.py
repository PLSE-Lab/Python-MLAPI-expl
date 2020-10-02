#!/usr/bin/env python
# coding: utf-8

# # [Tabular learning](https://docs.fast.ai/tabular.html) 
# I wanted to go a slightly different route from the LGBM that seems to be favorite here.
# I don't expect too great results but maybe I can average with LGBM results later to get a few precent more out of my other models.
# ## Deep in this case means embeddings+2 layers
# Not very deep I know I know, but lets see maybe Ill add some if that increases performance.
# The Idea is to use the continuous variables as they are. 
# The categorical variables are in a first layer transformed in a lower dimensional space via embeddings with dropout.
# The embeddings also use droput and the continuous variables batchnorm.
# After that blocks of BatchNorm, Dropout, Linear and ReLU.
# The Embedding sizes I chose to sqrt(n_cat) or a maximum of 100.
# 
# The preprocessing is done in a different kernel. Just standard stuff and I added a column for the date that can be decoded from OsBuildLab.
# I plan to add new features when I set the pipeline for my lgbm and this model up.
# In this kernel, I still need to add the prediction for the test case so I can submit this.
# 
# I would additionally like to explore the embeddings a bit more since one can make pretty plots of the relationship of for instance dates of weekdays vs weekends, or gaming vs non gaming.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from fastai import *
from fastai.tabular import *

from sklearn.metrics import roc_auc_score

from tqdm import tqdm


# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[ ]:


path=Path('../input')
fe_path=path/'feature-engineering-msmw'
comp_path=path/'microsoft-malware-prediction'
hp_path=path/'lgbm-hyperopt'


# In[ ]:


cats=np.load(fe_path/'categories.npy').item()
means,stds=np.load(fe_path/'means.npy').item(),np.load(fe_path/'stds.npy').item()


# In[ ]:


index='MachineIdentifier'
dep_var='HasDetections'


# In[ ]:


cont_names=list(means.keys())
cat_names=list(cats.keys())


# In[ ]:


feature_imp=pd.read_pickle(hp_path/'feature_importance')


# In[ ]:


idx=np.argsort(feature_imp.mean(axis=1))


# In[ ]:


frac=0.01
frac_important=np.cumsum(feature_imp.mean(axis=1)[idx])/np.max(np.sum(feature_imp.mean(axis=1)[idx]))
idx_important=np.where(frac_important>frac)[0]


# In[ ]:


feature_cols=feature_imp.iloc[idx[idx_important]].index
cat_names=[cat for cat in cat_names if cat in feature_cols]
cont_names=[cont for cont in cont_names if cont in feature_cols]

features=[index]+cat_names+cont_names+[dep_var]


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
p1 = plt.barh( idx_important,feature_imp.mean(axis=1)[idx[idx_important]], xerr=feature_imp.std(axis=1)[idx[idx_important]],orientation ='horizontal')
plt.yticks(idx_important, feature_imp.mean(axis=1).index[idx[idx_important]]);
plt.xscale('log')


# In[ ]:


df_trn=pd.read_hdf(fe_path/'train.h5',columns=features)


# In[ ]:


df_trn.set_index('MachineIdentifier',inplace=True)


# Validation set is therefore about 10% of the whole dataset

# In[ ]:


#df_trn=df_trn.iloc[:int(0.2*len(df_trn))]


# In[ ]:


N_trn=int(0.9*len(df_trn))
valid_idx =range(N_trn,len(df_trn))# np.argsort(df_trn.OSVersion_Elapsed)[N_trn:]
bs=1024


# This part creates the dataset as well as a fancy pytorch dataloaderfor training and validation set.

# In[ ]:


data = TabularDataBunch.from_df('.', df_trn, dep_var, valid_idx=valid_idx, cat_names=cat_names,bs=bs)#,test_df=df_test)


# Based on the categorical and continuous variables a model of a certain size is created. Although its a 0 or 1 case out_sz has to be 2. 

# In[ ]:


model = TabularModel(data.get_emb_szs(), n_cont=len(data.cont_names), out_sz=2, layers=[2000,4000,1000], ps=None, emb_drop=0.5,
                         y_range=[0,1], use_bn=True)


# Pretty stupid way of defining the metric but I couldn't make it work otherwise to work with the fastai library

# In[ ]:


def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score


# Now combining the data and model with a learner object.
# This one defines metric, loss function, a learning rate scheduler and handles all details about model saving including handling the predictions on a specific dataset (for now validation but I'll add test set soon)

# In[ ]:


data.show_batch()


# In[ ]:


learn=Learner(data, model,metrics=[auc_score,accuracy])


# In[ ]:


learn.lr_find(num_it=1000)
learn.recorder.plot()


# In[ ]:


learn.save('untrained')


# In[ ]:


learn.fit_one_cycle(10, 1e-2)


# In[ ]:


learn.save('embeddings-stage-1')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


plt.plot(learn.recorder.losses)


# In[ ]:


plt.plot(learn.recorder.lrs)


# In[ ]:


(x_cat,x_cont),y=data.one_batch(detach=False)

import matplotlib.gridspec as gridspec

n_plot=int(np.sqrt(len(cat_names)))-1
learn.load('untrained')
plt.figure(figsize=(16,16))
gs = gridspec.GridSpec(n_plot,n_plot)
for i in range(n_plot):
    for j in range(n_plot):
        ax = plt.subplot(gs[i,j])
        x_embed=to_np(learn.model.embeds[j+i*n_plot](x_cat[:,j+i*n_plot]))
        X_embedded = PCA(n_components=2).fit_transform(x_embed)
        color=to_np(x_cat[:,j+i*n_plot])
        color=color/np.max(color)
        ax.scatter(X_embedded[:,0],X_embedded[:,1],c=color)
        ax.set_title(cat_names[j+i*n_plot])


# In[ ]:


(x_cat,x_cont),y=data.one_batch(detach=False)

import matplotlib.gridspec as gridspec

n_plot=int(np.sqrt(len(cat_names)))-1
learn.load('embeddings-stage-1')

plt.figure(figsize=(16,16))
gs = gridspec.GridSpec(n_plot,n_plot)
for i in range(n_plot):
    for j in range(n_plot):
        ax = plt.subplot(gs[i,j])
        x_embed=to_np(learn.model.embeds[j+i*n_plot](x_cat[:,j+i*n_plot]))
        X_embedded = PCA(n_components=2).fit_transform(x_embed)
        color=to_np(x_cat[:,j+i*n_plot])
        color=color/np.max(color)
        ax.scatter(X_embedded[:,0],X_embedded[:,1],c=color)
        ax.set_title(cat_names[j+i*n_plot])


# In[ ]:


y_pred,y_true=learn.get_preds()


# In[ ]:


scr=to_np(auc_score(y_pred,y_true))


# In[ ]:


f"{scr:0.3}"


# In[ ]:


learn.show_results(rows=20)


# In[ ]:


df_sub=pd.read_csv(comp_path/'sample_submission.csv').set_index('MachineIdentifier')


# In[ ]:


features.remove(dep_var)


# In[ ]:


chk_size=1000*1024
df_test_iter=pd.read_hdf(fe_path/'test.h5',iterator=True, chunksize=chk_size, columns=features)


# In[ ]:


for df_test in tqdm(df_test_iter):
    df_test.set_index('MachineIdentifier',inplace=True)
    for col in cat_names:
        df_test[col]=df_test[col].cat.codes
    x_cat_np=df_test.loc[:,cat_names].values.astype(np.int64)+1
    x_cont_np=df_test.loc[:,cont_names].values.astype(np.float32)
    for idxs in np.array_split(range(df_test.shape[0]),chk_size//bs):
        x_cat=to_device(tensor(x_cat_np[idxs,:]),'cuda')
        x_cont=to_device(tensor(x_cont_np[idxs,:]),'cuda')
        pred=learn.model(x_cat,x_cont)
        df_sub.loc[df_test.index[idxs],'HasDetections']=to_np(pred)[:,1]


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(df_sub)


# In[ ]:


df_sub.head()


# In[ ]:


df_sub.to_csv('submission')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




