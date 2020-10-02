#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai2 --quiet')


# In[ ]:


from fastai2.vision.all import *


# In[ ]:


path = Path('../input/plant-pathology-2020-fgvc7')


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')


# In[ ]:


train_df.head()


# In[ ]:


imgs = get_image_files(path/"images")


# In[ ]:


train_df.iloc[0, 1:][train_df.iloc[0, 1:] == 1].index[0]


# In[ ]:


lbl_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


def get_data(idx, size=224, bs=64):
    dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),
                       get_y=Pipeline([lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0]]),
                       splitter=IndexSplitter(idx),
                       item_tfms=RandomResizedCrop(size+64),
                       batch_tfms=[*aug_transforms(size=size, flip_vert=True),
                                   Normalize.from_stats(*imagenet_stats)],
                      )
    return dblock.dataloaders(train_df, bs=bs)


# In[ ]:


from sklearn.metrics import roc_auc_score

def roc_auc(preds, targs, labels=range(4)):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return roc_auc(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return roc_auc(*args, labels=[1])

def rust_roc_auc(*args):
    return roc_auc(*args, labels=[2])

def scab_roc_auc(*args):
    return roc_auc(*args, labels=[3])


# In[ ]:


metric = partial(AccumMetric, flatten=False)


# In[ ]:


metrics=[
            error_rate,
            metric(healthy_roc_auc),
            metric(multiple_diseases_roc_auc),
            metric(rust_roc_auc),
            metric(scab_roc_auc),
            metric(roc_auc)]


# In[ ]:


import gc


# In[ ]:


dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),
                       get_y=Pipeline([lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0]]),
                       splitter=RandomSplitter(), 
                       item_tfms=RandomResizedCrop(224+64),
                       batch_tfms=[*aug_transforms(size=224, flip_vert=True),
                                   Normalize.from_stats(*imagenet_stats)],
                      )
dls =  dblock.dataloaders(train_df, bs=64)


# In[ ]:


gc.collect()


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


train_lbls = []
for _, lbl in dls.train.dataset:
    train_lbls.append(lbl)


# In[ ]:


for _, lbl in dls.valid.dataset:
    train_lbls.append(lbl)


# In[ ]:


imgs = train_df['image_id'].to_numpy()


# In[ ]:


tst_preds = []
skf = StratifiedKFold(n_splits=10, shuffle=True)
for _, val_idx in skf.split(imgs, np.array(train_lbls)):
    dls = get_data(val_idx, 128, 64)
    learn = cnn_learner(dls, resnet152, metrics=metrics)
    lr = 1e-3
    learn.fine_tune(1, lr)
    learn.save('initial')
    del learn
    torch.cuda.empty_cache()
    gc.collect()
    
    dls = get_data(val_idx, 256, 16)
    learn = cnn_learner(dls, resnet152, metrics=metrics)
    learn.load('initial');
    learn.freeze()
    learn.fine_tune(4, 1e-3)
    learn.save('v2')
    del learn
    torch.cuda.empty_cache()
    gc.collect()
    dls = get_data(val_idx, 448, 8)
    learn = cnn_learner(dls, resnet152, metrics=metrics)
    learn.load('v2');
    learn.unfreeze()
    learn.fit_one_cycle(3, slice(1e-5, 1e-4))
    tst_dl = learn.dls.test_dl(test_df)
    y, _ = learn.tta(dl=tst_dl)
    tst_preds.append(y)
    del learn
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


len(tst_preds)


# In[ ]:


tot = tst_preds[0]
for i in tst_preds[1:]:
    tot += i


# In[ ]:


tot = tot / 10


# In[ ]:


subm = pd.read_csv(path/"sample_submission.csv")


# In[ ]:


subm.iloc[:, 1:] = tot


# In[ ]:


subm.to_csv("submission3.csv", index=False)

