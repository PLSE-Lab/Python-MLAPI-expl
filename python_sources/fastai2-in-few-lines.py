#!/usr/bin/env python
# coding: utf-8

# Thanks this guy for metrics
# 
# https://www.kaggle.com/lextoumbourou/plant-pathology-2020-eda-training-fastai2
# 
# Fastai2 docs
# 
# https://dev.fast.ai/

# In[ ]:


get_ipython().system('pip install -q fastai2')


# In[ ]:


from fastai2.vision.all import *


# In[ ]:


path = Path("/kaggle/input/plant-pathology-2020-fgvc7")


# In[ ]:


train_df = pd.read_csv(path/"train.csv")
train_df.head()


# In[ ]:


train_df.query("image_id == 'Train_5'")


# In[ ]:


get_image_files(path/"images")[5]


# In the train df all rows have only one label, so it looks like a usual classification.

# In[ ]:


train_df.iloc[0, 1:][train_df.iloc[0, 1:] == 1].index[0]


# I am using the default size and transforms. I don't know if they are the best for this problem.

# # Datablock

# In[ ]:


LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


def get_data(size=224):
    return DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),
                       get_y=lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0],
                       splitter=RandomSplitter(seed=42),
                       item_tfms=Resize(size),
                       batch_tfms=aug_transforms(flip_vert=True),
                      )


# In[ ]:


dblock = get_data()


# In[ ]:


dsets = dblock.datasets(train_df)
dsets.train[0]


# In[ ]:


BS = (1024 - 256)//8


# In[ ]:


dls = dblock.dataloaders(train_df, bs=BS)
dls.show_batch()


# # Model

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

learn = cnn_learner(dls, resnet152, metrics=[
            error_rate,
            metric(healthy_roc_auc),
            metric(multiple_diseases_roc_auc),
            metric(rust_roc_auc),
            metric(scab_roc_auc),
            metric(roc_auc)]
        ).to_fp16()


# In[ ]:


# learn.lr_find()


# In[ ]:


# 1/0


# In[ ]:


# del learn


# In[ ]:


# import gc
# gc.collect()


# In[ ]:


lr = 3e-3


# In[ ]:


learn.fine_tune(4, lr)


# # Submission

# In[ ]:


test_df = pd.read_csv(path/"test.csv")
test_df.head()


# In[ ]:


tst_dl = learn.dls.test_dl(test_df)


# In[ ]:


preds, y = learn.get_preds(dl=tst_dl)


# In[ ]:


preds


# In[ ]:


subm = pd.read_csv(path/"sample_submission.csv")


# In[ ]:


subm.iloc[:, 1:] = preds


# In[ ]:


subm.to_csv("submission.csv", index=False, float_format='%.10f')


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:




