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


# !pip install -q fastai2
get_ipython().system('pip install -q git+https://github.com/fastai/fastai2')
get_ipython().system('pip install -q git+https://github.com/fastai/fastcore')


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


# LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


# BS = 128
BS = 8


# In[ ]:


def get_data(size=224):
    return DataBlock(blocks    = (ImageBlock, CategoryBlock),
                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),
                       get_y=lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0],
                       splitter=RandomSplitter(seed=42),
                       item_tfms=Resize(size),
                       batch_tfms=aug_transforms(flip_vert=True),
                      ).dataloaders(train_df, bs=BS)


# In[ ]:


dls = get_data((450, 800))


# In[ ]:


# dblock.summary(train_df)


# In[ ]:


# dsets = dblock.datasets(train_df)
# dsets.train[0]


# In[ ]:


# 1/0


# In[ ]:


# del learn


# In[ ]:


# import gc
# gc.collect()


# In[ ]:


# dls = dblock.dataloaders(train_df, bs=BS)
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


# from fastai2.callback.cutmix import CutMix


# In[ ]:


# loss = partial(CrossEntropyLossFlat, weights=tensor([1,1.5,1,1]))


# In[ ]:


metric = partial(AccumMetric, flatten=False)

def get_learner(size=224):
    dls = get_data(size)
    return cnn_learner(dls, resnet152, metrics=[
                        error_rate,
                        metric(healthy_roc_auc),
                        metric(multiple_diseases_roc_auc),
                        metric(rust_roc_auc),
                        metric(scab_roc_auc),
                        metric(roc_auc)],
                       ).to_fp16()


# In[ ]:


learn = get_learner((450, 800))


# In[ ]:


# learn.lr_find()


# In[ ]:


lr = 3e-3


# In[ ]:


# learn.fine_tune(1, lr)


# In[ ]:


# learn.fit_one_cycle(2, slice(lr/10, lr))


# In[ ]:


m = "multiple_diseases_roc_auc"
d = 0.005
learn.fine_tune(50, lr, freeze_epochs=1, cbs=[EarlyStoppingCallback(monitor=m, min_delta=d, patience=10),
                                              SaveModelCallback(monitor=m, min_delta=d),
                                              ReduceLROnPlateau(monitor=m, min_delta=d, patience=4)])


# In[ ]:


# learn.fit_one_cycle(2, slice(lr/1000, lr/100))


# In[ ]:


# learn.data = get_data(448)


# In[ ]:


# learn.fit_one_cycle(5, slice(lr/1000, lr/100))


# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15, 10))


# # Submission

# In[ ]:


test_df = pd.read_csv(path/"test.csv")
test_df.head()


# In[ ]:


tst_dl = learn.dls.test_dl(test_df)


# In[ ]:


preds, _ = learn.get_preds(dl=tst_dl)


# In[ ]:


subm = pd.read_csv(path/"sample_submission.csv")


# In[ ]:


subm.iloc[:, 1:] = preds


# In[ ]:


subm.to_csv("submission.csv", index=False)


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:




