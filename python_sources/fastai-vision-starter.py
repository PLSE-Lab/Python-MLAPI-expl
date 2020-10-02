#!/usr/bin/env python
# coding: utf-8

# I don't know how to use any advanced fastai techniques, so I just went as simple as possible in this notebook.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 16


# In[ ]:


import os
from pathlib import Path
print(os.listdir("../input"))
work_p = Path("./")
p = Path("../input")


# In[ ]:


len(os.listdir(p/"test"))


# This one won't work since we need write permissions. To fix, images should be moved to {work_p}

# In[ ]:


# folders = ["train", "test"]
# for c in classes:
#     print(c)
#     verify_images(p/c, delete=False, max_size=500)


# ## Data pipeline

# In[ ]:


train_df = pd.read_csv(p/"train.csv")#.sample(frac=0.3, random_state=2)
print(train_df.shape); train_df.head()


# In[ ]:


labels_count = train_df.Id.value_counts()
train_names = train_df.index.values


# I have no idea how to properly create a validation set, so I just duplicate every single occurence

# In[ ]:


for idx,row in train_df.iterrows():
    if labels_count[row['Id']] < 2:
        for i in row*math.ceil((2 - labels_count[row['Id']])/labels_count[row['Id']]):
            train_df = train_df.append(row,ignore_index=True)

print(train_df.shape)
# plt.hist(train_df.Id.value_counts()[1:],bins=100,range=[0,100]);
# plt.hist(train_df.Id.value_counts()[1:],bins=100,range=[0,100]);


# # Training

# In[ ]:


name = f'res50-full-train'


# In[ ]:


np.random.seed(2)
data = (ImageDataBunch.from_df(work_p, train_df, folder=p/"train", test=p/"test", valid_pct=0.20, ds_tfms=get_transforms(), size=224, bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# Looks like default transformation is not alright. Let's give it a go anyway.

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(2, max_lr=slice(6.31e-07, 3e-07))


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(2, max_lr=3e-03)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(2, max_lr=0.5e-02)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(2, max_lr=slice(5e-02, 2.5e-02))


# In[ ]:


# learn.save("stage-1")


# In[ ]:


# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(1)


# In[ ]:


# learn.load('stage-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Bigger size doesn't fit the kernel

# In[ ]:


# data_bigger = ImageDataBunch.from_df(work_p, train_df, folder=p/"train", valid_pct=0.20, ds_tfms=get_transforms(), size=448, bs=bs).normalize(imagenet_stats)
# learn_bigger = create_cnn(data_bigger, models.resnet34, metrics=error_rate)


# In[ ]:


# learn_bigger.fit_one_cycle(4)


# In[ ]:


# data_bigger.show_batch(rows=3, figsize=(7,6))


# # Interpretation

# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)
# losses,idxs = interp.top_losses()
# len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


# interp.plot_top_losses(9, figsize=(15,11))


# # Unfreezing, fine-tuning, and learning rates

# In[ ]:


# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(1)


# In[ ]:


# learn.load('stage-1');


# In[ ]:


# learn.lr_find()


# In[ ]:


# learn.recorder.plot()


# In[ ]:


# learn.unfreeze()
# learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))


# # Submission

# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)


# In[ ]:


def top_5_pred_labels(preds, classes):
    top_5 = np.argsort(preds.numpy())[:, ::-1][:, :5]
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels

def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'{name}.csv', index=False)


# In[ ]:


create_submission(preds, learn.data, name, learn.data.classes)


# In[ ]:




