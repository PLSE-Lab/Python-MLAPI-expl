#!/usr/bin/env python
# coding: utf-8

# This is a basic notebook to train a classifier using the fastai library. We will start with a Resnet18 that is light and fast to train.

# In[ ]:


import fastai
from fastai.vision import *


# Let's define some paths.

# In[ ]:


work_dir = Path('/kaggle/working/')
path = Path('../input')


# In[ ]:


train = 'train_images/train_images'
test =  path/'leaderboard_test_data/leaderboard_test_data'
holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'
sample_sub = path/'SampleSubmission.csv'
labels = path/'traininglabels.csv'


# In[ ]:


df = pd.read_csv(labels)
df_sample = pd.read_csv(sample_sub)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# There are some rows with low score, we will look into that later.

# In[ ]:


df[df['score']<0.75]


# There are 942 observed palmoil plantations, roughly 8% of total images.

# In[ ]:


(df.has_oilpalm==1).sum()


# We have to combine test and holdout for the submission

# In[ ]:


test_names = [f for f in test.iterdir()]
holdout_names = [f for f in holdout.iterdir()]


# We will use the [datablock](https://docs.fast.ai/data_block.html) API from fastai, it is so elegant! We create an ImageItemList to hold our data, using the `.from_df()`method. 
# Then we split the data in train/valid sets, I will use 0.2 (20% of data for validation) and a seed=2019, to be able to reproduce my results.
# Finally we add the test set, nice trick we just sum the lists to get the whole.

# In[ ]:


src = (ImageItemList.from_df(df, path, folder=train)
      .random_split_by_pct(0.2, seed=2019)
      .label_from_df('has_oilpalm')
      .add_test(test_names+holdout_names))


# We have to add some data augmentation, `get_transforms()` get us a basic set of data augmentations. And we set size=128 to train faster, you can test with 256, but have to reduce the batch size (bs=16). We have to normalize (substract the mean, divide by std) data for the GPU to work better, as I have not computed the actual value, I will just use ImageNet stats.

# In[ ]:


data =  (src.transform(get_transforms(), size=128)
         .databunch(bs=64)
         .normalize(imagenet_stats))


# Loot at data?

# In[ ]:


data.show_batch(3, figsize=(10,7))


# Let's impement the competition metric, luckyly it is already implemented in sklearn.  We have to modify it a little bit, 
# - First, fastai expects a pair (preds, targets) and sklearn expects (targets, preds)
# - Secondly, sklearn needs to vectors of equal shape. For our case, `preds` has shape (bs, 2), so we take the second column, the one that contains the probabilities of palmoil

# In[ ]:


#This was working perfectly some minutes ago!
from sklearn.metrics import roc_auc_score
def auc_score(preds,targets):
    return torch.tensor(roc_auc_score(targets,preds[:,1]))


# For some extrange reason thie metric does not always work as a callback for the learner.

# In[ ]:


learn = create_cnn(data, models.resnet18, 
                   metrics=[accuracy], #<---add aoc metric?
                   model_dir='/kaggle/working/models')


# ## Train
# First you have to compute the learning rate and choose the one where it is steeper.

# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


lr = 1e-2


# We will use `fit_one_cycle` to train the model, because it is awesome. First we train only the head of the model.

# In[ ]:


learn.fit_one_cycle(6, lr)


# In[ ]:


Then we unfreeze and train the whole model, with lower lr.


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-3))


# In[ ]:


learn.save('128')


# Let's compute our AUC over the validation set.

# In[ ]:


p,t = learn.get_preds()
auc_score(p,t)


# # View results
# 
# We can reviews our model, to see what it did worng. Probably some of this images even a human has a hard time evaluating.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# ## Sub file
# We have to create our sub file by concatenating both holdout and test names.

# In[ ]:


p,t = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


p = to_np(p); p.shape


# In[ ]:


ids = np.array([f.name for f in (test_names+holdout_names)]);ids.shape


# In[ ]:


#We only recover the probs of having palmoil (column 1)
sub = pd.DataFrame(np.stack([ids, p[:,1]], axis=1), columns=df_sample.columns)


# In[ ]:


sub.to_csv(work_dir/'sub.csv', index=False)


# In[ ]:




