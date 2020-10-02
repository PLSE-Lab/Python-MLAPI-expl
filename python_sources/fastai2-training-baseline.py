#!/usr/bin/env python
# coding: utf-8

# # fastai2 training baseline
# ## If you found this helpful, please upvote!

# In[ ]:


get_ipython().system('pip install --upgrade fastai2 > /dev/null')
get_ipython().system('pip install --upgrade fastcore > /dev/null')
get_ipython().system('pip install pretrainedmodels > /dev/null')


# In[ ]:


from fastai2.vision.all import *
import pretrainedmodels


# In[ ]:


panda_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                        splitter=RandomSplitter(),
                        get_x=ColReader('image_id',pref=Path('../input/panda-challenge-512x512-resized-dataset'),suff='.jpeg'),
                        get_y=ColReader('isup_grade'),
                        item_tfms=Resize(256),
                        batch_tfms=aug_transforms()
                       )


# In[ ]:


train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


dls = panda_block.dataloaders(train_df,bs=16)


# In[ ]:


dls.show_batch()


# In[ ]:


m = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
children = list(m.children())
head = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), 
                                  nn.Linear(children[-1].in_features,200))
model = nn.Sequential(nn.Sequential(*children[:-2]), head)


# In[ ]:


learn = Learner(dls,model,splitter=default_split,metrics=[accuracy,CohenKappa(weights='quadratic')])


# In[ ]:


learn.freeze()
learn.lr_find()


# In[ ]:


learn.freeze()
learn.fit_one_cycle(5,6e-3)


# In[ ]:


learn.save('stage-1-512.pth')


# In[ ]:


learn.load('stage-1-512.pth')


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5,slice(1e-6,5e-5))


# In[ ]:


learn.save('stage-2-512.pth')


# In[ ]:


learn.recorder.plot_loss()


# In[ ]:


learn.export()


# How to improve:
# - The dataset used (resized 512x512) is terrible. Need to determine how to create a better dataset.
