#!/usr/bin/env python
# coding: utf-8

# ### Set autoreload and defaults for notebook

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import from fastai vision

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# Let's peek inside of the training folder

# In[ ]:


get_ipython().system('ls ../input/train')


# 1. Set path
# 2. Import Transforms
# 3. Create databunch object.

# In[ ]:


path = Path("../input/")
tfms = get_transforms()
data = ImageDataBunch.from_folder(path,valid_pct=0.1,test="test", ds_tfms=tfms, size=224)


# - Create a learner object
# - Ideas: Use other resnets

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=error_rate)


# Idea: Fit more, Unfreeze and fit more

# In[ ]:


learn.fit_one_cycle(1)


# Let's peek into the sample submission

# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv').set_index('id')
sub.head()


# Dirt Logic: Don't copy blindly

# In[ ]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]

labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'id':fnames, 'predicted_class':labelled_preds}, columns=['id', 'predicted_class'])
df['id'] = df['id'].astype(str) + '.jpg'
df.to_csv('/kaggle/working/submission.csv', index=False)


# ## Finish

# In[ ]:




