#!/usr/bin/env python
# coding: utf-8

# # Fastai starter
# 
# This is some basic starter code for using [fastai](https://docs.fast.ai/) for this dataset. The code/model is based on [this kernel](https://www.kaggle.com/xhlulu/densenet-transfer-learning-iwildcam-2019) and uses a pretrained DenseNet121, along with [Mixup](https://arxiv.org/abs/1710.09412) as implemented by the fastai library.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd


# In[ ]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[ ]:


path = Path('../input')


# In[ ]:


# Load train dataframe
train_df = pd.read_csv(path/'train.csv')
train_df = pd.concat([train_df['id'],train_df['category_id']],axis=1,keys=['id','category_id'])
train_df.head()


# In[ ]:


# Load sample submission
test_df = pd.read_csv(path/'test.csv')
test_df = pd.DataFrame(test_df['id'])
test_df['predicted'] = 0
test_df.head()


# In[ ]:


train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.jpg') 
               for df, folder in zip([train_df, test_df], ['train_images', 'test_images'])]
data = (train.split_by_rand_pct(0.2, seed=123)
        .label_from_df(cols='category_id')
        .add_test(test)
        .transform(get_transforms(), size=32)
        .databunch(path=Path('.'), bs=64).normalize())


# In[ ]:


data.show_batch()


# ## Train model
# 
# Here we will create a model using the [`cnn_learner`](https://docs.fast.ai/vision.learner.html#cnn_learner) function, which will automatically download and load the pretrained weights. Mixup is easily added as a callback, which is done by the `mixup()` function. If you are interested in the mixup implementation in fastai, you can read more over [here](https://docs.fast.ai/callbacks.mixup.html).
# 
# We will fine-tune the pretrained model, then unfreeze and train the whole model.

# In[ ]:


learn = cnn_learner(data, base_arch=models.densenet121, metrics=[FBeta(),accuracy], wd=1e-5).mixup()


# The fastai library provides an implementation of a learning rate finder as described by [this paper](https://arxiv.org/abs/1506.01186). This allows us to choose the optimal learning rate for efficient training.
# 
# In a nutshell, the learning rate is adjusted over a single epoch, and the loss is plotted against the learning rate. The optimal learning rate is when the loss decreases the fastest.

# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# We will now fine-tune the final layer of our pretrained model.

# In[ ]:


lr = 2e-2
learn.fit_one_cycle(2, slice(lr))


# In[ ]:


learn.save('stage-1-sz32')


# We now will unfreeze the model, to retrain the entire model. The optimal learning rate has to be determined again.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Here, we use discriminative learning rates, where lower learning rates are used for the earlier layers in the model.

# In[ ]:


lr = 1e-3
learn.fit_one_cycle(4, slice(lr/100, lr))


# In[ ]:


learn.save('stage-2-sz32')


# ## Test predictions

# In[ ]:


test_preds = learn.TTA(ds_type = DatasetType.Test)
test_df['predicted'] = test_preds[0].argmax(dim=1)


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# ## Future work:
# 
# Some more tricks include:
# - label smoothing
# - Focal loss
# 
# Also, it would be helpful to implement cross-validation, and try some other pretrained models and do ensembling.
# 
# If you enjoyed this kernel, please give it an upvote! Thanks for reading!
# 
