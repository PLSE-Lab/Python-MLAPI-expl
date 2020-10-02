#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


PATH = "../input"
SEED = 42
sz = 224
bs = 64

sub_df = pd.read_csv(f"{PATH}/sample_submission.csv")
test_df = pd.read_csv(f"{PATH}/sample_submission.csv")


# ## Train Model

# In[ ]:


data = ImageDataBunch.from_folder(PATH, valid_pct=0.1, ds_tfms=get_transforms(), size=sz, bs=bs, seed=SEED)
data.add_test(ImageList.from_df(test_df, PATH, folder="test/test"))

learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(1)


# ## Generate Predictions

# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


sub_df.predicted_class = test_preds
sub_df.to_csv("submission.csv", index=False)


# In[ ]:


sub_df.head()


# In[ ]:




