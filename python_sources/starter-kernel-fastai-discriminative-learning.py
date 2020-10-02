#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# ## Loading data
# 
# We will be reducing Image size=124px at first and start training at first. THis makes training our neural nets whole lot faster

# In[ ]:


path = "../input"
SEED = 42
sz = 124
bs = 64
test_df = pd.read_csv(f"{path}/sample_submission.csv")
sub_df = pd.read_csv(f"{path}/sample_submission.csv")


# In[ ]:


data = ImageDataBunch.from_folder(path, valid_pct=0.1, ds_tfms=get_transforms(), size=sz, bs=bs, seed=SEED).normalize(imagenet_stats)
data.add_test(ImageList.from_df(test_df, path, folder="test/test"))


# In[ ]:


#Showing Images
data.show_batch(rows=3, figsize=(5,5))


# ## Training The Model

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir="/tmp/model/")


# In[ ]:


# Plotting to find the ideal learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-1/2


# In[ ]:


learn.fit_one_cycle(3, slice(lr))


# In[ ]:


# Saving the model
learn.save('stage1')


# ## Retraining the last layers
# 
# This technique of training model with the original size is mentioned in @jeremyhowards [fastai Lecture3](https://course.fast.ai/videos/?lesson=3). This gives us a 
# better neural net whcih can be trained at less time and is more accurate always :)

# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3/3


# In[ ]:


learn.fit_one_cycle(2, slice(lr))


# ## Generate predictions

# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


sub_df = pd.read_csv(f"{path}/sample_submission.csv")
sub_df.predicted_class = test_preds
sub_df.to_csv("submission.csv", index=False)


# In[ ]:


sub_df.head()


# # fin.
