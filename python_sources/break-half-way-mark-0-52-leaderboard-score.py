#!/usr/bin/env python
# coding: utf-8

# **This is a fork of @crazydiv fastai-starter kernel.**
# 
# You can easily get this score by making the following changes from the fastai-starter kernel:
# 
# - Use better Resnet Model(usually Resnet18< Resnet 34< Resnet 50< Resnet 101) give better accuracy and takes more time to train though
# - Use pretrained model(Don't forget to turn on Internet in the Kaggle Kernel section).
# 
# Also according @init27 it's highly recommended to watch Lecture 1 of fast.ai for this competition.
# 

# In[ ]:


from fastai.vision import *
from fastai.metrics import accuracy


# ## Data

# In[ ]:


# Copy for FastAI
get_ipython().system('mkdir -p data')
get_ipython().system('cp -R ../input/dsnet-kaggledays-hackathon/train/train data/train')
get_ipython().system('cp -R ../input/dsnet-kaggledays-hackathon/test/test data/test')
get_ipython().system('cp ../input/dsnet-kaggledays-hackathon/sample_submission.csv data/sample_submission.csv')


# In[ ]:


DATA_DIR = Path('data')


# In[ ]:


bs = 64


# In[ ]:


data = ImageDataBunch.from_folder(DATA_DIR, valid_pct=0.2, ds_tfms=get_transforms(), bs=bs, size=224).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# ## Model

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy)


# In[ ]:


learn.model


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-2/2)


# In[ ]:


# Save the model
learn.save('simple-model')


# In[ ]:


# Load the model
learn.load('simple-model');


# ## Inference

# In[ ]:


# Load submission file
sample_df = pd.read_csv(DATA_DIR/'sample_submission.csv')
sample_df.head()


# In[ ]:


# Generate test predictions
learn.data.add_test(ImageList.from_df(sample_df,DATA_DIR,folder='test'))


# In[ ]:


# Load up submission file
preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


# Convert predictions to classes
pred_classes = [data.classes[c] for c in list(preds.argmax(dim=1).numpy())]
pred_classes[:10]


# In[ ]:


# Add the prediction
sample_df.predicted_class = pred_classes
sample_df.head()


# In[ ]:


# Save the submission file
sample_df.to_csv('submission.csv',index=False)

from IPython.display import FileLink
FileLink('submission.csv')


# In[ ]:


# Clean up (for commit)
get_ipython().system('cp -R data/models models # Move the models out')
get_ipython().system('rm -rf data # Delete the data')


# ## More things you can do 
# 
# You can improve the accuracy even more by using following techniques in Deep Learning which usually works:
# - Using [Data augmentation](https://towardsdatascience.com/introduction-to-image-augmentations-using-the-fastai-library-692dfaa2da42)
# - Try out [Discriminative Learning rates](https://www.kaggle.com/kurianbenoy/starter-kernel-fastai-discriminative-learning)
# - And train for more epochs(isn't  it so obvious)
