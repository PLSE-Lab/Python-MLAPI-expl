#!/usr/bin/env python
# coding: utf-8

# This is a simple starter notebook for this challenge using the fastai2 library. 
# 
# As someone new to Deep Learning, I know how hard it is just to setup a working pipeline. Therefore, this is just a simple demonstraation of how you might do a whole pipeline from loading data until generating submissions.
# 
# Feel free to modify and improve on this code.
# 
# *PS: you might be able to get 0.6 - 0.7 accuracy using this (even more when training with the whole dataset, not just 10% like below and also training for more epochs)*

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai2 ')
from fastai2.vision.all import *


# In[ ]:


# train and test csv
train = pd.read_csv("/kaggle/input/shopee-product-detection-student/train.csv")
test = pd.read_csv("/kaggle/input/shopee-product-detection-student/test.csv")
# paths leading to images
train_path = Path("/kaggle/input/shopee-product-detection-student/train/train/train/")
test_path = Path("/kaggle/input/shopee-product-detection-student/test/test/test/")


# In[ ]:


# add the category to filename for easier usage with fastai API
train['filename'] = train.apply(lambda x: str(x.category).zfill(2) + '/' + x.filename, axis=1)
train


# In[ ]:


# train in a 10% subset of the data
# to speed up experimentation
# comment these lines out to increase accuracy (but necessitates longer training time)
from sklearn.model_selection import train_test_split
_, train = train_test_split(train, test_size=0.1, stratify=train.category)


# ## Loading Data

# We load the images easily using fastai2's API. We then crop and resize the image to 224x224 (the size the pre-trained model we will use was trained on) and then do some basic data augmentation and normalization (to ImageNet stats)

# In[ ]:


item_tfms = [RandomResizedCrop(224, min_scale=0.75)]
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
def get_dls_from_df(df):
    df = df.copy()
    options = {
        "item_tfms": item_tfms,
        "batch_tfms": batch_tfms,
        "bs": 32,
    }
    dls = ImageDataLoaders.from_df(df, train_path, **options)
    return dls


# In[ ]:


dls = get_dls_from_df(train)
dls.show_batch()


# ## Modeling, Training and Interpretation

# Use a pretrained ResNet-34 model. 
# 
# We first find a good learning rate using fastai2's learning rate finder. Then, training is done in 2 steps:
# * Train only our custom head(42 classes) for 1 epoch
# * Train the whole thing for 2 epoch
# 
# These 2 steps are automatically done by the `fine_tune` method fastai2 provided (and it does a bunch other cool optimizations too)
# 
# You can definitely train for more than these 3 epochs to get better accuracy. Using bigger models will help too.

# In[ ]:


learn = cnn_learner(dls, resnet34, metrics=accuracy, path="/kaggle/working")


# In[ ]:


slr = learn.lr_find()


# In[ ]:


print(f"Minimum/10: {slr.lr_min:.2e}, steepest point: {slr.lr_steep:.2e}")


# In[ ]:


learn.fine_tune(2, 5e-3)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Plot confusion matrix to see how well our model classify specific categories.

# In[ ]:


interp.plot_confusion_matrix(figsize=(15,15), dpi=60)


# In[ ]:


# get the most confused labels with at least 10 incorrect predictions
interp.most_confused(10)


# ## Predictions

# Create Test Dataloader from an existing dataloader (so that you can do TTA if you want to) .

# In[ ]:


test_images = test.filename.apply(lambda fn: test_path/fn)
test_dl = dls.test_dl(test_images)


# In[ ]:


preds = learn.get_preds(dl=test_dl, with_decoded=True)
preds


# * preds[0] -> probabilities
# * preds[1] -> ground truth (None in this case as we are training on the test set)
# * preds[2] -> decoded probabilities AKA our category/label predictions

# In[ ]:


# save raw predictions
torch.save(preds, "rawpreds")


# ## Submission

# In[ ]:


submission  = test[["filename"]]
submission["category"] = preds[2]


# We need to then zero-pad the submissions ('01' instead of '1').

# In[ ]:


# zero-pad the submissions
submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))


# In[ ]:


# preview
submission


# In[ ]:


# save the submissions as CSV
submission.to_csv("submission.csv")

