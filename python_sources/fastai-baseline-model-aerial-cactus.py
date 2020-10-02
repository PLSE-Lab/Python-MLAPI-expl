#!/usr/bin/env python
# coding: utf-8

# ## Introduction: Aerial Cactus Identification
# 
# Can you identify cacti on aerial imagery?
# 
# Researchers in Mexico are trying to build a system for the surveillance of protected areas. They want use advancements in drone technologies and object recognition to automate the surveillance.
# A first step and a proof of concept is the creation of a classifier capable of identifying cacti on aerial imagery.
# 
# This kernel shows how to use the [fast.ai library](https://docs.fast.ai/) to create a baseline model for the [Aerial Cactus Identification
# ](https://www.kaggle.com/c/aerial-cactus-identification) competition.
# 
# In this competition we're provided with a dataset that contains 17.5k of low-resolution (32px x 32px) images in the test set and 4k images in the training set.
# 
# I already did the setup, evaluation and prediction, so that you can get right into the training of the CNN.
# 
# To learn more about the fast.ai library check out the [documentation](https://docs.fast.ai) and the [fast.ai course 'Practical Deep Learning for Coders'](https://course.fast.ai/)

# ## Setup

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
from pathlib import Path
import os
import pandas as pd


# Check out the input-folder

# In[ ]:


print(os.listdir("../input"))


# Save path to train and test folders

# In[ ]:


train_dir = "../input/train/train/"
test_dir = "../input/test/test/"


# In[ ]:


print(os.listdir(train_dir)[:5])
print(os.listdir(test_dir)[:5])


# Take a look at train.csv

# In[ ]:


train_csv = pd.read_csv("../input/train.csv")
train_csv.head()


# When we take a look at the train folder and train_csv we can see, that train_csv contains the labels for the images inside the train folder.

# In[ ]:


tfms = get_transforms()


# In[ ]:


data = ImageDataBunch.from_df(
    df = train_csv,
    path = train_dir,
    test = "../../test",
    valid_pct = 0.2,
    bs = 16,
    size = 32,
    ds_tfms = tfms,
    num_workers = 0
).normalize(imagenet_stats)
print(data.classes)
data.show_batch(rows=2)


# ## Training

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")


# In[ ]:


learn.fit_one_cycle(1)


# ## Evaluation
# Create a ClassificationInterpretation object to generate confusion matrices and visualizations of the most incorrect images.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Plot the two images with the highest loss

# In[ ]:


interp.plot_multi_top_losses(2, figsize=(6, 6))


# A good way to summarize the performance of a classification algorithm is to create a confusion matrix. Confusion Matricies are used to understand which classes are most easily confused.

# In[ ]:


interp.plot_confusion_matrix()


# ## Prediction
# Let the neural network predict. It predicts a probability for every possible class. The class with the highest probability is taken as the predicted class

# In[ ]:


class_score, y = learn.get_preds(DatasetType.Test)
class_score = np.argmax(class_score, axis=1)


# ### Create a submission

# In[ ]:


submission  = pd.DataFrame({
    "id": os.listdir(test_dir),
    "has_cactus": class_score
})
submission.to_csv("submission.csv", index=False)
submission[:5]

