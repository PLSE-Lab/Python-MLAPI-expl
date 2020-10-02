#!/usr/bin/env python
# coding: utf-8

# ## Introduction: Dog Breed Identification
# 
# How well can you distinguish different dog breeds?
# 
# The task of this competition is to do fine-grained image classification on images of different dog breeds.
# 
# This kernel shows you how to use the [fast.ai library](https://docs.fast.ai/) to create a baseline model for the [Dog Breed Identification
# ](https://www.kaggle.com/c/dog-breed-identification) competition.
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
import os
import pandas as pd
from pathlib import Path


# Check out the inout folder

# In[ ]:


input_dir = Path("../input")
print(os.listdir(input_dir))


# Save the train and test directories

# In[ ]:


train_dir = input_dir/'train'
test_dir = input_dir/'test'


# Check out train- and test-folder

# In[ ]:


print(os.listdir(train_dir)[:5])
print(os.listdir(test_dir)[:5])


# Take a look at labels.csv

# In[ ]:


labels_path = "../input/labels.csv"
labels_csv = pd.read_csv(labels_path)
labels_csv.head()


# Load in the data

# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_csv(
    path = "../input",
    folder = "train",
    suffix = ".jpg",
    test = "test/test",
    bs = 16,
    size = 224,
    ds_tfms = tfms,
    num_workers = 0
).normalize(imagenet_stats)
print(data.classes[:10])
data.show_batch(rows=2)


# ## Training

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")


# In[ ]:


learn.fit_one_cycle(1)


# ## Evaluation
# Create a ClassificationInterpretation object to generate confusion matrices and visualizations of the most incorrect images

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Plot the four images with the highest loss

# In[ ]:


interp.plot_multi_top_losses(4, figsize=(10, 10))


# Typically, you would now create a confusion matrix, but there are 120 classes in this task, so the confusion matrix would be a mess. Fast.ai has this neat function that only shows the highest values of the confusion matrix:
# 

# In[ ]:


interp.most_confused(min_val=5)


# ## Prediction
# Let the neural network predict. It predicts a probability for every possible class. The class with the highest probability is taken as the predicted class

# In[ ]:


class_score, y = learn.get_preds(DatasetType.Test)


# ### Create a submission

# In[ ]:


# let's first check the sample submission file to understand the format of the submission file
sample_submission =  pd.read_csv(input_dir/"sample_submission.csv")
display(sample_submission.head(3))


# In[ ]:


classes_series = pd.Series(os.listdir(test_dir))
classes_series = classes_series.str[:-4]
classes_df = pd.DataFrame({'id':classes_series})
predictions_df = pd.DataFrame(class_score.numpy(), columns=data.classes)
submission = pd.concat([classes_df, predictions_df], axis=1)


# In[ ]:


submission.to_csv("submission.csv", index=False)
submission[:5]

