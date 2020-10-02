#!/usr/bin/env python
# coding: utf-8

# ## Introduction: Plant Seedlings Classification
# 
# Can you distinguish weeds from crop seedling?
# 
# If you are able to do this and thus remove all weeds from your field, you could achieve much better crop yields.
# 
# This kernel shows you how to use the [fast.ai library](https://docs.fast.ai/) to create a baseline model for the [Plant Seedlings Classification
# ](https://www.kaggle.com/c/plant-seedlings-classification) competition.
# 
# In this competition we are provided with 4750 labeled images of plant seedling in different stages of their growth. Our goal is to create a classifier capable of identifying the species of a crop given an image of it.
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


# Check out the input-folder

# In[ ]:


print(os.listdir("../input"))


# Save the train and test directories

# In[ ]:


train_dir = '../input/train/'
test_dir = '../input/test/'


# Check out train- and test-folder

# In[ ]:


print(os.listdir(train_dir)[:5])
print(os.listdir(test_dir)[:5])


# Load in the data

# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_folder(
    path = train_dir,
    test="../test",
    valid_pct = 0.2,
    bs = 16,
    size = 336,
    ds_tfms = tfms,
    num_workers = 0
).normalize(imagenet_stats)
data
print(data.classes)
data.show_batch()


# Create CNN

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")


# ## Training
# Fit one epoch

# In[ ]:


learn.fit_one_cycle(1)


# ## Evaluation
# Create a ClassificationInterpretation object to generate confusion matrices and visualizations of the most incorrect images.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Plot the 3 images with the highest loss

# In[ ]:


interp.plot_multi_top_losses(3, figsize=(6,6))


# A good way to summarize the performance of a classification algorithm is to create a confusion matrix. Confusion Matricies are used to understand which classes are most easily confused.
# 
# On the bottom you can see the classes the neural network predicted and on the left side you see the correct classes. So for every class you see which classes were predicted how often. 
# The dark line from the left top corner to the right bottom corner shows which 

# In[ ]:


interp.plot_confusion_matrix()


# Plot the highest values from the confusion matrix

# In[ ]:


interp.most_confused(min_val=5)


# ## Prediction
# Let the neural network predict. It predicts a probability for every possible class. The class with the highest probability is taken as the predicted class

# In[ ]:


class_score, y = learn.get_preds(DatasetType.Test)
class_score = np.argmax(class_score, axis=1)


# The predicted class that the neural network outputs is a numerical value. The desired form of the submission however is not a numerical value but the actual name of the class

# In[ ]:


predicted_classes = [data.classes[i] for i in class_score]
predicted_classes[:10]


# Create a submission

# In[ ]:


submission  = pd.DataFrame({
    "file": os.listdir(test_dir),
    "species": predicted_classes
})
submission.to_csv("submission.csv", index=False)
submission[:10]

