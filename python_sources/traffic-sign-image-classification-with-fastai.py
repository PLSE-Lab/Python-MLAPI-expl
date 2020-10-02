#!/usr/bin/env python
# coding: utf-8

# # Traffic Sign Image Classification with FastAI

# In[ ]:


# General Data Science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning
from fastai.vision import ImageDataBunch, get_transforms, cnn_learner
from fastai.vision.data import imagenet_stats
from fastai.vision.models import resnet34
from fastai.vision.learner import ClassificationInterpretation
from fastai.metrics import error_rate

# Miscellaneous
import os
from pathlib import Path
from PIL import Image
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# As I'm going throught the excellent deep learning course taught by the team at FastAI (available [here](https://course.fast.ai/)), I will be applying what I've learned to build deep learning models for various applications using different data sources here on Kaggle. This notebook follows the general methodology of the lesson 1 notebook (available [here](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)). In this notebook, we will be developing a deep learning model to identify German traffic signs.
# 
# We will begin by loading in the provided training data into a FastAI 'ImageDataBunch' object. We will make use of the dataset in the provided training set folder, randomly extracting 10% of it to be used for a validation set, and setting the batch size to 64 and the image size to 224x224. These images will also be normalized using the FastAI 'imagenet_stats' object, and augmented using the 'get_transforms' FastAI object.

# In[ ]:


data_path = Path("/kaggle/input/gtsrb-german-traffic-sign/")
data = ImageDataBunch.from_folder(data_path/"train",
                                  ds_tfms=get_transforms(do_flip=False),
                                  size=224,
                                  bs=64,
                                  valid_pct=0.1,
                                  seed=0).normalize(imagenet_stats)
data


# Let's take a look at some of these images with their associated labels.

# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# Viewing the above sample images, we can see that these photos are really quite fuzzy. We can also see that the labels given to these images are simply numeric labels. Let's check to see how many different sign types we are working with.

# In[ ]:


data.c


# Let's see if we can apply transfer learning to this problem, using a resnet34 to classify these traffic signs. We will use a FastAI 'cnn_learner' to encapsulate this model.
# 
# **Note**: The 'Internet' setting on Kaggle will need to be turned on in order to download the resnet34 weights.

# In[ ]:


learn = cnn_learner(data, resnet34, metrics=error_rate, model_dir=Path("/kaggle/working/model"))
learn.model


# Let's now train the 'head' of this model (i.e. the final linear layers of the model) for 4 epochs using the '1cycle policy', keeping the pretrained weights in the convolutional layers of the resnet34 model constant or 'frozen'.
# 
# **Note**: The 'GPU' setting on Kaggle should be turned on to make use of a GPU for model training.

# In[ ]:


learn.fit_one_cycle(4)


# It seems after only a few epochs of training, transfer learning has proven to be fruitful, yielding quite a low error rate. To better understand the cases in which the model made incorrect classifications, let's view the cases where the model was most confident in the misclassification it made.

# In[ ]:


interpretation = ClassificationInterpretation.from_learner(learn)
interpretation.plot_top_losses(9)


# For perhaps clarity, each image above has 4 numbers associated with it. They are the predicted class, actual class, loss, and model's predicted probability for the actual class respectively. Let's also take a look at the most common errors made by the model, where we will have for each instance the actual class, predicted class, and number of times this error was made.

# In[ ]:


interpretation.most_confused(min_val=2)


# To better understand why the model perhaps made these errors, for a few of the above cases, let's plot a sample image of the predicted and actual class label.

# In[ ]:


def img_comparison(label_0, label_1):
    """
    Plot samples of two image classes side-by-side.
    
    Args:
        label_0: int, first class label of interest
        label_1: int, second class label of interest
    """
    
    fig, ax = plt.subplots(1, 2)
    image_0 = np.array(Image.open(f"/kaggle/input/gtsrb-german-traffic-sign/meta/{label_0}.png"))
    image_1 = np.array(Image.open(f"/kaggle/input/gtsrb-german-traffic-sign/meta/{label_1}.png"))
    ax[0].imshow(image_0)
    ax[1].imshow(image_1)
    fig.tight_layout()


# In[ ]:


img_comparison(2, 5)


# In[ ]:


img_comparison(7, 8)


# Viewing the above samples, we can see why the model might make these errors, especially taking into consideration the fuzziness of the images.
# 
# To finish up our work, let's now see if we can perhaps further improve the model's performance. We'll run a learning rate find operation for our learner.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Let's now unfreeze the model weights. Based on the above graph, in which the loss seems to increase for learning rates greater than 1e-4, we'll train this unfrozen model for 2 more epochs, again using the 1cycle policy, training the initial model layer with a learning rate of 1e-6 and the final model layer with a learning rate of 1e-4, using learning rates between those two values for all model layers in between.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))


# Viewing the above results, we can see that we have further improved model validation performance, and achieved quite a low error rate.
