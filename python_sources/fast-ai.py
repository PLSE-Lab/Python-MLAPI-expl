#!/usr/bin/env python
# coding: utf-8

# # Using fast.ai on the MIMIC-CXR dataset
# 
# ### **Welcome to this kernel!** 
# 
# This exercise is adapated and inspired from the very excellent "Hello World Deep Learning" [kernel](https://www.kaggle.com/wfwiggins203/hello-world-deep-learning-siim) by Walter Wiggins (2018-2019). In this kernel, we will use fast.ai and the MIMIC-CXR database to see if we can detect chest tubes in AP chest x-rays. We will be using 6,000 images in our dataset: **3,000** with ET tubes and **3,000** without.
# 
# The central point of this exercise is to demonstrate that "deep learning" hardly requires tech expertise, and that a few lines of code can easily be recycled to look at positive and negative examples of just about anything.
# 
# >Moreover, cloud computing is so powerful now that we don't need to have our own supercomputer to run experiments. This experiment runs on cloud computing provided to us for free from Kaggle (and Google, by extension).
# 
# # How to use this kaggle kernel
# 
# To explore this model and data set, please <kbd>Fork</kbd> the notebook in the menu above (i.e. create a copy under your Kaggle profile). 
# 
# >When you get into the draft environment, please ensure that you see **"GPU on"** and **"Internet on"** under **Settings** so you can utilize the cloud GPU for faster model training.
# 
# In this Notebook editing environment, each block of text is referred to as a **cell**.  Cells containing formatted text are **Markdown** cells, as they use the *Markdown* formatting language. Similarly, cells containing code are **code** cells.
# 
# Clicking within a cell will allow you to edit the content of that cell (a.k.a. enter **edit mode**). You can also navigate between cells with the arrow keys. Note that the appearance of Markdown cells will change when you enter edit mode.
# 
# You can **run code cells** (and format Markdown cells) as you go along by clicking within the cell and then clicking the **blue button with *one* arrow** next to the cell or at the bottom of the window. You can also use the keyboard shortcut <kbd>SHIFT</kbd> + <kbd>ENTER</kbd> (press both keys at the same time).

# ## 1. Loading Python modules
# For this experiment, we're using the [**Python** programming language](https://www.python.org/) with the [**FastAI**](https://www.fast.ai/) *backend* for model computation.

# In[ ]:


# Environment: Python 3, preloaded packages found in: https://github.com/kaggle/docker-python

import numpy as np           # used for linear algebra
import pandas as pd          # used for data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *  # used for neural network development


# ## 2. View data
# 
# Next, we will look at our data set.

# In[ ]:


path = Path('/kaggle/input/mimic-cxr/present_absent/3k/')
classes = ['present','absent']
path.ls()


# We will resize our images from approximately 1k x 2k pixels to 256 x 256. This will make them easier to work with and effectively require less computation overall.

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=312, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ## 3. Training the model
# Now, we will **train the model**. This will take approximately 18 minutes per epoch (**8 hours for 25 epochs**).
# 
# We will be using the resnet50 model, a 

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=[accuracy, AUROC()])
# this line is necessary to save the model (usually read-only access)
learn.model_dir='/kaggle/working/'


# For fast.ai, we want to use this call to find the best learning rate for our model. We will comment this out to save computation/gpu time. 
# A previous experiment has the best learning rate as between 5e-3 and 5e-2.

# In[ ]:


# learn.lr_find(start_lr=1e-5, end_lr=5e-1)
# learn.recorder.plot()


# Looks like there is approximately 80% accuracy. Not great, but a start for our simple model.
# 
# **Please don't try to run this on your own unless you're cool with using all of your GPU time!**
# >If you reduce the number of epochs to a smaller number, you can run this on your own without destroying your GPU time.

# In[ ]:


learn.fit_one_cycle(80, max_lr=slice(5e-3,5e-2))
learn.recorder.plot_losses()
learn.save('stage-1')


# ## 4. Evaluating training results and model performance
# 
# Once training is finished, **evaluating the training curves** can help us **tune our hyperparameters** in subsequent experiments by telling us if we're **underfitting** or by showing us when the model begins to **overfit**.
# 
# Finally, we will demonstrate the model output on the test data set (in this case, one of each type of radiograph).
# 
# >At the end of your experiments, you want to evaluate network performance on an **independent, *held-out* test data set**, as the model will be *indirectly* exposed to the validation data set over successive experiments, potentially **biasing the model hyperparameters** toward overfitting to the validation data.

# In[ ]:


learn.load('stage-1');
interp = ClassificationInterpretation.from_learner(learn)


# This method below will show up heatmaps to make sure that the model is looking at the right thing to determine if there is or is not a chest tube.
# 
# >We can see that the model is often misclassifying images by looking at the wrong thing. In a subsequent kernel we can look into how we can fix this.

# In[ ]:


interp.plot_confusion_matrix()
interp.plot_top_losses(64, figsize=(24,24), heatmap=False)
interp.plot_top_losses(64, figsize=(24,24), heatmap=True, alpha=0.3)

