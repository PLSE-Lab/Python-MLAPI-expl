#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from pathlib import Path


# For this project I'm going to see how well fast.ai's vision library can identify which patient has pneumonia and which doesn't.
# 
# I will first build a simple model using just the basic ResNet-34. From there we will try to optimize it with the techniques learned from Fast.AI lessons.

# In[ ]:


from fastai import *
from fastai.vision import *

path=Path("../input/chest-xray-pneumonia/chest_xray/chest_xray/")


# In[ ]:


data = ImageDataBunch.from_folder(path,test='test',train='train',valid='val', size=224)
data.show_batch(rows=3)


# Lets see how many classes we are working with.

# In[ ]:


print(data.classes)
len(data.classes),data.c


# Lets create a basic CNN and see how well it performs.

# In[ ]:


from fastai.metrics import error_rate
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# As we can see with each epoch the training loss and error rate drops. But after epoch 1 we can see that validation loss adn error rate start to increase. This is due to the model overfitting to the training data (validation loss > training loss).
# 
# Lets explore what kind of images our model mislabelled.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

print(len(data.valid_ds)==len(losses)==len(idxs))

interp.plot_top_losses(9)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=2)


# Our model was only 82% accurate. Can we improve the accuracy even more? Possibly we can try unfreezing all the layers of resnet, try to find a better learning rate than the default value, and data augmentation.
# 
# ## Unfreezing
# The reason our model was being built so fast is because we only been training the last few layers that were added to the end of Resnet by the Fast.AI team. This allowed us to avoid training the weights of the earlier layers and focus training only on the last few layers. The problem with this method is we are trying to use Transfer Learning to a dataset that Resnet probably never seen before. We know from Visualizing and understand Convolutional models by Matthew D. Zeiler and Rob Fergus that Resnet learned features that were relevant to the dataset it was initially trained on. esnet probably has not learned the actual deep features of the images properly. Therefore we need to unfreeze the earlier layers to force Resnet to learn more accurate features.
# 
# Unfreezing, fine-tuning, and learning rates

# In[ ]:


learn.unfreeze()


# ## Data Augmentation
# 
# Data augmentation is basically modifying the data in certain ways that the underlying features still present but the CNN is still presented with a picture that is different from the original. Fast.AI allows many different transformation and we will apply all of them except invertion. I won't invert the image since x-ray pictures are all ways taken with the patient facing the doctor.

# In[ ]:


from fastai.vision import get_transforms

transforms=get_transforms(max_rotate=45, max_zoom=1.5, max_lighting=0.5, max_warp=0.3)
data = ImageDataBunch.from_folder(path,test='test',train='train', ds_tfms=transforms, valid='val', size=224)


# ## Learning Rate
# 
# Learning rate (LR) is the hyperparameter that determines how much a model should change. Choosing the right LR is crucial to having a model to converge on the weights that reduce error the most. Choosing a LR too small can take too long for the model to converge to the minimum. Choosing a LR to large can cause the model to converge to a subpotimal minimum or never converge at all.
# 
# We can use learn.lr_find() and learn.recorder.plot() to pick a learning rate thats ideal

# In[ ]:


learn.lr_find()
learn.recorder.plot()
plt.title("Loss Vs Learning Rate")
plt.show()


# We can see that from 1e-06 to 1e-02 the learning rate isable to sufficently reduce loss. However past 1e-02 the loss dramatically increases. Therefore we should choose a LR before 1e-02. Jeremy (the Fast.AI lecturer) recommends to pick a LR one fold before the loss sky-rockets. Therefore we are going to pick a LR <= 1e-03. We can also go with 1e-06 as the LR but that will take a long time for training. What Fast.AI allows us to do is pass a range of LR to the training sessions. Since we know that LR is fine up to 1e-02 we can just train the earlier layers with a large LR and then start fine turning with a smaller LR.   

# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-3))


# We have reduced our error rate but its still not as close as some of the top kernels in this Dataset. Other things we can do to reduce our error rate is to use a bigger CNN like ResNet-50, use a different optimizer, train for more epochs or try to implement more ResBlocks.
