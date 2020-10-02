#!/usr/bin/env python
# coding: utf-8

# # White Blood Cell Classification with ResNet34

# # The Dataset
# 
# This dataset contains 12,500 augmented images of various types of white blood cells (JPEG) with accompanying cell type labels (CSV). There are approximately 3,000 images for each of 4 different cell types grouped into 4 different folders (according to cell type). The cell types are Eosinophil, Lymphocyte, Monocyte, and Neutrophil.
# 
# ## Importance:
# 
# White blood cell (WBCs) counting is an important indicator of health and is important for many diagnostic tests. Currently, doctors utilize expensive automated counters like flow cytometers, or manually count blood cells on a microscope slide. Therefore, providing an automated way to detect and count WBCs would be advantageous. Detecting the WBCs is the first step for achieving this goal.

# # Dataset loading

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import os
import pandas as pd


# In[ ]:


bs = 32 # batch size
sz=224 # image size


# In[ ]:


base_image = Path('../input/dataset2-master/dataset2-master/images/')
base_image/'TRAIN'
data = ImageDataBunch.from_folder(path=base_image,train='TRAIN',valid='TEST',size=sz,bs=bs,num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch()


# In[ ]:


print(data.classes)
len(data.classes),data.c


# # Training of model
# 
# We will use transfer learning to train a ResNet34 model pretrained on the ImageNet dataset on our dataset instead.

# In[ ]:


model_path=Path('/tmp/models/')
learn = create_cnn(data, models.resnet34, metrics=error_rate,model_dir=model_path)


# We will use the amazing learning rate finder to determine the optimal learning rate for training.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,max_lr=1e-2)


# In[ ]:


learn.save('stage-1-224')


# The previous training only trained the last layer of the model. We can try to unfreeze and train earlier layers of the model.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('stage-2-224')


# Interestingly, the learning rate finder does not work (it stops prematurely as loss wasn't decreasing when adjusting the learning rate) and the training these earlier layers does not significantly improve the model. I will load the previously trained model and analyze how well it did.

# In[ ]:


learn.load('stage-1-224')


# # Results
# We will now analyze the results of the model.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# Overall, the model is performing quite well! There does seem to be some confusion between neutrophils and eosinophils. In addition, for some reason the model predicts neutrophil for monocyte but not the vice versa. I have tried progressive resizing but that did not improve training. 
# 
# *If you have any input regarding how to improve model, please leave a comment below!*
# 
# 
# # Future work
# * Note again that the dataset (in the folder `dataset2-master`) used here was already augmented, along with the augmentations done by `fastai`. It would be interesting to see what the results would be just using the 366 files provided by the original dataset.
# 
# * The original dataset provides annotations for bounding box detection, so we could try in future work to predict the bounding boxes. 
# 
# # Acknowledgements
# 
# * The techniques are inspired by fastai lesson 1.
# * Original dataset from https://github.com/Shenggan/BCCD_Dataset
# * Thanks to Paul Mooney for the augmented dataset
# 
# **If you enjoyed this kernel, please leave an upvote!**
