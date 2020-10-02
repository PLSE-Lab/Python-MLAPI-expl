#!/usr/bin/env python
# coding: utf-8

# # Import The Neccesary Libraries

# In[ ]:


from fastai import *
from fastai.vision import * 


# # Load in the data and display some samples

# In[ ]:


path = '../input/v2-plant-seedlings-dataset'
data = ImageDataBunch.from_folder(path,
                                 ds_tfms=get_transforms(do_flip=False),
                                 size = 224,
                                 bs=28,
                                 valid_pct = 0.2).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(7,6))


# # Here we use the resnet34 model.

# In[ ]:


learner = cnn_learner(data,models.resnet34,metrics=accuracy,model_dir='/tmp/model/')


# # Train for three cycles and analyze training results 

# In[ ]:


learner.fit_one_cycle(3)


# # Save the first training stage and then unfreeze the model

# In[ ]:


learner.save('/kaggle/working/stage-1')
learner.unfreeze()


# # Use the learning rate finder 

# In[ ]:


learner.lr_find()
learner.recorder.plot()


# # Train for 5 cycles

# In[ ]:


learner.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


# # Plot a confusion matrix along with some predictions

# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(5,5), dpi=120)
interp.plot_top_losses(9, figsize=(15,11))


# # Other Links Where You Can Find Me
# GitHub: https://github.com/Terrance-Whitehurst
# 
# Kaggle: https://www.kaggle.com/twhitehurst3
# 
# LinkedIn: https://www.linkedin.com/in/terrance-whitehurst-242423173/
# 
# Website: https://www.terrancewhitehurst.com/
# 
# Blog: https://medium.com/@TerranceWhitehurst
# 
# Youtube:https://www.youtube.com/channel/UCwt6d06n0cbD5l-eoXBTEcA?view_as=subscriber
# 
