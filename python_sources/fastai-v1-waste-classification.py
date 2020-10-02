#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# # Create a path to the dataset

# In[ ]:


path = Path('../input/dataset/DATASET')
path.ls()


# # Load the data in using the ImageDataBunch

# In[ ]:


data = ImageDataBunch.from_folder(
    path,
    train = "TRAIN",
    valid = "TEST",
    ds_tfms=get_transforms(do_flip=False),
    size = 128,
    bs=32,
    valid_pct=0.2,
    num_workers=0
).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=10,figsize=(10,10))


# # Create the learner

# In[ ]:


learn = create_cnn(data, models.resnet50,metrics=accuracy,model_dir='/tmp/model/')
learn.lr_find()
learn.recorder.plot()


# # Train the model

# In[ ]:


learn.fit_one_cycle(5)


# # Plot the losses of the training and validation

# In[ ]:


learn.recorder.plot_losses()


# # Plot the top losses of your model

# In[ ]:


inter = ClassificationInterpretation.from_learner(learn)
inter.plot_top_losses(9,figsize=(20,20))


# # Plot a confusion matrix

# In[ ]:


inter.plot_confusion_matrix(figsize=(10,10))


# # Unfreeze model and train

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2)


# # Find and plot the learning rate

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# # Train the model

# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


# # Plot the top losses of the model

# In[ ]:


learn.recorder.plot_losses()


# # Plot the top losses

# In[ ]:


inter = ClassificationInterpretation.from_learner(learn)
inter.plot_top_losses(9,figsize=(20,20))


# # Plot confusion matrix

# In[ ]:


inter.plot_confusion_matrix(figsize=(10,10),dpi=75)
learn.save('waste-clf-fastai-V1')


# In[ ]:




