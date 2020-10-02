#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# # Create a path to the dataset

# In[ ]:


path = Path('../input/heroes/heroes/')
path.ls()


# # Create a databunch to feed the data into the model

# In[ ]:


data = ImageDataBunch.from_folder(
    path,
    train = '.',
    valid_pct = 0.1,
    ds_tfms=get_transforms(max_warp=0,flip_vert=True,do_flip=True),
    size = 128,
    bs=16
).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=8,figsize=(10,10))


# # Create the learner

# In[ ]:


learn = create_cnn(data,models.resnet50,metrics=accuracy,model_dir='/tmp/model/')
learn.lr_find()
learn.recorder.plot()


# # Train the model

# In[ ]:


learn.fit_one_cycle(5)


# # Plot the first stage losses of the training and validation

# In[ ]:


learn.recorder.plot_losses()


# # Save the first stage of the model

# In[ ]:


learn.save('overwatch-stage-1')


# # Unfreeze the model and train

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2)


# # Find the learning rate and plot it

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# # Train the model

# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-4))


# # Plot the losses of the training and validation

# In[ ]:


learn.recorder.plot_losses()


# # Plot the top losses from the training 

# In[ ]:


inter = ClassificationInterpretation.from_learner(learn)
inter.plot_top_losses(10,figsize=(20,20))


# # Plot confusion matrix

# In[ ]:


inter.plot_confusion_matrix(figsize=(10,10))

