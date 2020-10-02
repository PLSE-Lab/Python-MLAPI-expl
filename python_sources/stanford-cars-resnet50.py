#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
from PIL import Image


# # Data Preprocessing

# In[ ]:


img = Image.open('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/Acura Integra Type R 2001/00198.jpg')
width, height = img.size


# In[ ]:


width, height


# In[ ]:


path = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/'


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path,valid_pct=0.2, 
                                  ds_tfms=get_transforms(), 
                                  size=300, bs=64, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(10,12))


# # Training Model

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=[error_rate,accuracy], model_dir = '/kaggle/working/')


# As we know that 'fit_one_cycle' (based on Leslie Smith's 1-cycle policy) helps in getting
# better results than 'fit' method.

# In[ ]:


learn.fit_one_cycle(6)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-3, 1e-5))


# In[ ]:


learn.save('stage-2')


# # Interpretation

# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp.plot_top_losses(6, figsize=(30,26))


# In[ ]:


interp.most_confused(min_val=4)

