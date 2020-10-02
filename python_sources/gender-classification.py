#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load libraries
from fastai import *
from fastai.vision import *
import pandas as pd


# ### Load data
# If you download data from internet : Remember to turn on the Internet settings

# In[ ]:


size = 96 # ssize of input images
bs = 32 # batch size
tfms = get_transforms(do_flip=False,flip_vert=True)


# In[ ]:


path = Path('../input/gender/gender')


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='Train',test='Testing',valid='Validation',
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


path.ls()


# ### Create your learner

# In[ ]:


model = models.resnet18


# In[ ]:


data.path = '/tmp/.torch/models'


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.save("stage-1")


# In[ ]:


lr = 2e-2


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.fit_one_cycle(4,slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr /100
learn.fit_one_cycle(4,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save("stage-2")


# In[ ]:


size = 128


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='Train',test='Testing',valid='Validation',
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


lr = lr /100
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-3')


# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


from fastai.vision import Image,pil2tensor
from PIL import Image
import cv2

def array2tensor(x):
    """ Return an tensor image from cv2 array """
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return Image(pil2tensor(x,np.float32).div_(255))


# In[ ]:


get_ipython().system(' wget http://66.media.tumblr.com/f740c7cdd5f87b93005343d42bc11e4c/tumblr_nq5bz8wSdH1uyaaeio2_1280.jpg')


# In[ ]:


get_ipython().system(' ls')


# In[ ]:


img = cv2.imread('tumblr_nq5bz8wSdH1uyaaeio2_1280.jpg')
img = array2tensor(img)

learn.predict(img)

