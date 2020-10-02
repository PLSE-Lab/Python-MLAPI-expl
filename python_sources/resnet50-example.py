#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


# ## Copy weigths to .keras/models

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet* ~/.keras/models/')
get_ipython().system('cp ../input/imagenet_class_index.json ~/.keras/models/')


# In[ ]:


get_ipython().system('ls ~/.keras/models')


# ## Read example image

# In[ ]:


fig, ax = plt.subplots(1, figsize=(12, 10))
img = image.load_img('../input/Kuszma.JPG')
img = image.img_to_array(img)
ax.imshow(img / 255.) 
ax.axis('off')
plt.show()


# ## Use model with pretrained weights

# In[ ]:


resnet = ResNet50(weights='imagenet')


# In[ ]:


img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
preds = resnet.predict(x)
decode_predictions(preds, top=5)


# She is not really a pedigree dog so there are no right or wrong answers here :)
# German sepherd is our best guess as well.
