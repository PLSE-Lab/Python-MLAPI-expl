#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp ../input/segmentation-models-zip-003/efficientnet_pytorch-0.4.0.xyz /tmp/pip/cache/efficientnet_pytorch-0.4.0.tar.gz')
get_ipython().system('cp ../input/segmentation-models-zip-003/pretrainedmodels-0.7.4.xyz /tmp/pip/cache/pretrainedmodels-0.7.4.tar.gz')
get_ipython().system('cp ../input/segmentation-models-zip-003/segmentation_models_pytorch-0.0.3.xyz /tmp/pip/cache/segmentation_models_pytorch-0.0.3.tar.gz')


# In[ ]:


get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ efficientnet-pytorch')
get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ segmentation-models-pytorch')


# In[ ]:


import segmentation_models_pytorch as smp


# In[ ]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
get_ipython().system('cp ../input/efficientnet-pytorch-b0-b7/efficientnet-b0-355c32eb.pth /tmp/.cache/torch/checkpoints/')


# In[ ]:


model = smp.Unet("efficientnet-b0", encoder_weights="imagenet", classes=4, activation="sigmoid")


# In[ ]:


model


# In[ ]:




