#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' git clone https://github.com/lopuhin/tpu-imagenet.git')
get_ipython().system(' cd tpu-imagenet && git show -s')


# In[ ]:


get_ipython().system(' cd tpu-imagenet && git pull')


# In[ ]:


get_ipython().system(' export PYTHONPATH=tpu-imagenet')


# In[ ]:


get_ipython().system(' ls /kaggle/input/')


# In[ ]:


from pathlib import Path
from kaggle_datasets import KaggleDatasets

gcs_paths = [KaggleDatasets().get_gcs_path(p.name)
             for p in Path('/kaggle/input/').iterdir()]
gcs_paths


# In[ ]:


get_ipython().system(" tpu-imagenet/train.py {' '.join(gcs_paths)} --epochs 3")

