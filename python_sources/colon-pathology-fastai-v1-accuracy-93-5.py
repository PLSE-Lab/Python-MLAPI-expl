#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = get_ipython().getoutput("ldconfig -p|grep cudart.so|sed -e 's/.*\\.\\([0-9]*\\)\\.\\([0-9]*\\)$/cu\\1\\2/'")
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

get_ipython().system('pip install torch_nightly -f https://download.pytorch.org/whl/nightly/{accelerator}/torch_nightly.html')


import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


# In[ ]:


get_ipython().system('pip3 install git+https://github.com/fastai/fastai.git')


# In[ ]:


import fastai
print(fastai.__version__)
from fastai import *
from fastai.vision import *


# In[ ]:


PATH = "../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"

model_dir= "../input/"
sz=150
bs =16


# In[ ]:


tfms=get_transforms()


# In[ ]:


data = (ImageItemList.from_folder(PATH)
        .random_split_by_pct()
        .label_from_folder()
        .transform(tfms, size=150)
        .databunch(num_workers=0))


# In[ ]:


#learn.model_dir = model_dir


# In[ ]:


import pathlib


# In[ ]:


model_dir = pathlib.Path('.')


# In[ ]:


model_dir


# In[ ]:



bs =8


# In[ ]:


learn = create_cnn(data, models.resnet18, metrics=accuracy,model_dir = model_dir)


# In[ ]:





# In[ ]:



learn.fit(2)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


bs = 32


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'learn.unfreeze()\nlearn.fit_one_cycle(3, slice(1e-5,3e-4), pct_start=0.05)')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(4) 


# In[ ]:




