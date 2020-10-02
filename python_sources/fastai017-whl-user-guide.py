#!/usr/bin/env python
# coding: utf-8

# # User Guide

# Some competitions require that the `internet` be turned `off` such as:
# 
# - [Prostate cancer diagnosis using the Gleason grading system](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
# 
# 
# This dataset allows you to easily load `fastai2` dependencies with the internet tab set to `off`

# # Loading fastai2

# This dataset provides the 3 files you need to load fastai2:
# - fastai2 version 0.0.17
# - fastcore version 0.1.18
# - fastprogress version 0.2.3 
# 
# ### Step 1: 
# In your kernel click on the Add data tab on the right top of the notebook.
# 
# ![data.PNG](attachment:data.PNG)

# ### Step 2:
# Under the Search Datasets bar, enter `fastai017` and fastai017.whl should come up as an option.  Click on the `Add` tab
# 
# ![fast017.PNG](attachment:fast017.PNG)

# ### Step 3
# Once the dataset has loaded you will find fastai017.whl under the input section.
# 
# ![input.PNG](attachment:input.PNG)

# You can now load all fastai2 dependencies with the internet off.

# You can load the dependancies like so:

# In[ ]:


get_ipython().system('pip install ../input/fastai017-whl/fastprogress-0.2.3-py3-none-any.whl')
get_ipython().system('pip install ../input/fastai017-whl/fastcore-0.1.18-py3-none-any.whl')
get_ipython().system('pip install ../input/fastai017-whl/fastai2-0.0.17-py3-none-any.whl')


# In[ ]:


from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *


# Check that all the dependancies loaded ok

# In[ ]:


BLUE = '\033[94m'
BOLD   = '\033[1m'
ITALIC = '\033[3m'
RESET  = '\033[0m'

import fastai2; print(BOLD + BLUE + "fastai2 Version: " + RESET + ITALIC + str(fastai2.__version__))
import fastprogress; print(BOLD + BLUE + "fastprogress Version: " + RESET + ITALIC + str(fastprogress.__version__))
import fastcore; print(BOLD + BLUE + "fastcore Version: " + RESET + ITALIC + str(fastcore.__version__))
import sys; print(BOLD + BLUE + "python Version: " + RESET + ITALIC + str(sys.version))
import torchvision; print(BOLD + BLUE + "torchvision: " + RESET + ITALIC + str(torchvision.__version__))
import torch; print(BOLD + BLUE + "torch version: " + RESET + ITALIC + str(torch.__version__))


# And voila you are all set to go!
