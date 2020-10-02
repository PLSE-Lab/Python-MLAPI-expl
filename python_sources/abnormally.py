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
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('git clone https://github.com/StevenLiuWen/ano_pred_cvpr2018.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'ano_pred_cvpr2018/Codes/checkpoints')


# In[ ]:


get_ipython().system('git clone https://github.com/deathvn/pretrains')
get_ipython().run_line_magic('cd', 'pretrains')


# In[ ]:


get_ipython().system('wget http://download943.mediafire.com/72eak7xz3mig/gfpe28rfs4tptm1/flownet-SD.ckpt-0.data-00000-of-00001')


# In[ ]:


get_ipython().system('wget http://download1525.mediafire.com/r60h6iawvwyg/feur15guonc09ul/flownet-SD.ckpt-0.meta')


# In[ ]:


get_ipython().system('wget http://download855.mediafire.com/7s1mgqdicnrg/l241oo8msf1lg6o/ped2.data-00000-of-00001')


# In[ ]:


get_ipython().run_line_magic('cd', '../../../Data')
get_ipython().system('git clone https://github.com/deathvn/ped2')


# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


get_ipython().system('mkdir Codes/frames')
get_ipython().system('mkdir npy')


# In[ ]:


get_ipython().run_line_magic('cd', 'Codes')


# In[ ]:


get_ipython().system('python inference.py --dataset  ped2                        --test_folder  ../Data/ped2/testing/frames                          --gpu  0                        --snapshot_dir    checkpoints/pretrains/ped2                        --evaluate compute_auc')

