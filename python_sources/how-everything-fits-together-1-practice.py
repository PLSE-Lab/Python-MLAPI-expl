#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


ls /kaggle/input/


# In[ ]:


ls /kaggle/working/


# In[ ]:


get_ipython().system('pip install -U kaggle')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'chmod 600 /tmp/.kaggle/kaggle.json\nexport KAGGLE_USERNAME="zhannabitiukova"\nexport KAGGLE_KEY="API_KEY"\nif [ ! -f /kaggle/input/learn-together/train.csv ]; then \n    mkdir -p learn-together/\n    cd learn-together\n    kaggle competiotions download -c learn-together\n    unzip -o \'*.csv.zip\'\n    cd -\nfi')


# In[ ]:


get_ipython().system('realpath learn-together/')


# In[ ]:


import os
if os.path.exists("/kaggle/input/learn-together"):
    DATA_PATH = "/kaggle/input/learn-together"
else:
    DATA_PATH = "/kaggle/working/learn-together"
DATA_PATH = "/kaggle/input/learn-together"    


# In[ ]:


print(DATA_PATH)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('uname -a')


# In[ ]:


get_ipython().system('free -h')


# In[ ]:


get_ipython().system('nproc --all')


# In[ ]:


get_ipython().system('pip list')


# In[ ]:




