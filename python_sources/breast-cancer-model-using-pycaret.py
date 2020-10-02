#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data=data.drop(['Unnamed: 32'],axis=1)


# In[ ]:


from pycaret.classification import*


# In[ ]:


clf=setup(data,target='diagnosis')


# In[ ]:


compare_models()


# In[ ]:


model=create_model('et')


# In[ ]:


tuned_model=tune_model('et')


# **Model Accuracy=99.26%**

# In[ ]:


plot_model(tuned_model, plot = 'confusion_matrix')


# In[ ]:


plot_model(tuned_model, plot = 'boundary')


# In[ ]:


plot_model(tuned_model, plot = 'feature')


# In[ ]:


evaluate_model(tuned_model)

