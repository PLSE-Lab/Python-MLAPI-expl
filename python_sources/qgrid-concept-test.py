#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is a test notebook to check if Qgrid works.

# In[ ]:


# !conda install -y nodejs
# !jupyter labextension install qgrid


# In[ ]:


# !jupyter nbextension enable --py --sys-prefix widgetsnbextension
# !jupyter nbextension enable --py --sys-prefix qgrid


# In[ ]:


import qgrid
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
qgrid_widget = qgrid.show_grid(df, show_toolbar=True)
qgrid_widget

