#!/usr/bin/env python
# coding: utf-8

# # Superheroes NLP Dataset
# 
# Starter pack.
# 

# #### Import

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import seaborn as sns


# #### Settings

# In[ ]:


pd.set_option('display.max_columns', None)
sns.set(color_codes=True)

# Inline print matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# Retina display
get_ipython().run_line_magic('config', 'InlineBackend.figure_format="retina"')

# Figure size
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [15, 3]


# ##### Load the dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/superheroes_nlp_dataset.csv")
df.head(2)


# #### List all columns

# In[ ]:


df.columns


# #### List all "superpowers" columns

# In[ ]:


superpowers_cols = df.columns[df.columns.str.startswith("has_")]
superpowers_cols[:10]


# #### Most common "superpowers" columns

# In[ ]:


df[superpowers_cols].sum().sort_values(ascending=False)[:5]


# #### Gender feature

# In[ ]:


title = "Distribution of genders:"
df['gender'].value_counts().plot.bar();


# #### Show superheroes

# In[ ]:


hulk = df.query("name == 'Hulk'")
hulk_img = "https://www.superherodb.com" + hulk['img'].values[0]

import requests
import IPython.display as Disp
Disp.Image(requests.get(hulk_img).content)

