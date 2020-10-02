#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.externals import joblib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Uncoment for local training - it takes 2.5 days on a 16 core CPU with 128 GB of RAM

#train_values = np.load('../input/train_values.npy')
#umap = UMAP()
#train_values_2D = umap.fit_transform(train_values)
#filename = 'umap.sav'
#joblib.dump(umap, filename)


# In[ ]:


train_2D = pd.read_csv("../input/bengali-umap-2d-embedding/train_2D.csv").values
train = pd.read_csv('../input/bengaliai-cv19/train.csv')


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = train['grapheme_root'].values, s = 10.0)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = train['vowel_diacritic'].values, s = 10.0)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = train['consonant_diacritic'].values, s = 10.0)


# In[ ]:




