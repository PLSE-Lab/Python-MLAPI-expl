#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


y = train['label'].values
train = train[test.columns].values
test = test[test.columns].values


# In[ ]:


train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'umap = UMAP()\ntrain_test_2D = umap.fit_transform(train_test)')


# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]


# In[ ]:


np.save('train_2D', train_2D)
np.save('test_2D', test_2D)


# In[ ]:


plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)


# In[ ]:




