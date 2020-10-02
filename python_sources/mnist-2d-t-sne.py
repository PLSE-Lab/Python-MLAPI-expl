#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


train = train[test.columns].values
test = test[test.columns].values


# In[ ]:


train_test = np.vstack([train, test])
train_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(n_components=2)\ntrain_test_2D = tsne.fit_transform(train_test)')


# In[ ]:


train_2D = train_test_2D[:train.shape[0]]
test_2D = train_test_2D[train.shape[0]:]


# In[ ]:


np.save('train_2D', train_2D)
np.save('test_2D', test_2D)

