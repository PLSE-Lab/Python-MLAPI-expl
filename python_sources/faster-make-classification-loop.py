#!/usr/bin/env python
# coding: utf-8

# mhviraf's terrific ["Synthetic data for (next?) Instant Gratification"](https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification) kernel shows how to generate data very much like the data in this comp (if you are reading this, please upvote his kernel). At heart of that kernel is a loop that repeatedly calls `sklearn.datasets.make_classification`. Below is a  refactored version of his loop that runs about 10x quicker. It's probably the only useful thing I've done so far so I thought I'd share it in case someone finds it handy.

# In[ ]:


from sklearn.datasets import make_classification 
import numpy as np
import pandas as pd


# ### Original

# In[ ]:


def gen_data_0():
    # generate dataset 
    train, target = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)
    train = np.hstack((train, np.ones((len(train), 1))*0))

    for i in range(1, 512):
        X, y = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)
        X = np.hstack((X, np.ones((len(X), 1))*i))
        train = np.vstack((train, X))
        target = np.concatenate((target, y))
    return train, target


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train0, target0 = gen_data_0()')


# ### Refactored

# In[ ]:


def gen_data(N=512, M=255):
    train = np.zeros((N**2, M + 1,), dtype=np.float)
    target = np.zeros((N**2,), dtype=np.float)
    for i in range(N):
        X, y = make_classification(N, M, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)
        X = np.hstack([X, i * np.ones((N, 1,))])

        start, stop = i * N, (i + 1) * N
        train[start: stop] = X
        target[start: stop] = y
    return train, target


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train, target = gen_data()')


# In[ ]:


np.random.seed(2019)
train0, target0 = gen_data_0()

np.random.seed(2019)
train, target = gen_data()


# In[ ]:


np.allclose(train, train0) and np.allclose(target, target0)


# ^ the results are the same.

# In[ ]:




