#!/usr/bin/env python
# coding: utf-8

# This notebook compares 3 ways to compute QWK metric.

# In[ ]:


from numba import jit 
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ## @afajohn Method
# https://www.kaggle.com/afajohn/quadratic-weighted-kappa-with-numpy-flavor

# In[ ]:


def quadKappa(act,pred,n=4,hist_range=(0,3)):
    
    O = confusion_matrix(act,pred)
    O = np.divide(O,np.sum(O))
    
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E))
    
    num = np.sum(np.multiply(W,O))
    den = np.sum(np.multiply(W,E))
        
    return 1-np.divide(num,den)


# ## CPMP Method
# https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-657027

# In[ ]:


@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


# ## SKLEARN method
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

# ## Timing

# Let's test on 1M rows (the training data is 11 times larger)

# In[ ]:


size = 1000000
a = np.random.randint(0, 4, size)
p = np.random.randint(0, 4, size)
a.size, p.size


# @afajohn Method:

# In[ ]:


get_ipython().run_line_magic('timeit', 'quadKappa(a,p)')


# It takes about 1.5 second.
# 
# SKLEARN method:

# In[ ]:


get_ipython().run_line_magic('timeit', 'cohen_kappa_score(a, p, weights="quadratic")')


# It also takes about 1.5 second.
# 
# CPMP method.  We run it once to compile it. 

# In[ ]:


get_ipython().run_line_magic('timeit', 'qwk3(a,p)')


# It takes about 5 ms.  It is about 300 times faster than the other two methods.
