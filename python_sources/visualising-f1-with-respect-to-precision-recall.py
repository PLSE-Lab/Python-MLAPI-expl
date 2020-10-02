#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


precision_min, precision_max = 0, 1
recall_min, recall_max = 0, 1
xx, yy = np.meshgrid(np.arange(precision_min, precision_max, 0.1),
                     np.arange(recall_min, recall_max, 0.1))


# In[ ]:


Z = np.array([2.0*r*p/(r + p) for p, r in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)


# In[ ]:


plt.contourf(xx, yy, Z)
plt.ylabel('recall')
plt.xlabel('precision')
plt.title('F1')
plt.colorbar()
plt.show()


# In[ ]:




