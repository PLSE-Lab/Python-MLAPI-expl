#!/usr/bin/env python
# coding: utf-8

# I implemented continuous approximation of quadratic weighted cohen's kappa for regressor on Keras.  
# The below function can be used as regressor loss function (but for loss, shoud change 1-k to k for smaller the better).  
# The performance of this loss is same level as binary_crossentropy in this competiton.  
# 
# I don't know the reason, there seems some local optima around 1.0, then when you use this, recommend to use other loss for the warming up (1 or 2 epochs). 

# In[ ]:


import keras.backend as K
# For scoring coding on continuous manner.
def QWKloss_score(y_true, y_pred):
    N = K.cast(K.shape(y_true)[0], 'float32')
    
    WC = (y_pred - y_true)**2 / N
    WE = (y_pred - K.transpose(y_true))**2 / (N**2)
    
    k = K.sum(WC) / K.sum(WE)
    
    return 1-k


# In[ ]:


import numpy as np
from sklearn.metrics import cohen_kappa_score

# check ~ comparing sklearn
for _ in range(10):
    N = 1000
    y_true = np.random.randint(5, size=(N))
    y_pred = np.random.randint(5, size=(N))  # cast to int is necessary for sklearn. but youcan use float in Keras use.

    skl = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    s = QWKloss_score(K.variable(y_true.reshape(-1, 1)), K.variable(y_pred.reshape(-1, 1)))
    org = K.get_value(s)
    
    print(skl, org)

