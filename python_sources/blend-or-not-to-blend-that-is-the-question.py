#!/usr/bin/env python
# coding: utf-8

# # Weighted Average example

# Blending is the best way to explore diversity from models.
# 
# Taking that into account, why not to blend a LightGBM and Keras based models?
# 
# I grab output submissions files from that kernels:
# 
# - https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb
# - https://www.kaggle.com/todnewman/keras-neural-net-for-champs

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


sub1 = pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')
sub2 = pd.read_csv('../input/keras-neural-net-for-champs/workingsubmission-test.csv')
print( sub1['scalar_coupling_constant'].describe() )
print( sub2['scalar_coupling_constant'].describe() )


# In[ ]:


#Mean absolute difference
( sub1['scalar_coupling_constant'] - sub2['scalar_coupling_constant']).abs().mean()


# In[ ]:


# I used 0.6 weight for LGB just because it performed a little bit better in Public LB.
sub1['scalar_coupling_constant'] = 0.6*sub1['scalar_coupling_constant'] + 0.4*sub2['scalar_coupling_constant']
sub1.to_csv('weighted-avg-blend-lgb-keras-1.csv', index=False )
sub1['scalar_coupling_constant'].describe()

