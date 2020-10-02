#!/usr/bin/env python
# coding: utf-8

# # Blending some of the top Kernels

# More Blending inspired from Giba's Kernel: [https://www.kaggle.com/titericz/blend-or-not-to-blend-that-is-the-question].
# 
# 
# I grabbed output submissions files from following public kernels:
# 
# - https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb
# - https://www.kaggle.com/todnewman/keras-neural-net-for-champs
# - https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# - https://www.kaggle.com/vicensgaitan/giba-r-data-table-simplefeat-cyv-interaction

# In[ ]:


#loading packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
print(os.listdir("../input/no-memory-reduction-workflow-for-each-type-lb-1-28"))


# In[ ]:


#reading all submission files
sub1 = pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')
sub2 = pd.read_csv('../input/keras-neural-net-for-champs/workingsubmission-test.csv')
sub3 = pd.read_csv('../input/no-memory-reduction-workflow-for-each-type-lb-1-28/LGB_2019-07-18_-1.2243.csv')
sub4 = pd.read_csv('../input/giba-r-data-table-simplefeat-cyv-interaction/submission-2.csv')
print( sub1['scalar_coupling_constant'].describe() )
print( sub2['scalar_coupling_constant'].describe() )
print( sub3['scalar_coupling_constant'].describe() )
print( sub4['scalar_coupling_constant'].describe() )


# In[ ]:


# Random weights to each submission by trying and experimenting
sub1['scalar_coupling_constant'] = 0.25*sub1['scalar_coupling_constant'] + 0.2*sub2['scalar_coupling_constant'] + 0.3*sub3['scalar_coupling_constant'] + 0.25*sub4['scalar_coupling_constant']
sub1.to_csv('submission.csv', index=False )


# In[ ]:


#plotting histogram
sub1['scalar_coupling_constant'].plot('hist', bins=100)

