#!/usr/bin/env python
# coding: utf-8

# # Blending some of the top Kernels

# More Blending inspired from Giba's Kernel : [https://www.kaggle.com/titericz/blend-or-not-to-blend-that-is-the-question].
# 
# other used models:=
# 1. https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# 2. https://www.kaggle.com/lpachuong/statstack
# 3. https://www.kaggle.com/vicensgaitan/giba-r-data-table-simplefeat-cyv-interaction
# 4. https://www.kaggle.com/todnewman/keras-neural-net-for-champs
# 
# This blend is to show blending diffrent models make a huge diffrences. 
# 
# 

# In[ ]:


#loading packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


#reading all submission files
sub1 = pd.read_csv('../input/blender/LGB_2019-07-11_-1.4378.csv')
sub2 = pd.read_csv('../input/blender/submission-2.csv')
sub3 = pd.read_csv('../input/blend3/stack_median.csv')
sub4 = pd.read_csv('../input/blender2/submission.csv')
sub5 = pd.read_csv('../input/blender2/submission-giba-1.csv')
sub6 = pd.read_csv('../input/blender/workingsubmission-test.csv')
print( sub1['scalar_coupling_constant'].describe() )
print( sub2['scalar_coupling_constant'].describe() )
print( sub3['scalar_coupling_constant'].describe() )
print( sub4['scalar_coupling_constant'].describe() )
print( sub5['scalar_coupling_constant'].describe() )
print( sub6['scalar_coupling_constant'].describe() )


# In[ ]:


# Random weights to each submission by trying and experimenting
sub1['scalar_coupling_constant'] = 0.30*sub1['scalar_coupling_constant'] + 0.20*sub2['scalar_coupling_constant'] + 0.30*sub3['scalar_coupling_constant'] + 0.10*sub4['scalar_coupling_constant'] + 0.10*sub5['scalar_coupling_constant'] 
sub1.to_csv('submission1236.csv', index=False )


# In[ ]:


#plotting histogram
sub1['scalar_coupling_constant'].plot('hist', bins=100)

