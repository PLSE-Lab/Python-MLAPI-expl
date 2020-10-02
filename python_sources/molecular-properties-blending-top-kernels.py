#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
os.listdir('../input')


# In[ ]:


one = pd.read_csv('../input/champs-blending-tutorial/1.csv')  ## distance
two = pd.read_csv('../input/champs-blending-tutorial/2.csv')  ## LGB + features
three = pd.read_csv('../input/champs-blending-tutorial/3.csv') ## MPNN
four = pd.read_csv('../input/otherkernelsadded/submission-2.csv')  ## GIBA
five = pd.read_csv('../input/otherkernelsadded/submission-giba-1.csv')  ## GIBA+
six = pd.read_csv('../input/otherkernelsadded/workingsubmission-test.csv')   ##
seven = pd.read_csv('../input/otherkernelsadded/LGB_2019-07-18_-1.2243.csv') ## LGB

submission = pd.DataFrame()
submission['id'] = one.id
submission['scalar_coupling_constant'] = (0.40*one.scalar_coupling_constant) +                                         (0.29*two.scalar_coupling_constant) +                                         (0.19*three.scalar_coupling_constant) +                                         (0.04*five.scalar_coupling_constant) +                                         (0.04*six.scalar_coupling_constant) +                                         (0.04*seven.scalar_coupling_constant)

submission.to_csv('my_blend_1.csv', index=False)

