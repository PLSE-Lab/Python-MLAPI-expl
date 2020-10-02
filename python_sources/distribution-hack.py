#!/usr/bin/env python
# coding: utf-8

# This kernel simply produces 9 output files, one for each target category. I submitted each of these to the competition to see how much of each target type exists in the test set distribution. Results:
# 
# - carpet 0.06
# - concrete 0.16
# - fine concrete 0.09
# - hard tiles 0.06
# - hard tiles large space 0.10
# - soft pvc 0.17
# - soft tiles 0.23
# - tiled 0.03
# - wood 0.06
# 
# Also posted a [discussion thread](https://www.kaggle.com/c/career-con-2019/discussion/85204) .
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ss = pd.read_csv('../input/sample_submission.csv')
ss["surface"].value_counts()


# In[ ]:


ss.to_csv('concrete.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "hard_tiles"
ss.to_csv('hard_tiles.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "carpet"
ss.to_csv('carpet.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "soft_tiles"
ss.to_csv('soft_tiles.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "hard_tiles_large_space"
ss.to_csv('hard_tiles_large_space.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "fine_concrete"
ss.to_csv('fine_concrete.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "tiled"
ss.to_csv('tiled.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "wood"
ss.to_csv('wood.csv', index=False)
ss.head(10)


# In[ ]:


ss['surface'] = "soft_pvc"
ss.to_csv('soft_pvc.csv', index=False)
ss.head(10)


# In[ ]:





# In[ ]:





# In[ ]:




