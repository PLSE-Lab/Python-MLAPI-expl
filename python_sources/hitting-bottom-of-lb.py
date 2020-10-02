#!/usr/bin/env python
# coding: utf-8

# # Simple EDA and submission

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport # for EDA


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')


# In[ ]:


train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile


# ** If you scroll through the report you are able to find the Correlation between various attributes, individual EDA of each variable, interactions and can even find missing values**
# 
# 

# In[ ]:


test_profile = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})
test_profile


# In[ ]:


sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# Just use Pandas profiling for everything related to EDA folks!
# 
# ## Fin.
