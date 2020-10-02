#!/usr/bin/env python
# coding: utf-8

# Hi I'm new to this competition and this is my first time posting to kernel.  
# I found out that the **pandas-profiling** is very useful for beginners to start EDA.  
# I showed the result of pandas-profiling using test and train data which takes only a minute.  
# Check it out! I'm so sorry that this might not be useful for experts!  

# In[ ]:


import pandas as pd
import pandas_profiling as pdp
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
pdp.ProfileReport(train)


# In[ ]:


pdp.ProfileReport(test)


# In[ ]:





# In[ ]:




