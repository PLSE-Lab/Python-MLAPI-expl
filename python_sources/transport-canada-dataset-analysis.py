#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


print('Loading datasets...')
file_path = '../input/canada-aircraft-accidents/'
CADORS_Aircraft_Event_Information = pd.read_csv(file_path + 'CADORS_Aircraft_Event_Information.csv', sep=',')
CADORS_Occurrence_Category = pd.read_csv(file_path + 'CADORS_Occurrence_Category.csv', sep=',')
CADORS_Occurrence_Information = pd.read_csv(file_path + 'CADORS_Occurrence_Information.csv', sep=',')
CADORS_Occurrence_Event_Information = pd.read_csv(file_path + 'CADORS_Occurrence_Event_Information.csv', sep=',')
CADORS_Aircraft_Information = pd.read_csv(file_path + 'CADORS_Aircraft_Information.csv', sep=',')
print('Datasets loaded')


# In[ ]:


DATA1 = CADORS_Aircraft_Event_Information
DATA2 = CADORS_Occurrence_Category 
DATA3 = CADORS_Occurrence_Information 
DATA4 = CADORS_Occurrence_Event_Information 
DATA5 = CADORS_Aircraft_Information 


# In[ ]:


DATA1


# In[ ]:


DATA2


# In[ ]:


DATA3


# In[ ]:


DATA4


# In[ ]:


DATA5


# In[ ]:


DATA5.columns


# In[ ]:


DATA3.columns

