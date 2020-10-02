#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# how to give multiple assignment


# In[ ]:


mike,sarah,bob=21,16,13


# In[ ]:


mike


# In[ ]:


age,name=22,"shruti"


# In[ ]:


age


# In[ ]:


#create 2 variable
age1=12
age2=18


# In[ ]:


age1+age2


# In[ ]:


age1-age2


# In[ ]:


age2%age1


# In[ ]:


firstName="shruti"
lastName="Nagpurkar"


# In[ ]:


firstName+ " "+ lastName


# In[ ]:


"Hi" * 10


# In[ ]:


sentence="shruti was playing basketball"


# In[ ]:


sentence[0]


# In[ ]:


sentence[0:6]
#index starts with 0 and after : end + 1


# In[ ]:


sentence[:1]


# In[ ]:


sentence[:-8]


# In[ ]:


#placeholders in strings
name="jake"
sent="%s is 15 year old"
sent%name


# In[ ]:


sent%("shruti")


# In[ ]:


sent="%s %s is the principal"
sent%("prakash","nitin")


# In[ ]:


sent="%s is %d year old"
sent%("radha",12)


# In[ ]:




