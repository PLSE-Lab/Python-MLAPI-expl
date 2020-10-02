#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Loading Liberary
import numpy as np 
import pandas as pd 
import os
import json
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ad_compaign_data = pd.read_csv('../input/data.csv')
ad_compaign_data.head()


# In[ ]:


ad_compaign_data.info()


# In[ ]:


ad_compaign_data.describe()


# In[ ]:





# In[ ]:


ad_compaign_data['age'].unique()


# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 1, 1)
plt.title('Distribution of clicks');
ad_compaign_data['clicks'].plot('hist', label='clicks');
plt.legend();


# In[ ]:




