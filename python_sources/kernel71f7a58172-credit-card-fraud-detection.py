#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns


# In[ ]:


creditcard = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


creditcard.head()


# In[ ]:


creditcard.describe()


# In[ ]:


creditcard_Class0 = creditcard.loc[creditcard.Class == 0]
creditcard_Class1 = creditcard.loc[creditcard.Class == 1]


# In[ ]:



creditcard_Cl = creditcard.groupby('Class').aggregate({'Class': 'count'}).rename(columns={'Class': 'Class_count'}).reset_index()
print(creditcard_Cl)
sns.barplot('Class', 'Class_count', data = creditcard_Cl)


# In[ ]:


creditcard_Class1.head(2)


# In[ ]:


import pandas_profiling
creditcard_Class1.profile_report()

