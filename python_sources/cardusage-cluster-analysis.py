#!/usr/bin/env python
# coding: utf-8

# **LIB IMPORTS : **
# ----------------

# In[ ]:


import fun_py
import pandas as pd


# In[ ]:


cardData=pd.read_csv(r"../input/card-usage/CreditCardUsage.csv")


# **FUNCTIONS :**
# --------------
# 

# In[ ]:


fun_py.welcome_msg()
    

    


# In[ ]:





# In[ ]:





# In[ ]:


cardData.isnul


# In[ ]:


cardData.corr()['BALANCE'].sort_values()


# In[ ]:


X = cardData[:]
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[ ]:


X.drop('CUST_ID',axis=1,inplace=True)


# In[ ]:


fun_py.data_groupcols(X)


# In[ ]:


Clus_dataSet = StandardScaler().fit_transform(X)


# In[ ]:


X_BAL_PUR = X.iloc[:,0:3].values
X_BAL_PUR


# In[ ]:


X['BALANCE'].max()


# In[ ]:


import scipy.cluster.hierarchy as sch


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Dendrogram')
plt.xlabel('BALANCE AND PURCHASES')
plt.ylabel('Euclidean distances')
#plt.grid(True)
dendrogram = sch.dendrogram(sch.linkage(X_BAL_PUR, method = 'ward'))
plt.show()


# In[ ]:


X.head(5)


# In[ ]:


import pandas_profiling


# In[ ]:


pr = pandas_profiling.ProfileReport(X)


# In[ ]:


pr


# In[ ]:


X['PURCHASES_FREQUENCY'].min()


# In[ ]:


X['TENURE'].value_counts()


# In[ ]:


X.head()

