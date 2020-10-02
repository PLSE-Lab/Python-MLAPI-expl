#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/Autism_Data.arff',na_values="?")


# In[ ]:


data.head(10)


# In[ ]:


data.info()


# In[ ]:


# data.replace("?",np.nan,inplace=True)


# In[ ]:


data.head(20)


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(data.isnull(),cmap="viridis",cbar=False,yticklabels=False)


# In[ ]:


data['age']=data['age'].apply(lambda x:float(x))


# In[ ]:


data['age'].describe()


# In[ ]:


# data_p=data
# data_p.dropna(inplace=True)


# In[ ]:


data.loc[data.age == 383, 'age'] = 38


# In[ ]:


data['age'].mean()


# In[ ]:


data["sum_missing_rowWise"] = data.isnull().sum(axis=1)


# In[ ]:


data.head().iloc[:,0:10]


# In[ ]:


# unnecessary, "result" is already given and is the sum of score results!
# data["a_scores_sum"] = data.iloc[:,0:10].sum(axis=1)


# In[ ]:


data.to_csv("autism-screening.csv.gz",index=False,compression="gzip")


# In[ ]:




