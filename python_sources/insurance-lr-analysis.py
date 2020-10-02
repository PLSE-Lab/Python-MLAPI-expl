#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('pwd')
get_ipython().system('ls /kaggle/input')
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os



# In[ ]:


data=pd.read_csv(r"../input/insurance.csv")


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


graphs = sns.pairplot(data)
graphs.set()


# In[ ]:


num_data=data.select_dtypes(include=np.number)
cat_data=data.select_dtypes(exclude=np.number)


# In[ ]:


encode_cat_data = pd.get_dummies(cat_data)


# In[ ]:


fin_df= [num_data,encode_cat_data]
fin_data=pd.concat(fin_df,axis=1)
fin_data.head()


# In[ ]:


graphs = sns.pairplot(fin_data)
graphs.set()


# In[ ]:


boxP = sns.boxplot(data = fin_data.age ,orient = 'h' ,color = 'red')


# In[ ]:


fin_data.corr()

