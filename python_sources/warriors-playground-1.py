#!/usr/bin/env python
# coding: utf-8

# # 2Sig Financial Modeling Competition
# **Stephen Curry and Kevin Durant**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


len(df['id'].unique())


# The # of id (or assets) is 1424

# test part for 3 ids 

# In[ ]:


id_number = df['id'].unique()


# In[ ]:


plt.plot(id_number)


# In[ ]:


id_number[800:900]


# In[ ]:


lenID = {}
for id_no in id_number:
    lenID[id_no] = len(df[df['id'] == id_no])


# In[ ]:


lenID


# In[ ]:


print(id_10['y'].std(axis=0))
plt.plot(id_10['y'])
plt.show()


# In[ ]:


plt.hist(id_10['y'])


# In[ ]:


id_11 = df[df['id'] == 11]
print(id_11['y'].std(axis=0))
plt.plot(id_11['y'])
plt.show()
id_11.shape


# In[ ]:


plt.hist(id_11['y'])


# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.cross_validation import cross_val_score
import matplotlib


# In[ ]:


#id_10.head()
x_train_id_10 = id_10.loc[:, "derived_0":"technical_44"]
x_train_id_10.head()


# In[ ]:


id_10["technical"]


# In[ ]:


adsf


# In[ ]:


model_lasso = LassoCV(0.5).fit(x_train, y_target)


# In[ ]:


mean_values = x_train.mean(axis=0)
x_train.fillna(mean_values, inplace=True).head()


# In[ ]:


y_mean = df['y'].mean(axis=0)
y_target = df['y']
y_target.fillna(y_mean, inplace=True)


# In[ ]:


model_lasso = LassoCV(0.5).fit(x_train, y_target)


# In[ ]:


coef = pd.Series(model_lasso.coef_, index = x_train.columns)


# In[ ]:


imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Lasso')


# In[ ]:





# In[ ]:




