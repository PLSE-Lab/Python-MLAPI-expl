#!/usr/bin/env python
# coding: utf-8

# In[24]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()


# In[25]:


df = pd.read_csv("../input/train.csv")
dg = pd.read_csv("../input/test.csv")
#print(df.dtypes)
#df.head()
cols2Keep = dg.columns[dg.isnull().sum() == 0]


# In[28]:


df.isnull().sum().sort_values(ascending = False)


# In[35]:


df.BsmtExposure.value_counts()


# In[32]:


df.PoolQC.value_counts()


# In[33]:


df.MiscFeature.value_counts()


# In[34]:


df.Alley.value_counts()


# In[31]:


df.shape


# In[ ]:





# In[ ]:





# In[19]:


y = df.SalePrice
X = df[cols2Keep]
newX = dg[cols2Keep]


# In[20]:


print(X.isnull().sum())


# In[11]:


from sklearn.preprocessing import Imputer

my_imputer = Imputer() 


# In[ ]:


df.columns.values


# In[ ]:


varnames = df.columns.values

for varname in varnames:
    if df[varname].dtype == 'object':
        lst = df[varname].unique()
        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))


# In[ ]:


df.describe()


# In[ ]:


myVars = ["LotArea","OverallQual","OverallCond","BedroomAbvGr","KitchenAbvGr","FullBath","TotRmsAbvGrd"]


# In[ ]:


df.plot.scatter(x = "LotFrontage", y ='SalePrice')


# In[ ]:


def plotVar(name):
    df.plot.scatter(x = name, y ='SalePrice')
    plt.title(name + " vs price")


# In[ ]:


for n in myVars:
    plotVar(n)


# In[ ]:


from sklearn import linear_model 


# In[ ]:


X_train = df[myVars]
y_train = df['SalePrice']


# In[ ]:


model = linear_model.LinearRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


dg = pd.read_csv("../input/test.csv")
print(dg.dtypes)
dg.head()


# In[ ]:


X_test = dg[myVars]


# In[ ]:


y_test = model.predict(X_test)


# In[ ]:


sum(y_test < 0)


# In[ ]:


dg.head()


# In[ ]:


dg['SalePrice'] = y_test


# In[ ]:


dg.head()


# In[ ]:


outfile = dg[['Id','SalePrice']]


# In[ ]:


outfile[outfile['SalePrice'] < 0]


# In[ ]:


outfile.loc[756,'SalePrice']  = 0


# In[ ]:


outfile.to_csv('output.csv', index = False)


# In[ ]:




