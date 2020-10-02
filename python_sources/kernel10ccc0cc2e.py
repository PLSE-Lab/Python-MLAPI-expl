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


df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# # **MSZoning**

# In[ ]:


df["MSZoning"]


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


x=df[["MSZoning","SalePrice"]]


# In[ ]:


grp=x[["MSZoning","SalePrice"]]
y=grp.groupby(["MSZoning"],as_index=False).mean()
y


# In[ ]:


y.plot(kind="bar")


# # **The kind of Zone affects the Sale Price, different kinds of zones sell for diff mean prices**

# In[ ]:


import seaborn as sns


# In[ ]:


sns.boxplot(x="MSZoning",y="SalePrice",data=x)


# # **The Zone does affect the cost(some zones have many outliers), but RH and RM have similar distributions**

# # **LotFrontage**

# In[ ]:


sns.regplot(x="LotFrontage",y='SalePrice',data=df)


# # **Most of the points are all spread across, there is no clear distinction. All kinds of Sale Prices are observed for all values of LotFrontages<150**

# In[ ]:


sns.boxplot(x="LotFrontage",y="SalePrice",data=df)


# # **The box plot suggests that the distribution is not significantly different, many distributions seem alike**

# # **MSSubClass **

# In[ ]:


a=df[["MSSubClass","SalePrice"]]
a.head()


# In[ ]:


sns.boxplot(x="MSSubClass",y="SalePrice",data=a)


# In[ ]:


b=a.groupby(["MSSubClass"],as_index=True).mean()
b.plot(kind="bar")


# 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES

# In[ ]:


sns.boxplot(x="MSSubClass",y="SalePrice",data=df)


# # **LotArea**

# In[ ]:


x=df[df["LotArea"]<75000]
x=x[["LotArea","SalePrice"]]
x.head()


# In[ ]:


x["LotArea"].min()


# In[ ]:


x["LotArea"].max()


# In[ ]:


sns.regplot(x="LotArea",y="SalePrice",data=x)


# # **Street**

# In[ ]:


x=df[["Street","SalePrice"]]
x.head()


# In[ ]:


y=x.groupby("Street",as_index=False).mean()


# In[ ]:


y


# In[ ]:


sns.boxplot(x='Street',y='SalePrice',data=x)


# # **On an average houses with Gravel roads sold for lesser prices then the ones with Pavement. The Gravel kind not having outliers supports the above statement**

# # **Alley**

# In[ ]:


x=df[["Alley","SalePrice"]]
x.head()


# In[ ]:


sns.boxplot(x="Alley",y="SalePrice",data=x)


# In[ ]:


y=x.groupby("Alley",as_index=True).mean()
y


# # **Clear Distinction can be seen between Gravel and Pavement**

# # **LotShape**

# In[ ]:


x=df[['LotShape','SalePrice']]
x.head()


# In[ ]:


sns.boxplot(x='LotShape',y='SalePrice',data=x)


# # **All distributions seem similar**

# In[ ]:


y=x.groupby("LotShape",as_index=True).mean()
y


# # **The averages vary but the distributions are quite similar**

# # **LandContour**

# In[ ]:


x=df[['LandContour','SalePrice']]
x.head()


# In[ ]:


sns.boxplot(x='LandContour',y='SalePrice',data=df)


# In[ ]:




