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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris.head()


# In[ ]:


sns.set(style="white", color_codes=True)


# In[ ]:


warnings.filterwarnings("ignore")


# In[ ]:


iris.plot(kind='scatter',x='SepalWidthCm',y='PetalLengthCm')


# In[ ]:


sns.FacetGrid(iris, hue="SepalLengthCm", size=9)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


sns.boxplot(x="Species",y="PetalWidthCm",data=iris)


# In[ ]:


sns.boxplot(x="Species",y="SepalWidthCm",data=iris)


# In[ ]:


sns.boxplot(x="Species",y="PetalLengthCm",data=iris)


# In[ ]:


sns.boxplot(x="Species",y="SepalLengthCm",data=iris)


# In[ ]:



ax = sns.stripplot(x="Species", y="SepalLengthCm", data=iris, jitter=True)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[ ]:


sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)


# In[ ]:


sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")


# In[ ]:


from pandas.tools.plotting import andrews_curves


# In[ ]:


andrews_curves(iris.drop("Id", axis=1), "Species")


# In[ ]:


target = iris.iloc[:,6]


# In[ ]:




