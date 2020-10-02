#!/usr/bin/env python
# coding: utf-8

# Iris data sets. Lets see the data visulization

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


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# In[ ]:


import seaborn as sns


# In[ ]:


iris["Species"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
iris.plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm")
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x="SepalLengthCm",data = iris)
plt.xticks(rotation = "vertical")


# In[ ]:


sns.jointplot(x="SepalLengthCm", y = "SepalWidthCm", data = iris)


# In[ ]:


sns.FacetGrid(iris,size=5, hue="Species").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
                                  


# In[ ]:


sns.boxplot(x="Species", y="PetalLengthCm", data = iris)


# In[ ]:


sns.boxplot(x="Species", y="PetalLengthCm", data = iris)
sns.stripplot(x="Species", y="PetalLengthCm", data = iris)


# In[ ]:


sns.pairplot(iris.drop("Id", axis=1),hue = "Species",size = 2)
    


# In[ ]:




