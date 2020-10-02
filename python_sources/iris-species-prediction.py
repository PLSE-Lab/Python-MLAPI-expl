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


# Let's import some useful libraries that we are using throughout our visualisation.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from pandas.plotting import parallel_coordinates
get_ipython().run_line_magic('matplotlib', 'inline')


# Load and read Iris data.
# Quick look at column of data.

# In[ ]:


data= pd.read_csv("../input/Iris.csv")
data.columns


# Let's check the data type of each column 

# In[ ]:


data.dtypes


# In[ ]:


data.set_index('Id', inplace=True)
data.head(10)


# Let's find the correlation

# In[ ]:


data.corr()


# In[ ]:


sns.clustermap(data.corr(), method = 'single', cmap = 'coolwarm')


# In[ ]:


sns.countplot(x='Species',data=data)


# In[ ]:


sns.violinplot(x="Species", y="PetalLengthCm", data=data, palette='rainbow')


# From above plotting, we can see that Petal  length seems to increase from setosa to versicolor to virginica.

# In[ ]:


sns.boxplot(x = 'Species', y = 'PetalWidthCm', data =data, palette='rainbow')


# From the above box plot, we can see that Iris- virginica has more petal width compared to other.

# In[ ]:


sns.swarmplot(x = 'Species', y = 'SepalLengthCm', data = data, palette = 'rainbow')


# In[ ]:


sns.stripplot(x="Species", y="SepalWidthCm", data=data)


# In[ ]:


sns.pairplot(data, hue="Species", palette="husl")


# In[ ]:


sns.pairplot(data,
             x_vars=["SepalWidthCm", "SepalLengthCm"],
             y_vars=["PetalWidthCm", "PetalLengthCm"], hue = 'Species')


# In[ ]:


parallel_coordinates(data, 'Species')


# In[ ]:


from sklearn.model_selection import train_test_split
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = data.Species
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn. linear_model import LogisticRegression
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[ ]:




