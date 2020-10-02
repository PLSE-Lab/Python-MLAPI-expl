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


# The irragular growth of the breast cells leads to *breast cancer*. Due to the mutation in the gene, the cells start growing in an uncontrolled way. Those uncntrolled growth or tumers could be visualized through the mammogram. According to the study all irregular growths are not cancer. From the mammogram the tumer size, texture etc could be analyzed to classify the cancerous cells.
# In this note book a breast cancer data set has been analyzed. The data set comprises the numerical values derived from the mammogram. 

# # Load data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
data


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# # Exploratory Data Analysis

# In[ ]:


data.nunique()


# In[ ]:


import seaborn as sns
sns.countplot(x="diagnosis", data=data)


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


sns.pairplot(data, hue="diagnosis")


# The data set contains the features:
# * mean_radius
# * mean_texture
# * mean_perimeter
# * mean_area
# * mean_smoothness
# * diagnosis (0=positive and 1=negative)
# 
# We are having 2 classes here to train our model to learn whether a tumor is cancerous or not. The model prediction and accuracy completely depends on the involved feature set. 
# From the pair plot we can visualize that the cancerou and noncancerous tumarous showing diffrent features and in some cases the features are coinsiding i.e. showing indistinguishable feature values.

# # Classification

# In[ ]:


X= data.iloc[:,:5].values
y= data.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)


# In[ ]:


y_pred=neigh.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf) 


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph


# In[ ]:


y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:




