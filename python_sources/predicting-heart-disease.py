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


df = pd.read_csv ('../input/heart.csv')


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns
sns.set()
df ['age'].plot (kind = 'hist', bins=10)


# In[ ]:


import seaborn as sns
sns.set()
df ['sex'].plot (kind = 'hist', bins=10)


# In[ ]:


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x="age", y="chol", hue="target",data=df)


# In[ ]:


df.plot.hexbin(x= 'trestbps', y = 'thalach',gridsize=25,cmap = 'coolwarm')


# In[ ]:


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x="age", y="chol", hue="target", size = 'chol',data=df)


# In[ ]:


import sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


df.columns


# In[ ]:


X = df [['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca', 'thal']]


# In[ ]:


y= df ['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


print (X_train.describe())
print (X_test.describe())


# In[ ]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier (max_depth=4)
dtree.fit(X_train,y_train)


# In[ ]:


from sklearn.tree import export_graphviz
import graphviz


# In[ ]:


feature_name = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca', 'thal']
target_name = np.array(['No Disease','Heart Disease Presence'])


# In[ ]:


export_graphviz (dtree, 
                out_file = "heart_tree.dot",
                feature_names = feature_name,
                class_names =   target_name,
                 rounded = True,
                 filled = True)


# In[ ]:


with open("heart_tree.dot") as heart_tree_image:
    heart_tree_graph = heart_tree_image.read()
graphviz.Source(heart_tree_graph)


# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print (classification_report(y_test,predictions))
print (confusion_matrix(y_test,predictions))


# In[ ]:




