#!/usr/bin/env python
# coding: utf-8

# introduction
# website phishing dataset problem. Fitting Decision tree and creating confusion matrix of predicted values and real values I was able to get 87% accuracy 

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


# # Step  1. #import dataset

# In[ ]:


data = pd.read_csv('../input/Website Phishing.csv')
data.head()


# # STEP # 2: IMPORT Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

a=len(data[data.Result==0])
b=len(data[data.Result==-1])
c=len(data[data.Result==1])
print(a,"times suspecious(0) repeated in Result")
print(b,"times phishy(-1) repeated in Result")
print(c,"times legitimate(1) repeated in Result")
sns.countplot(data['Result'])
sns.heatmap(data.corr(),annot=True)
data.info()
data.describe()


# 

# In[ ]:





# # 3 adding all data to X and label to Y**

# In[ ]:


# adding all data to X and label to Y
x = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[ ]:


x.head()


# In[ ]:


y.head()


# # 4 using cross validation with key !
# # Splitting the data** 
# 

# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# # 4 decision-tree classifier using entropy Gini is the default selection criteria
# **# Entropy is a function to measure the quality of split 
# **
# **

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy') # function to measure the quality of split 
tree.fit(x_train,y_train)
score = cross_val_score(tree, x, y, cv= 10) #
print(score)
print(score.mean())
y_pred = tree.predict(x_test)



# In[ ]:


from sklearn import metrics


# In[ ]:


acc = metrics.accuracy_score(y_test,y_pred)
print(acc)


# In[ ]:


from sklearn.metrics import confusion_matrix

y_true = [1, 0, -1]
y_pred = [0, 0, -1]

classes=[1, 0, -1]

confusion_matrix(y_true, y_pred, labels=[1, 0, -1])


# In[ ]:


from sklearn.metrics import f1_score(y_true, y_pred)
y_true = [1, 0, -1]
y_pred = [0, 0, -1]

