#!/usr/bin/env python
# coding: utf-8

# Chao Tong experimental kernel functions

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


student = pd.read_csv("../input/student-mat.csv")


# In[ ]:


student.describe()


# In[ ]:


student.head()


# In[ ]:


list(student.columns)


# In[ ]:


student.isnull()


# In[ ]:


import seaborn as sns


# In[ ]:


b = sns.factorplot(x="school", y="age", hue="sex", data=student, kind="bar", palette="muted" )


# In[ ]:


jp = sns.jointplot(x="studytime", y="absences", data=student)


# In[ ]:


sns.distplot(student.G1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


# In[ ]:


X,y = student.iloc[:,:-3],student.iloc[:,-2]


# In[ ]:


le=LabelEncoder()
for col in X.columns.values:
    if X[col].dtypes=='object':
        le.fit(X[col].values)
        X[col]=le.transform(X[col])


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


# In[ ]:


predicted = cross_val_predict(regr, X_test, y_test, cv=10)

fig, ax = plt.subplots()
ax.scatter(y_test, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

