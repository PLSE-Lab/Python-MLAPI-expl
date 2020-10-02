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


# # 1. Importing required libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sn
import pandas_profiling
import zipfile 


# In[ ]:


# This import helps to run more than one command in single cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# # 2. Importing data and getting some basic insights

# In[ ]:


data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.profile_report()


# * this dataset have 215 entries and 15 features.
# * salary feature has null values.
# * they must be filled or removed.
# * in next step we will fill it with the mean value

# In[ ]:


salary_mean = data['salary'].mean()
data['salary'].fillna(salary_mean, inplace=True)


# In[ ]:


data['gender'].value_counts()
data['ssc_b'].value_counts()
data['hsc_b'].value_counts()
data['hsc_s'].value_counts()
data['degree_t'].value_counts()
data['specialisation'].value_counts()


# In[ ]:


def count_plot(x_cols, y_col, data):
    
    for x_col in x_cols:
        sn.countplot(x_col,hue= y_col, data=data )
        plt.show()


# In[ ]:


cat_cols= ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
count_plot(cat_cols, 'status',data)


# Dropping "ssc_b" and "hsc_b" columns because they does not affect the status of placement

# In[ ]:


data.drop(["ssc_b","hsc_b"],axis=1,inplace=True)


# # 3. feature labeling

# In[ ]:


data['gender'] = data.gender.map({"M":0, "F":1})
data['hsc_s'] = data.hsc_s.map({"Commerce":0, "Science":1, "Arts":2})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0, "Sci&Tech":1, "Others":2})
data["specialisation"] = data.specialisation.map({"Mkt&Fin":0, "Mkt&HR":1})
data["workex"] = data.workex.map({"Yes":1, "No":0})
data["status"] = data.status.map({"Placed":1, "Not Placed":0})


# # 4. Classification on Status of placement

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop(['status',"salary","sl_no"],axis=1)
Y = data['status']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[ ]:


Dec_tree = DecisionTreeClassifier(criterion='entropy')


# In[ ]:


Dec_tree.fit(x_train,y_train)
y_pred = Dec_tree.predict(x_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


Ran_for = RandomForestClassifier(criterion="entropy")


# In[ ]:


Ran_for.fit(x_train, y_train)
y_pred_ranfor = Ran_for.predict(x_test)


# In[ ]:


accuracy_score(y_test, y_pred_ranfor)
print(classification_report(y_test,y_pred_ranfor))


# In[ ]:


cats = list(X.columns)
imps = Ran_for.feature_importances_

for z in zip(cats, imps):
  print(z,"\n")


# * hsc_s does not have much feature importance
# * degree_t does not have much feature importance
# * so next time training we can remove this two features
# 
# 
