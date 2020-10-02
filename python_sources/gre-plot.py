#!/usr/bin/env python
# coding: utf-8

# In[118]:


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


# In[117]:


def printline(no_of_lines):
    print('_' * no_of_lines)


# In[55]:


import plotly
print(plotly.__version__)
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[4]:


df = pd.read_csv('../input/Admission_Predict.csv')
print(df.columns)
printline(100)
print(df.info())


# In[5]:


df1 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
print(df1.columns)
printline(100)
print(df1.info())


# In[41]:


dataset = pd.concat([df,df1])
print(dataset.columns)
printline(100)
print(dataset.describe())


# In[42]:


dataset


# In[43]:


plt.xlabel("Student Research record")
plt.ylabel("No. of Student")
sns.barplot( x=["Done Research","Not Done Research"],
            y=[len(dataset[dataset["Research"] == 1]), len(dataset[dataset["Research"] == 0])])


# In[72]:


print(dataset.columns)
fig, ax = plt.subplots()
sns.distplot(dataset["CGPA"], bins=25, color="g", ax=ax)
plt.show()


# In[40]:


corr_table = dataset.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_table,annot=True)
plt.show()


# ## Appying Machine leaning models

# In[78]:


X_col = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
X_data = dataset[X_col]
y_data = dataset['Chance of Admit ']


# In[98]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


# In[103]:


X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)


# In[104]:


reg = DecisionTreeRegressor().fit(X_train, y_train)


# In[107]:


y_pred = reg.predict(X_val) 


# In[108]:


r2_score(y_val, y_pred)  


# In[129]:


from joblib import dump, load
dump(reg, 'gre.joblib') 


# In[119]:


clf = load('../input/gre.joblib') 


# In[115]:


y_pred = clf.predict(X_val) 
r2_score(y_val, y_pred)  


# In[130]:


os.chdir("/kaggle/input")
path = os.getcwd()
print(path)
os.chmod(path,777)


# In[ ]:




