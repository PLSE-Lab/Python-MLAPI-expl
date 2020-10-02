#!/usr/bin/env python
# coding: utf-8

# Titanic: Machine Learning from Disaster
# =====================================================

# In[218]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# ------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lnmd
import sklearn as skl


# ## Data Import

# In[172]:


df  = pd.read_csv('../input/train.csv', sep=',')
# df  = pd.read_csv('train.csv', sep=',')


# ## Data Wrangling, Data Transormation and Exploratory Data Analysis

# ### Train Data preparation

# In[173]:


print(df)


# In[174]:


df.head(5)


# In[175]:


df.columns


# VariableDefinitionKey
# * survival Survival 0 = No, 1 = Yes 
# * pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd 
# * sex Sex 
# * Age Age in years 
# * sibsp # of siblings / spouses aboard the Titanic 
# * parch # of parents / children aboard the Titanic 
# * ticket Ticket number 
# * fare Passenger fare 
# * cabin Cabin number 
# * embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[176]:


df.dtypes


# In[177]:


df.describe()


# In[178]:


df.Sex.head(5)


# In[179]:


ordered_sex = ['female', 'male']

# df.Sex = df.Sex.astype("category",
#      ordered=True,
#      categories=ordered_sex).cat.codes

df.Sex = df.Sex.astype(pd.api.types.CategoricalDtype(categories = ordered_sex, ordered = True)).cat.codes


# In[180]:


df.Sex.head(5)


# In[181]:


del df['Name'], df['Embarked'], df['Ticket'], df['Cabin']


# In[182]:


df.dtypes


# In[183]:


df.describe()


# In[184]:


dfc = df.iloc[:,1:]
dfc_res = dfc.corr()
print(dfc_res)


# In[185]:


plt.imshow(dfc_res, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(dfc_res.columns))]
plt.xticks(tick_marks, dfc_res.columns, rotation='vertical')
plt.yticks(tick_marks, dfc_res.columns)
plt.show()


# In[186]:


pd.plotting.andrews_curves(dfc,"Survived")


# In[187]:


pd.plotting.parallel_coordinates(dfc,"Survived")


# In[188]:


x_train = df.iloc[:,2:]
x_train.head(3)


# In[189]:


x_train.isnull().values.any()


# In[190]:


x_train.plot.hist(alpha=0.75)


# In[191]:


#print(x_train["Pclass"].unique())
#print(x_train["Sex"].unique())
print(x_train["Age"].unique())
#print(x_train["SibSp"].unique())
#print(x_train["Parch"].unique())
#print(x_train["Fare"].unique())


# In[192]:


x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())


# In[193]:


x_train.describe()


# In[219]:


min_max_scaler = skl.preprocessing.MinMaxScaler()
x_train_sc = min_max_scaler.fit_transform(x_train)
x_train = pd.DataFrame(data = x_train_sc[0:,0:],    # values
             # index = x_train_sc[0:,0], # 1st column as index
             columns = x_train.columns) # 1st row as the column names


# In[195]:


x_train.values.shape


# In[196]:


x_train["Age"].plot.hist(alpha=0.5)


# In[197]:


x_train["Pclass"].plot.hist(alpha=0.5)
x_train["Sex"].plot.hist(alpha=0.5)


# In[198]:


x_train["SibSp"].plot.hist(alpha=0.5)


# In[199]:


x_train["Parch"].plot.hist(alpha=0.5)


# In[200]:


x_train["Fare"].plot.hist(alpha=0.5)


# In[201]:


y_train = df.iloc[:,[1]]
y_train.head(3)


# In[202]:


y_train.isnull().values.any()


# In[203]:


y_train["Survived"].unique() 


# In[204]:


y_train.plot.hist(alpha=0.75)


# ### Test Data preparation

# In[205]:


df  = pd.read_csv('../input/test.csv', sep=',')
# df  = pd.read_csv('test.csv', sep=',')


# In[206]:


del df['Name'], df['Embarked'], df['Ticket'], df['Cabin']


# In[207]:


df.head(5)


# In[208]:


ordered_sex = ['female', 'male']
df.Sex = df.Sex.astype(pd.api.types.CategoricalDtype(categories = ordered_sex, ordered = True)).cat.codes


# In[209]:


df.isnull().values.any()


# In[210]:


#print(df["Pclass"].unique())
#print(df["Sex"].unique())
#print(df["Age"].unique())
#print(df["SibSp"].unique())
#print(df["Parch"].unique())
print(df["Fare"].unique())


# In[211]:


df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df.head(5)


# In[212]:


x_test = df.iloc[:,1:]
x_test.head(5)


# In[220]:


min_max_scaler = skl.preprocessing.MinMaxScaler()
x_test_sc = min_max_scaler.fit_transform(x_test)
x_test = pd.DataFrame(data = x_test_sc[0:,0:],    # values
             #index = x_train_sc[0:,0],    # 1st column as index
             columns = x_test.columns) # 1st row as the column names


# In[214]:


x_test.head(5)


# ## Data Modeling

# ### Logistic Regression, 1st attempt

# In[215]:


model = lnmd.LogisticRegression(solver='lbfgs')
model.fit(x_train.values, y_train.values.ravel())
score = model.score (x_train, y_train)
print(score)


# ## RESULT OUTPUT

# In[216]:


y_pred = model.predict(x_test)
print(y_pred)
type(y_pred)


# In[217]:


y = pd.DataFrame(data = y_pred[0:],    # values
             # index = x_train_sc[0:,0],    # 1st column as index
             columns = ["Survived"]) # 1st row as the column names

# y.head(5)
output  = pd.DataFrame.join(df[["PassengerId"]], y[["Survived"]])
output.head(5)
output.to_csv("output.csv",index=False)

