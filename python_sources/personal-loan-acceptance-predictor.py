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


# In[ ]:


df=pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
df.head()


# All the feature all numerical so we dont need to perform any kind of transformation

# Checking if the data has null values or not

# In[ ]:


import seaborn as sns
sns.heatmap(df.isnull())


# So as par the heatmap there are no null values in the data

# In[ ]:


df.shape


# In[ ]:


X=pd.DataFrame(columns=['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','CreditCard','Online'],data=df)


# In[ ]:


Y=df['Personal Loan']


# In[ ]:


sns.countplot(data=df,x='Securities Account',hue='Personal Loan',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='Age',hue='Personal Loan',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='Income',hue='Personal Loan',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='Family',hue='Personal Loan',palette='Set2')


# In[ ]:


sns.countplot(data=df,x='Education',hue='Personal Loan',palette='Set2')


# Now we will see feature correlation to help us choose the best features for training the model

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(df[df.corr().index].corr(),annot=True)
plt.show()


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_impo=pd.Series(model.feature_importances_,index=X.columns)
feat_impo.nlargest(5).plot(kind='barh')


# As we can see features mentioned in the graph are the top features which we will use

# In[ ]:


X=pd.DataFrame(columns=['Family','CCAvg','Education','Income','CD Account'],data=df)
X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
DT=DecisionTreeClassifier()
max_depth = range(1, 11, 2)
print(max_depth)
param_grid = dict(max_depth=max_depth)
grid_search = GridSearchCV(DT, param_grid,scoring="accuracy", n_jobs=-1, cv=10)
grid_result = grid_search.fit(X_train,Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:


DT=DecisionTreeClassifier(max_depth=5)


# In[ ]:



DT.fit(X_train,Y_train)


# In[ ]:


Y_pred=DT.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,Y_pred))


# In[ ]:


print(confusion_matrix(Y_test,Y_pred))

