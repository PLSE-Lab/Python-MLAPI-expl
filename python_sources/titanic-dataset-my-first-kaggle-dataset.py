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


# <a id="toc_section"></a>
# ## Table of Contents
# * [Importing all the Required Libraries](#section1)
# * [Exploring the Data](#section2)
# * [Visualizing given dataset](#section3)
# * [Building the Feature Engineering Machine](#section4)
# * [Modelling](#section13)

# <a id="section1"></a>
# ## Importing the libraries
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install ppscore')
get_ipython().system('pip install lazypredict')


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


train.head()


# Kaggle doesn't support ppscore. But **I would highly advise everyone to use ppscore** which is a much better alternative to the same old correlation. Because there are a lot of trends which ppscore captures which correlation fails to do.

# In[ ]:


import ppscore as pps


# In[ ]:


pps.matrix(train)


# <a id="section2"></a>
# ## Exploring the data
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# We see that there is a lot of null values here!

# In[ ]:


sns.set_style('whitegrid')


# <a id="section3"></a>
# ## Visualizing given dataset
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


sns.countplot(x=train['Survived'],data=train,hue='Sex')
plt.show()


# In[ ]:


sns.countplot(x=train['Survived'],data=train,hue='Pclass')
plt.show()


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=True,bins=30)


# In[ ]:


train.info()


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train["Fare"].hist(bins=70,figsize=(10,4))
plt.show()


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# <a id="section4"></a>
# ## Building the Feature Engineering Machine
# 
# ### [Back To Table of Contents](#toc_section)
# 

# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age']=train[["Age","Pclass"]].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


train.drop('Cabin',inplace=True,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


sex=pd.get_dummies(train['Sex'])
sex


# Male and Female are multicollinear columns. We don't want multicollinearity in the dataset as it is bad for the model as it will make the multicollinear columns as a whole more statistically significant and reduce the significance of other columns in the model.

# In[ ]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
sex


# In[ ]:


embark =pd.get_dummies(train['Embarked'],drop_first=True)
embark


# In[ ]:


train=pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


import re
l=[]
x=0
for i in train["Name"]:
    if(str(i).find("Mr.")>0 or str(i).find("Mrs.")>0):
        l.append(1)
    else:
        l.append(0)
    print(l[x],i)
    x+=1


# In[ ]:


train['Maritial_Status'] = l


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train.head()
y=train["Survived"]


# In[ ]:


x=train.drop("Survived",axis=1)


# In[ ]:


import lazypredict
import sys


# In[ ]:


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.4,random_state =23)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# <a id="section13"></a>
# # Modelling
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


scaler.fit(train)
scaled_features=scaler.transform(train)
train=pd.DataFrame(scaled_features,columns=train.columns)
train.head()


# In[ ]:


from sklearn.model_selection import train_test_split as tits


# In[ ]:


x_train,x_test,y_train,y_test=tits(x,y,test_size=0.2,random_state=23)


# In[ ]:


from sklearn.linear_model import LogisticRegression as lr


# In[ ]:


logmodel=lr()


# In[ ]:


logmodel.fit(x_train,y_train)


# In[ ]:


predictions =logmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report as cr


# In[ ]:


print(cr(y_test,predictions))


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test.isna().sum()


# In[ ]:


test['Age']=test[["Age","Pclass"]].apply(impute_age,axis=1)


# In[ ]:


test.drop('Cabin',inplace=True,axis=1)


# In[ ]:


sex=pd.get_dummies(test['Sex'])
sex


# In[ ]:


sex=pd.get_dummies(test['Sex'],drop_first=True)
sex


# In[ ]:


embark =pd.get_dummies(test['Embarked'],drop_first=True)
embark


# In[ ]:


test=pd.concat([test,sex,embark],axis=1)


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


import re
l=[]
xx=0
for i in test["Name"]:
    if(str(i).find("Mr.")>0 or str(i).find("Mrs.")>0):
        l.append(1)
    else:
        l.append(0)
    print(l[xx],i)
    xx+=1


# In[ ]:


test['Maritial_Status'] = l


# In[ ]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


x.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rdmf = RandomForestClassifier(n_estimators=100,max_depth=5, criterion='entropy')
rdmf.fit(x,y)


# In[ ]:


scaler.fit(test)
scaled_features=scaler.transform(test)
test=pd.DataFrame(scaled_features,columns=test.columns)
test.head()


# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x,y)


# In[ ]:


test['Fare'] = test['Fare'].fillna((test['Fare'].mean()))


# After a lot of experimenting I went with the random forest model as it gave the best accuracy.

# In[ ]:


predictions =rdmf.predict(test)


# In[ ]:


ft=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


predi = pd.DataFrame(predictions, columns=['predictions'])
predi.head()


# In[ ]:


data = [ft["PassengerId"], predi["predictions"]]
headers = ["PassengerId", "Survived"]
final = pd. concat(data, axis=1, keys=headers)


# In[ ]:


final.to_csv("res1.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




