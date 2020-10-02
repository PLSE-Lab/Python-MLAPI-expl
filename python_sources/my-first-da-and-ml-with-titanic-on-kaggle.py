#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
init_notebook_mode(connected=True)
cufflinks.go_offline()


# In[ ]:


titanic = pd.read_csv('../input/titanic/train.csv')


# Let check the data

# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# It can be clearly seen that the Cabin is missing a lot of data and then the Age columns.
# > Let visualize the missing data

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# # Insight
# Let explore this dataset a little bit
# 

# See how many ppl has survived

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(titanic['Survived'])


# Let see the number group by Male and Female

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(titanic['Survived'],hue=titanic['Sex'])


# The number of female survived is more than that of male. If you have seen the movie, a lot of men sacrified their life for women and children.

# In[ ]:


titanic['Pclass'].value_counts()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(titanic['Pclass'],hue=titanic['Sex'])


# In[ ]:


fg=sns.FacetGrid(data=titanic,col='Sex',height=6)
fg.map(sns.countplot,'Survived',hue=titanic['Pclass'],palette='rainbow').add_legend()


# * Poor people always die *what a pity!
# * It seems like men in 3rd class couldn't survive and almost women in 1st class survived based on our data

# In[ ]:


titanic['Age'].iplot(kind='hist',bins=30)


# People on Titanic were really young, from 20 to 30

# ## Data Cleaning
# Let try to input missing value for **Age**
# 

# In[ ]:


corr = titanic.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,6))
sns.heatmap(corr,mask=mask,cmap='coolwarm',annot=True)


# In[ ]:


sns.boxplot(x='Pclass', y='Age', data=titanic)


# In[ ]:


def inputAge(data):
    age=data[0]
    _class= data[1]
    if pd.isnull(age):
        if _class==1:
            return 37
        elif _class==2:
            return 29
        else:
            return 25
    else:
        return age


# In[ ]:


titanic['Age'] = titanic[['Age','Pclass']].apply(inputAge,axis=1)


# Check data again

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[ ]:


titanic.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sex=pd.get_dummies(titanic['Sex'],drop_first=True)
embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
titanic.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([titanic,sex,embark],axis=1)


# In[ ]:


train.head()


# # Building a Logistic Regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X_train= train.drop('Survived',axis=1)
y_train = train['Survived']

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


test= pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(inputAge,axis=1)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


import plotly.express as px
fig = px.box(test,x='Pclass', y="Fare")
fig.show()


# In[ ]:


boolean = pd.isnull(test['Fare'])
test[boolean]


# In[ ]:


test.loc[152,'Fare'] = 7.89


# In[ ]:


sex=pd.get_dummies(test['Sex'],drop_first=True)
embark=pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)


# In[ ]:


predict = logmodel.predict(test)


# In[ ]:


df= pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


df['Survived'] = predict


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:




