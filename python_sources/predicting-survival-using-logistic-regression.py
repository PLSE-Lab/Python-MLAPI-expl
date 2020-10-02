#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic = pd.read_csv('../input/titanic_train.csv')


# ## EDA
# we can use seaborn to create a heatmap to see where we are missing data

# In[ ]:


sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# * Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

# **Lets check the survived/Died  count** 

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x='Survived',data=titanic,palette='RdBu_r')


# **Survived/Died by Sex**

# In[ ]:


sns.countplot(x='Survived',data=titanic,hue='Sex',palette='RdBu_r')


# **Survived/Died by Class**

# In[ ]:


sns.countplot(x='Survived',data=titanic,hue='Pclass',palette='rainbow')


# **Lets check the  Age distribution**

# In[ ]:


sns.distplot(titanic['Age'].dropna(),kde=False,bins=30,color='darkred')


# **Sibling/Spouse count**

# In[ ]:


sns.countplot(x='SibSp',data=titanic)


# **Fair distribution**

# In[ ]:


titanic['Fare'].hist(bins=40,figsize=(8,4))


# ## Data Cleaning
# I want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class. For example:

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass',y='Age',data=titanic)


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[ ]:


def impute_age(cols):
    age=cols[0]
    pClass=cols[1]
    if(pd.isnull(age)):
        if pClass==1:
            return 37
        elif pClass == 2:
            return 29
        else:
            return 24
    else:
        return age


# In[ ]:


titanic['Age']=titanic[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check the heat map again

# In[ ]:


sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Let's go ahead and drop **Cabin** column

# In[ ]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[ ]:


titanic.head()


# **Converting Categoricaal features**
# * I'll need to convert categorical features to dummy variables using pandas! Otherwise machine learning algorithm won't be able to directly take in those features as inputs.

# In[ ]:


sex=pd.get_dummies(titanic['Sex'],drop_first=True)
embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
titanic.head()


# Here we don't need **Name and Ticket** columns. Also we can drop **Sex and Embarked** columns as we have created categorical data for them.

# In[ ]:


titanic.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:


titanic=pd.concat([titanic,sex,embark],axis=1)


# In[ ]:


titanic.head()


# ## Building a Logistic model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(titanic.drop('Survived',axis=1),titanic['Survived'],
                                                 test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lgmodel=LogisticRegression()


# In[ ]:


lgmodel.fit(x_train,y_train)


# In[ ]:


predictions=lgmodel.predict(x_test)


# ## Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


sns.countplot(x=predictions)


# In[ ]:




