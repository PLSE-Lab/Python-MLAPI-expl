#!/usr/bin/env python
# coding: utf-8

# <h1>Titanic Disaster Survival Using Logistic Regression</h1>

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


# 

# **Load the Data**

# In[ ]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")


# **View the data using head function which returns top  rows**

# In[ ]:


train_data.head()
test_data.head()


# **Explaining Dataset**
# 
# survival : Survival 0 = No, 1 = Yes <br>
# pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd <br>
# sex : Sex <br>
# Age : Age in years <br>
# sibsp : Number of siblings / spouses aboard the Titanic parch # of parents / children aboard the Titanic <br>
# ticket : Ticket number fare Passenger fare cabin Cabin number <br>
# embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton <br>
# 
# 
# 

# **Import Seaborn for visually analysing the data**

# In[ ]:


import seaborn as sns


# **Find out how many survived vs Died using countplot method of seaboarn**

# In[ ]:


sns.countplot(x='Survived',data=train_data)


# **Male vs Female Survival**

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train_data)


# **See age group of passengeres travelled **<br>
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph

# In[ ]:


sns.distplot(train_data['Age'].dropna())


# **Fill the missing values**<br> we will fill the missing values for age. In order to fill missing values we use fillna method.<br> For now we will fill the missing age by taking average of all age 

# In[ ]:


train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())


# **We can verify that no more null data exist** <br> we will examine data by isnull mehtod which will return nothing

# In[ ]:


train_data[train_data['Age'].isnull()]
#test_data[train_data['Age'].isnull()]


# **Alternatively we will visualise the null value using heatmap**<br>
# we will use heatmap method by passing only records which are null. 

# In[ ]:


sns.heatmap(train_data.isnull())
#sns.heatmap(test_data.isnull())


# **We can see cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it**

# In[ ]:


train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_data.head()
#test_data.head()


# **Preaparing Data for Model**<br>
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[ ]:


train_data.info()


# **We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems name and ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Embarked and Sex Columns to dummies numerical values******

# In[ ]:


sex=pd.get_dummies(train_data['Sex'],drop_first=True)


# In[ ]:


train_data['Sex_m']=sex


# In[ ]:


train_data.drop(['Name','Sex','Ticket','Embarked','Sex'],axis=1,inplace=True)


# In[ ]:


sex=pd.get_dummies(test_data['Sex'],drop_first=True)


# In[ ]:


test_data['Sex_m']=sex


# In[ ]:


test_data.drop(['Name','Sex','Ticket','Embarked','Sex'],axis=1,inplace=True)


# In[ ]:


test_data.head()


# In[ ]:


#train_data.head()
test_data.tail()


# In[ ]:


features=['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_m']
target='Survived'


# In[ ]:


#test_data[test_data.isnull()==True]


# **Building Model using Logestic Regression**

# **Build the model**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


l_model=LogisticRegression()


# In[ ]:


l_model.fit(train_data[features],train_data[target])


# In[ ]:


predict=l_model.predict(test_data[features])


# **See how our model is performing**

# In[ ]:


submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predict})


# In[ ]:


submission.head()


# **Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
# 
# Note: <br>
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
# F1 score - F1 Score is the weighted average of Precision and Recall.
# 
# 
