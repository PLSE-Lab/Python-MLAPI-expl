#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read Dataset**

# In[ ]:


data_titanic = pd.read_csv('../input/titanic/train.csv')


# **Print 5 top rows**

# In[ ]:


data_titanic.head(5)


# **See all columns**

# In[ ]:


data_titanic.columns


# **Count total Passengers**

# In[ ]:


print('Total passengers', str(len(data_titanic)))


# **Analyze Data**

# In[ ]:


sns.countplot( x = 'Survived', data = data_titanic)


# In[ ]:


sns.countplot( x = 'Survived', hue = 'Sex', data = data_titanic)


# In[ ]:


sns.countplot( x = 'Survived', hue = 'Pclass', data = data_titanic)


# In[ ]:


data_titanic['Age'].hist().plot()


# In[ ]:


data_titanic['Fare'].hist().plot(bins = 20, figsize = (10, 5))


# In[ ]:


data_titanic.info()


# In[ ]:


sns.countplot( x = 'SibSp', data = data_titanic)


# **Data Wrangling**

# In[ ]:


data_titanic.isnull()


# In[ ]:


data_titanic.isnull().sum()


# In[ ]:


sns.heatmap(data_titanic.isnull(), cmap = 'viridis')


# In[ ]:


sns.boxplot(x = 'Pclass', y = 'Age', data = data_titanic)


# In[ ]:


data_titanic.head(7)


# In[ ]:


data_titanic.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


data_titanic.head(5)


# In[ ]:


data_titanic.dropna(inplace = True)


# In[ ]:


sns.heatmap(data_titanic.isnull())


# In[ ]:


data_titanic.isnull().sum()


# In[ ]:


data_titanic.head(5)


# In[ ]:


sex = pd.get_dummies(data_titanic['Sex'], drop_first = 'True')
sex.head(5)


# In[ ]:


embarked = pd.get_dummies(data_titanic['Embarked'], drop_first = True)
embarked.head(5)


# In[ ]:


Pclass = pd.get_dummies(data_titanic['Pclass'], drop_first = True)
Pclass.head(5)


# In[ ]:


data_titanic = pd.concat([data_titanic, sex, embarked, Pclass], axis = 1)
data_titanic.head(5)


# In[ ]:


data_titanic.drop(['Sex', 'Name', 'PassengerId', 'Ticket', 'Embarked'], axis = 1, inplace = True)


# In[ ]:


data_titanic.head(5)


# In[ ]:


data_titanic.drop(['Pclass'], axis = 1, inplace = True)


# In[ ]:


data_titanic.head(5)


# **Train Data**

# In[ ]:


x = data_titanic.drop('Survived', axis = 1)
y = data_titanic['Survived']


# In[ ]:


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


classification_report(y_test, predictions)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, predictions)

