#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode,plot, iplot


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


#import Data. TD is the name of DataFrame
TD=pd.read_csv('../input/train.csv')


# In[ ]:


TD.head()


# **Exploratory Analysis**:
# Who Survived?

# In[ ]:


sns.countplot(x='Survived', data=TD)


# shows that, more than 500 not survived

# To determine, who survived based on their gender:

# In[ ]:


sns.countplot(x='Survived', data=TD, hue='Sex')


# Shows that, the number of died man is 5 times more than the number of died women.

# In[ ]:


sns.countplot(x='Survived', data=TD, hue='Pclass')


# In[ ]:


sns.countplot(x='Survived', data=TD, hue='Sex', palette='RdBu_r')


# In[ ]:


sns.countplot(x='SibSp', data=TD)


# In[ ]:


sns.countplot(x='SibSp', data=TD, hue='Sex')


# In[ ]:


sns.countplot(x='Parch', data=TD)


# In[ ]:


sns.countplot(x='Parch', data=TD, hue='Sex')


# In[ ]:


sns.countplot(x='SibSp', data=TD, hue='Survived')


# In[ ]:


sns.countplot(hue='SibSp', data=TD, x='Survived')


# In[ ]:





# **Analysis the Age of Passengers:******

# In[ ]:


TD['Age'].plot.hist(bins=40)


# In[ ]:





# **Data Cleaning**:

# In[ ]:


TD.isnull()


# In[ ]:


sns.heatmap(TD.isnull(), yticklabels=False, cbar= False, cmap='viridis')


# Shows that, Age and Embarked Culomn have lots of missing values

# To clean the Age missing value, we want to put the mean of age instead of missing value
# But, instead of putting the total mean, we use the mean of the age based on Pclass of the passeneger

# In[ ]:


sns.boxplot('Pclass', y='Age', data=TD)


# In[ ]:


df1 = TD.groupby('Pclass')['Age'].mean().reset_index()
print(df1)


# **Function Code to put age mean intead of missing values:**

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38.23
        elif Pclass == 2:
            return 29.87
        else:
            return 25.14
    else:
            return Age


# In[ ]:


TD['Age'] = TD[['Age', 'Pclass']].apply(impute_age, axis=1)


# ***Check the implemention the above code:***

# In[ ]:


sns.heatmap(TD.isnull(), yticklabels=False, cbar= False, cmap='viridis')


# **As lots of values of Embarked are missing, so we decide to delete the Cabin culomns:**

# In[ ]:


TD.drop('Cabin', axis=1, inplace=True)


# In[ ]:


sns.heatmap(TD.isnull(), yticklabels=False, cbar= False, cmap='viridis')


# **Now, delete any other missing value:**

# In[ ]:


TD.dropna(inplace=True)


# **After treating with missing value, categorical variable must change to binary code:
# in this Data set, Sex and Embarked:**

# In[ ]:


sex=pd.get_dummies(TD['Sex'], drop_first=True)


# In[ ]:


embark=pd.get_dummies(TD['Embarked'], drop_first=True)


# **Add two new column to DaraFrame:**

# In[ ]:


TD=pd.concat([TD, sex, embark], axis=1)


# In[ ]:


TD.head()


# **Delete the columns that we dont need for machine learning algorithm, such as string variable and etc:
# in this case, we want to delete 'Sex', 'Embarked', 'Name', PassengerId'**

# In[ ]:


TD. drop(['Sex', 'Embarked', 'Name', 'PassengerId', 'Ticket'],axis=1, inplace=True)


# In[ ]:


TD. drop(['Ticket'],axis=1, inplace=True)


# In[ ]:


TD.head()


# **Now, the DataFrame is ready to use by any of Machine Learning Algorithm:**

# **Apply Logistic Regression:**

# In[ ]:


X=TD.drop('Survived',axis=1)
y=TD['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train )


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:




