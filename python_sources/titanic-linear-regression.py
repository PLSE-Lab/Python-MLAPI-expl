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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head()


# ## The first step is look in the data to find null/missing values

# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')


# ## Let's drop the columns with a high number of missing values and the columns with no relevant information

# In[ ]:


columns = ['PassengerId','Name','Ticket','Cabin']
train.drop(columns, axis=1, inplace = True)


# In[ ]:


train.head()


# ## **Let's look again the heatmap of null/missing values**

# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')


# ## We can see a high number of missing values in the column AGE
# 
# **We can fill those missing informations with the mean of each class**

# In[ ]:


sns.boxplot('Pclass','Age',data = train, palette = 'inferno')


# ## Let's create a function to extract the age according the class of each passenger

# In[ ]:


def Class_Mean_Age(column):
    age = column[0]
    Class = column[1]
    
    if pd.isnull(age):
        if Class ==1:
            return 37
        elif Class == 2:
            return 29
        else:
            return 24
    else:
        return age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(Class_Mean_Age, axis=1)


# ## To finish, let's drop the line with the last missing value in the Embarked column

# In[ ]:


train.dropna(inplace = True)


# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')


# # With a clean dataset, We can explore some relations with our target column

# In[ ]:


sns.countplot('Survived', data = train, hue = 'Sex', palette = 'magma')

# Count of survivers by sex


# In[ ]:


sns.countplot('Pclass', data = train, hue = 'Survived', palette = 'inferno')

# Count of Survived by Passenger Class


# ## Let's use get_dummies to transform the categorical data

# In[ ]:


Sex = pd.get_dummies(train['Sex'], drop_first = True) 
Embarked = pd.get_dummies(train['Embarked'], drop_first = True)


# ## Now We can drop the original columns Sex and Embarked

# In[ ]:


train.drop(['Sex','Embarked'], axis = 1, inplace = True)


# ## Finishing, let's concating the new columns Sex and Embarked to our dataset Train

# In[ ]:


train = pd.concat([train, Sex, Embarked], axis = 1)
train.head()


# # Splitting the data into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.drop('Survived', axis = 1), train['Survived'], test_size = 0.3)


# # Creating a model
# 
# **Let's create a linear model and fit it with the train splitted data**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel =  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='auto',
          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
logmodel.fit(X_train,y_train)


# ## Creating a prediction

# In[ ]:


predictions = logmodel.predict(X_test)


# ## Comparing the results

# In[ ]:


print(classification_report(y_test,predictions))


# ## The model had accuracy of 80% to predict non survivors and 75% to predict survivors

# # Now We can repeat the process to test dataset

# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
test.head()


# In[ ]:


columns = ['PassengerId','Name','Ticket','Cabin']
test.drop(columns, axis=1, inplace = True)


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(Class_Mean_Age, axis=1)


# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')


# ## On this test dataset We have a missing value in the column Fare. 
# ## Exclude this line is not the best solution, for this, I'll fill this value with the Fare mean of the class

# In[ ]:


group = test.groupby('Pclass')
group.Fare.mean()


# In[ ]:


def meanFareClass(column):
    fare = column[0]
    Class = column[1]
    
    if pd.isnull(fare):
        if Class==1:
            return 94.28
        elif Class==2:
            return 22.2
        else:
            return 12.46
    else:
        return fare


# In[ ]:


test['Fare'] = test[['Fare','Pclass']].apply(meanFareClass, axis=1)


# In[ ]:


plt.figure(figsize = (12,6))
sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')


# # Dataset clean, is time to change or categorical parameters

# In[ ]:


Sex = pd.get_dummies(test['Sex'], drop_first = True) 
Embarked = pd.get_dummies(test['Embarked'], drop_first = True)
test.drop(['Sex','Embarked'], axis = 1, inplace = True)


# In[ ]:


test = pd.concat([test, Sex, Embarked], axis = 1)
test.head()


# # Now we can make the predictions

# In[ ]:


predictions = logmodel.predict(test)


# # Creating the submission file to kaggle

# In[ ]:


Id = pd.read_csv('../input/titanic/test.csv')
Id = Id.PassengerId


# In[ ]:


submission = pd.DataFrame({"PassengerId": Id,"Survived": predictions})
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)

