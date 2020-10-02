#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# In this Notebook, we will predict which passenger is going to survive the Titanic Disaster by using various classifiers of machine learning. Along with that you will understand how to visualize and prepare data for model training.

# # WORKFLOW STAGES
# 
# 1. Acquire training and test dataset
# 2. Wrangle, prepare, cleanse the data.
# 3. Model, predict and solve the problem.
# 4. Generate the final solution.
# 5. Submit the results.

#  ## 1. Acquire training and test dataset

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


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# ## 2. Wrangle, prepare, cleanse the data.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Training Data
# 1. Lets first check and clean the training data we have loaded

# In[ ]:


train.head()


# Inorder to prepare data its always a better Idea to visualize your dataset. As you will get a clear idea on the missing values in dataset.

# In[ ]:


sns.heatmap(train.isnull(),cbar=False)


# From the above figure we can clearly state that 30% of age data is missing while in Cabin more than 90% of data is missing.
# Our idea would be to fill up the missing value in age based upon its relation with other feature and drop the Cabin column. 
# 
# Now before updating the Age values lets visualize few other column as well

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=train,palette='rainbow')


# So from the above to plots we get to know that most of the people who died in the tragerdy were from 3rd Class and embarked as S. But this data wont help in filling up the missing age.
# 
# So lets try creating a box diagram for age with respect to PClass because from this we will get the age group of people in each class.  

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# So As expected older people tends to belong to 1st class and so on.
# So now we will take mean of the age for each PClass and assign it to the passenger whos age are missing based upon there PClass.

# In[ ]:


def compute_age(col):
    age = col[0]
    P_class = col[1]
    if pd.isnull(age):
        if P_class == 1:
            return 37
        elif P_class ==2:
            return 29
        else:
            return 24
    else:
        return age
    


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(compute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),cbar=False)


# As you can see that now we age for all the passengers.
# For Cabin more than 90% of data is missing so we will drop the column.

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# Now, as you can see there is no Cabin Column. And we have also assigned age to the passengers with missing age. Now will drop the row which may have any missing value and was not visible during visualization.

# In[ ]:


print("Number of rows before dropping any row with missing value is: ",train.shape[0])
train.dropna(inplace=True)
print("Number of rows after dropping any row with missing value is: ",train.shape[0])


# So we have successfully removed two rows from the train dataset as they had some missing values. So now are we ready to use this dataset...?
# No, Not yet. As in Machine learning we cannot pass raw data as input. i.e the data type cannot we object. It has to be either float or int. Lets check...

# In[ ]:


train.info()


# So we found that we had 4 columns of object dtype out of which we will get dummies value for sex and embark as they have only 2 & 3 class respectively. And get rid of Name and Ticket column as they are hardly of any use in our current problem statement.

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)
train.head()


# ### 2.2 Test Data

# We will follow the same procedure as we did for training data

# In[ ]:


sns.heatmap(test.isnull(),cbar=False)


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(compute_age,axis=1)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)
test.head()


# In[ ]:


sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(), inplace=True)


# In[ ]:


train.head()


# ## 3. Model Prediction and Solving Problem
# 
# In this section I will be running few classifiers, perform prediction on each classifier, compute the confusion matrix and classification report and based on which choose the best out of them for prediction on the test set.
# 
# Classifiers:
# 1. Logistic Regression
# 2. DecisionTreeClassifier
# 3. Random Forest Classification
# 4. SVM
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.20, 
                                                    random_state=101)


# ### 3.1 Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


prediction = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# ### 3.2 DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)


# In[ ]:


dt_pred = dt_model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))


# ### 3.3 Random Forest Classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf= RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)


# In[ ]:


rf_pre=rf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))


# ### 3.4 SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC(gamma='scale')
clf.fit(X_train,y_train)


# In[ ]:


clf_pre=clf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,clf_pre))
print(classification_report(y_test,clf_pre))


# Out of all the classifier we checked Random forest classification turns out to have the maximum accuracy i.e 85% while SVM had the least i.e 63%. Hence we will use RF for prediction and generating our result.

# 1. ### 4. Generate the final solution.
# 

# In[ ]:


test_prediction = rf.predict(test)


# In[ ]:


test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')


# In[ ]:


new_test.head()


# In[ ]:


df= new_test[['PassengerId' ,'Survived']]
df.to_csv('predictions.csv' , index=False)


# ## Please vote if you like this notebook
# # THANK YOU :)

# In[ ]:




