#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:



train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


# In[ ]:


train["Survived"].shape


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# In[ ]:


print(train.shape)

print(test.shape)


# In[ ]:


#Checking null values present on each column so that we can remove the unnecessary column

train.isnull().sum()


# In[ ]:


#Checking null values present on each column so that we can remove the unnecessary column

test.isnull().sum()


# # Removing Columns that are not important

# In[ ]:


#Removing column "Cabin" since it has many null values

train.drop(['Cabin'], axis=1,inplace=True )

test.drop(['Cabin'], axis=1,inplace=True )


# # Finding out missing values of each column and filling them up
# 

# In[ ]:


#Finding out missing values of "Embarked" and then filling them up by S

train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace=True)

#test['Embarked'].value_counts()
#test['Embarked'].fillna('S', inplace=True)


# In[ ]:


#Finding out missing values of "Fair" and fiiling them up by mean values

#train['Fare'].fillna(train['Fare'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# # for column "Age" of train dataset

# In[ ]:


#Finding out missing values for "Age" of train
#Creating a variable called train_age and storing the random values from  mean and std

train_age=np.random.randint(train['Age'].mean() - train['Age'].std() , train['Age'].mean() + train['Age'].std(), 177)

train_age


# In[ ]:


#Checking null values present in age of train
train['Age'][train['Age'].isnull()]


# In[ ]:


#Replacing these null values by train_age
train['Age'][train['Age'].isnull()]=train_age


# In[ ]:


train.isnull().sum()


# # For column "Age" of test dataset

# In[ ]:


#Finding out missing values for "Age" of test 
#Creating a variable called test_age and storing the random values from  mean and std

test_age=np.random.randint(train['Age'].mean() - train['Age'].std() , train['Age'].mean() + train['Age'].std(), 86)

test_age


# In[ ]:


#Checking null values present in age of test
test['Age'][test['Age'].isnull()]


# In[ ]:


#Replacing these null values by train_age
test['Age'][test['Age'].isnull()]=test_age


# In[ ]:


test.isnull().sum()


# # Checking for each and every column of Train dataset if they are needed or not by checking the number of people survived or died

# In[ ]:


#for Pclass

train.groupby(['Pclass'])['Survived'].mean() #Therefore, Pclass can not be removed.


# In[ ]:


#For Sex

train.groupby(['Sex'])['Survived'].mean() #Therefore, Sex matters too


# In[ ]:


#For Embarked

train.groupby(['Embarked'])['Survived'].mean() #Therefore, Embarked matters too


# In[ ]:


#For Age since its numerical data so we are plotting the graph 

sns.distplot(train['Age'][train['Survived']==0])
sns.distplot(train['Age'][train['Survived']==1])

#Therefore, Age matters too


# In[ ]:


#For Fare

sns.distplot(train['Fare'][train['Survived']==0])
sns.distplot(train['Fare'][train['Survived']==1])

#Therefore, Fair matters too


# In[ ]:


#For Ticket, we have to remove it since it doesn't matter

train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)


# In[ ]:


test


# In[ ]:


#For "SibSp" and "Parch", we are going to add these two columns to a new column called "Family" for both train and test dataset

train['Family']=train['SibSp'] + train['Parch'] + 1
test['Family']=test['SibSp'] + test['Parch'] + 1

train['Family'].value_counts()
test['Family'].value_counts()


# In[ ]:


#For Family

train.groupby(['Family'])['Survived'].mean() #Therefore, Family matters too


# In[ ]:


#Creating a separate column for people travelling alone, with 2 or more than 2 & less than 4 and with more than 11

def cal1(number):
    if number==1:
        return"Alone"
    elif number>1 & number<5:
        return"Medium"
    else:
        return"Large"
    
train['Family_size']=train['Family'].apply(cal1)
train


# In[ ]:


test


# In[ ]:


#Creating a separate column for people travelling alone, with 2 or more than 2 & less than 4 and with more than 11

def cal2(number):
    if number==1:
        return"Alone"
    elif number>1 & number<5:
        return"Medium"
    else:
        return"Large"
    
test['Family_size']=test['Family'].apply(cal2)
test


# In[ ]:


#Removing columns like "SibSp", "Parch" and "Family"

train.drop(['SibSp','Parch', 'Family'], axis=1, inplace=True)
test.drop(['SibSp','Parch', 'Family'], axis=1, inplace=True)


# In[ ]:


#We need "PassengerId" for test so storing it in "passengerid"

passengerid=test['PassengerId'].values


# In[ ]:


#We don't need passenger id for training the dataset we only need it while testing so removing "PassengerId" and "Name"

train.drop(['PassengerId','Name'], axis=1, inplace=True)
test.drop(['PassengerId','Name'], axis=1, inplace=True)


# In[ ]:


train


# In[ ]:


test


# In[ ]:


#Converting categorical values into numerical values for train

train=pd.get_dummies(columns=['Pclass','Sex','Embarked','Family_size'], drop_first=True, data=train)
train


# In[ ]:


#Converting categorical values into numerical values for train

test=pd.get_dummies(columns=['Pclass','Sex','Embarked','Family_size'], drop_first=True, data=test)
test


# In[ ]:


train.shape


# In[ ]:


test.shape


# # Extracting X and Y for train

# In[ ]:


X=train.iloc[:,1:].values
print("The shape of X:",X.shape)

Y=train.iloc[:,0].values
print("The shape of Y:",Y.shape)


# # Applying Train and Test split

# In[ ]:


#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print("The shape of X_train:",X_train.shape)
print("The shape of Y_train:",Y_train.shape)
print(Y_test.shape)


# # Training the model by applying Decision Tree Classifier and passing X_train and Y_train to it.

# In[ ]:


classifier1=DecisionTreeClassifier() 
classifier1.fit(X_train, Y_train)


# # Predicting using classifier and finding Accuracy

# In[ ]:


#Predicting
Y_predict=classifier1.predict(X_test)
print(Y_predict.shape)

#Accuracy
AS1=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using decision tree classifier:", AS1)


# # Using Grid-Search-CV class and Training the model

# In[ ]:


# Using Grid-Search-CV class and Training the model
#Creating Variable
param_dist={"criterion":["gini","entropy"],
            "max_depth":[1,2,3,4,5,6,7,8,None],
            "max_features":[1,2,3,4,5,6,7,None],
            "random_state":[0,1,2,3,4,5,6,7,8,9,None],
            "max_leaf_nodes":[0,1,2,3,4,5,6,7,8,9,None],
            "max_features" : ["auto","sqrt","log2",None],
            "min_samples_leaf" : [1,2,3,4,5,6,7,8,None],
            "min_samples_split" : [1,2,3,4,5,6,7,8,None]}

#Applying Grid-Search-CV
grid=GridSearchCV(classifier1, param_grid=param_dist, cv=10, n_jobs=-1)

#Training the model after applying Grid-Search-CV
grid.fit(X_train,Y_train)


# # Finding Optimal Hyperparameter Value

# In[ ]:


OHV=grid.best_params_ 
print("The values of Optimal Hyperparameters are",OHV)


# # Calculating Accuracy

# In[ ]:


Acc=grid.best_score_
print("The Accuracy Score is",Acc)
print("Accuracy using DecisionTreeClassifier:", Acc*100,"%")


# In[ ]:


grid.best_estimator_


# In[ ]:


classifier2=DecisionTreeClassifier(criterion= 'gini', max_depth= 7, max_features= 'auto', max_leaf_nodes= None, min_samples_leaf= 8, min_samples_split= 2, random_state= 4)

classifier2.fit(X_train, Y_train)


# In[ ]:


# Predicting
Y_predict=classifier2.predict(X_test)
print(Y_predict.shape)

# Accuracy
AS2=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using decision tree classifier:", AS2)


# # Extracting all columns of test

# In[ ]:


X_test=test.iloc[:,:].values


# In[ ]:


Y_test=classifier2.predict(X_test)


# In[ ]:


Y_test.shape


# In[ ]:


passengerid.shape


# In[ ]:


#Creating an empty dataframe since passenger_id and Y_test have same number of rows
Final = pd.DataFrame()
Final


# In[ ]:


#Adding these 2 columns "passengerid" and "survived" then passing Y_test value in survived column

Final['passengerid'] = passengerid
Final['survived'] = Y_test
Final


# In[ ]:


#Converting it into csv file

Final.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




