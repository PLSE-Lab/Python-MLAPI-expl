#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

# machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[ ]:


# load train & test data from csv files 
train_data = pd.read_csv("C:/Users/Lenovo/Desktop/train.csv")
test_data= pd.read_csv("C:/Users/Lenovo/Desktop/test.csv")


# # 1. Exploratory Data Analysis (EDA)

# In[ ]:


# preview train data
train_data.head()


# In[ ]:


# preview test data
test_data.head()


# In[ ]:


print(train_data.shape,test_data.shape)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.columns


# In[ ]:


test_data.columns


# In[ ]:


# combining the train and test data
combine_data = pd.concat((train_data,test_data),axis = 0).reset_index(drop=True)
combine_data.info()


# In[ ]:


combine_data.head()


# In[ ]:


# Checking Categorical Data
train_data.select_dtypes(include=['object']).columns


# In[ ]:


# Checking Numerical Data
train_data.select_dtypes(include=['int64','float64']).columns 


# In[ ]:


combine_data.Sex.value_counts()


# In[ ]:


combine_data.Pclass.value_counts()


# In[ ]:


combine_data.Survived.value_counts()


# # Analyze by visualizing variables with bar plot, histogram and scatter plot

# In[ ]:


# Visualize variables with bar plot
combine_data.Sex.value_counts().plot(kind='bar')


# In[ ]:


combine_data.Survived.value_counts().plot(kind='bar')


# In[ ]:


combine_data.Pclass.value_counts().plot(kind='bar')


# In[ ]:


combine_data.Parch.value_counts().plot(kind='bar')


# In[ ]:


combine_data.Embarked.value_counts().plot(kind='bar')


# In[ ]:


# visualize variables with histogram
combine_data.Age.plot(kind ='hist',title = 'histogram for Age' )


# In[ ]:


combine_data.Parch.plot(kind ='hist',title = 'histogram for Parch' )


# In[ ]:


combine_data.Pclass.plot(kind ='hist',title = 'histogram for Pclass' )


# In[ ]:


combine_data.Survived.plot(kind ='hist',title = 'histogram for Survived' )


# In[ ]:


# visulize variables with scatter plot
combine_data.plot.scatter(x='Age',y='Fare',title='Age vs Fare')


# In[ ]:


combine_data.plot.scatter(x='Age',y='Survived',title='Age vs Survived')


# In[ ]:


combine_data.plot.scatter(x='Age',y='Pclass',title='Age vs Pclass')


# # Analyze by pivoting features
# - we can  analyze our feature correlations by pivoting features.
# 

# In[ ]:


combine_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


combine_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


combine_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


combine_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # correlation matrix

# In[ ]:


#plotting the correlation matrix of all Numerical features
plt.figure(figsize=(17,15))
sns.heatmap(combine_data.corr(), annot=True, cmap='cubehelix_r',square=True) 
plt.show()


# In[ ]:


combine_data_corr = combine_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
combine_data_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
combine_data_corr[combine_data_corr['Feature 1'] == 'Age']


# In[ ]:


combine_data[combine_data.Age.isnull()].head()


# # Dealing with  missing values in  variables

# In[ ]:


# finding missing values

Train_data_missing = train_data.isnull().sum()
print("training data:",sep="Train_data_missing")
print(Train_data_missing)
print('\n')
Test_data_missing = test_data.isnull().sum()
print("testing data:",sep="Test_data_missing")
print(Test_data_missing)
print('\n')
Combine_data_missing = combine_data.isnull().sum()
print("combining data:",sep="Combine_data_missing")
print(Combine_data_missing)


# #  Age 
# - we are filling missing values using median method.Median age of Pclass groups is the best choice because
# of its high correlation with Age. for better accuracy we can group with 'Sex' variable
# 

# In[ ]:


# Filling  missing values in Age column with the medians of Sex and Pclass groups
combine_data['Age'] = combine_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
combine_data['Age'].head(10)


# In[ ]:


combine_data['Age'].isnull().sum()


#  # Fare
#  -  only one passenger who has missed Fare value.   
#     - 

# In[ ]:


combine_data[combine_data.Fare.isnull()]


# In[ ]:


med_fare = combine_data.loc[(combine_data.Pclass == 3) & (combine_data.Embarked=='S'),'Fare'].median()
print (med_fare)
# Filling  missing values in Fare column with the medians Fare
combine_data['Fare'] = combine_data['Fare'].fillna(med_fare)


# In[ ]:


combine_data['Fare'].isnull().sum()


# # Embarked 
# -  Embarked is a categorical variable and only 2 missing values in the whole dataset.

# In[ ]:


combine_data[combine_data.Embarked.isnull()]


# In[ ]:


# number of people embarked at a particular points
combine_data.Embarked.value_counts()


# In[ ]:


pd.crosstab(combine_data[combine_data.Survived != -888].Survived,combine_data[combine_data.Survived != -888].Embarked)


# In[ ]:


# Filling missing values in Embarked columns with S
combine_data['Embarked'] = combine_data['Embarked'].fillna('S')


# In[ ]:


combine_data['Embarked'].isnull().sum()


# # Cabin

# In[ ]:


combine_data["Cabin"].head()


# In[ ]:


combine_data["Cabin"].describe()


# In[ ]:


combine_data["Cabin"].isnull().sum()


# In[ ]:


# Replace the missing Cabin number by  'X' if not
combine_data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in combine_data['Cabin'] ])


# In[ ]:


g = sns.countplot(combine_data["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[ ]:


g = sns.factorplot(y="Survived",x="Cabin",data=combine_data,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# In[ ]:


# Dropping the Cabin feature because of 80% missing values
combine_data.drop(['Cabin'], inplace=True, axis=1)


# In[ ]:


combine_data.head()


# # 2. Feature Engineering

# **Feature: Age state(child or adult)**

# In[ ]:


# we can convert age in the two category Adult(if age>18) otherwise Child
combine_data['AgeState'] = np.where(combine_data['Age']>=18,'Adult','Child')
combine_data['AgeState'].value_counts()


# In[ ]:


pd.crosstab(combine_data[combine_data.Survived != -88].Survived , combine_data[combine_data.Survived != -88].AgeState)


# **Feature:Title from Name**

# In[ ]:


#  convert name variable in Title
combine_data_title = [i.split(",")[1].split(".")[0].strip() for i in combine_data['Name']]
combine_data['Title'] = pd.Series(combine_data_title)
combine_data['Title'].head()


# 

# In[ ]:


g = sns.countplot(x="Title",data=combine_data)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[ ]:


# We can replace many titles with a more common name such as Rare
combine_data['Title'] = combine_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combine_data['Title'] = combine_data['Title'].replace('Mlle', 'Miss')
combine_data['Title'] = combine_data['Title'].replace('Ms', 'Miss')
combine_data['Title'] = combine_data['Title'].replace('Mme', 'Mrs')
    
combine_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels(["Master","Miss","Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# In[ ]:


# we can convert the categorical titles to integer.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
combine_data['Title'] = combine_data['Title'].map(title_mapping)
combine_data['Title'] = combine_data['Title'].fillna(0)

combine_data.head()


# In[ ]:


# we need to drop Name feature from data and also no need to PassengerID
combine_data = combine_data.drop(['Name'], axis=1)


# In[ ]:


combine_data.head()


# **Feature:Sex**

# In[ ]:


g = sns.factorplot(x="Sex",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels(["Male","Female"])
g = g.set_ylabels("survival probability")


# In[ ]:


# convert Sex feature into categorical value 0 for male and 1 for female
combine_data["Sex"] = combine_data["Sex"].map({"male": 0, "female":1})


# In[ ]:


combine_data.head()


# In[ ]:





# In[ ]:


# we can also drop Ticket variable
combine_data = combine_data.drop(['Ticket'], axis=1)


# In[ ]:


combine_data.head()


# In[ ]:


# we can create Age bands accourding to our choice
combine_data.loc[ combine_data['Age'] < 18, 'Age'] = 0
combine_data.loc[(combine_data['Age'] >= 18) & (combine_data['Age'] <36), 'Age'] = 1
combine_data.loc[(combine_data['Age'] >= 36) & (combine_data['Age'] < 54), 'Age'] = 2
combine_data.loc[(combine_data['Age'] >= 54) & (combine_data['Age'] < 72), 'Age'] = 3
combine_data.loc[ combine_data['Age'] >= 72, 'Age']
combine_data.head()


# **Embarked**

# In[ ]:


g = sns.factorplot(x="Embarked",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels(["S","Q","C"])
g = g.set_ylabels("survival probability")


# In[ ]:


# we can convert categorical feature to integer
embarked_mapping = {"S": 0 , "C": 1, "Q": 2 }
combine_data['Embarked'] = combine_data['Embarked'].map(embarked_mapping)


# In[ ]:


combine_data.head()


# **AgeState**

# In[ ]:


g = sns.factorplot(x="AgeState",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels(["Child","Adult"])
g = g.set_ylabels("survival probability")


# In[ ]:


# we can convert categorical feature to integer
agestate_mapping = {'Child': 1, 'Adult': 2}
combine_data['AgeState'] =  combine_data['AgeState'].map(agestate_mapping)


# In[ ]:


combine_data.head()


# In[ ]:


# create Fare Bands
combine_data.loc[ combine_data['Fare'] <= 7.91, 'Fare'] = 0
combine_data.loc[(combine_data['Fare'] > 7.91) & (combine_data['Fare'] <= 14.454), 'Fare'] = 1
combine_data.loc[(combine_data['Fare'] > 14.454) & (combine_data['Fare'] <= 31), 'Fare']   = 2
combine_data.loc[ combine_data['Fare'] > 31, 'Fare'] = 3
# combine_data['Fare'] = combine_data['Fare'].astype(int)


# In[ ]:


combine_data.head()


# **Family Size**

# In[ ]:


#We can create a new feature from combination of  Parch and SibSp.

combine_data['FamilySize'] = combine_data['SibSp'] + combine_data['Parch'] + 1

combine_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


g = sns.factorplot(x="FamilySize",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels([1, 2,3,4,5,6,7])
g = g.set_ylabels("survival probability")


# In[ ]:


# we can create another feature IsAlone from FamilySize
combine_data['IsAlone'] = 0
combine_data.loc[combine_data['FamilySize'] == 1, 'IsAlone'] = 1
combine_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


g = sns.factorplot(x="IsAlone",y="Survived",data=combine_data,kind="bar")
g = g.set_xticklabels(["0","1"])
g = g.set_ylabels("survival probability")


# In[ ]:


combine_data.head()


# In[ ]:


# we need to drop Parch, SibSp, and FamilySize features in favor of IsAlone.
combine_data = combine_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)


# In[ ]:


combine_data.head()


# In[ ]:


combine_data.info()


# In[ ]:


Combine_data_missing = combine_data.isnull().sum()
print("combining data:",sep="Combine_data_missing")
print(Combine_data_missing)


# # 3. Machine Learning Models
# - we split combine data in train and test data. we need to remove response variable(Survived) from test data

# In[ ]:


train_data = combine_data.loc[0:890,:]
train_data = train_data.astype(int)
train_data.head()


# In[ ]:


# Separate train data features 
train_data = train_data.drop(['PassengerId'],axis=1)
X_train = train_data.drop(labels = ['Survived'],axis = 1)
Y_train = train_data['Survived']


# In[ ]:


print(train_data.shape)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


# we drop Survived feature from test data
test_data = combine_data.loc[891:,:]
test_data = test_data.drop(['Survived'],axis=1)
test_data = test_data.astype(int)
test_data.head()


# In[ ]:


X_test = test_data.drop(['PassengerId'],axis=1)
X_test.head()


# In[ ]:


X_test.isnull().sum()


# In[ ]:


# Logistic Regression

log_reg = LogisticRegression()
scores_log_reg = cross_val_score(log_reg, X_train, Y_train, scoring='accuracy',cv=10)
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
acc_log_reg = np.mean(scores_log_reg)
acc_log_reg


# In[ ]:


print(Y_pred)


# In[ ]:


# Random Forests

random_forest=RandomForestClassifier(random_state =1 )
scores_rf = cross_val_score(random_forest, X_train, Y_train, scoring='accuracy',cv=10)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = np.mean(scores_rf)
acc_random_forest


# 

# In[ ]:


# Support Vector Machines

svc = SVC()
scores_svc = cross_val_score(svc, X_train, Y_train, scoring='accuracy',cv=10)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = np.mean(scores_svc)
acc_svc


# 

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
scores_decision_tree = cross_val_score(decision_tree, X_train, Y_train, scoring='accuracy',cv=10)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = np.mean(scores_decision_tree)
acc_decision_tree


# # Model evaluation
# - We can now rank our evaluation of all the models to choose the best one for our problem. 
#   we have used K-coss fold validation technique to estimate the accuracy and reduce overfitting of the models due to over training.
# 
# - we choose to use Random Forest because due to overfitting in decision tree to their training set.
#   support vector machine &  Randome Forest model to good preditionfor this problem.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','Random Forest','Suport Vector Machine','Decision tree'],
    'Score': [acc_log_reg, acc_random_forest, acc_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# # Prediction

# In[ ]:


submission_data = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_data['PassengerId'] = test_data['PassengerId']
submission_data['Survived'] = Y_pred.astype(int)
submission_data.to_csv('Titanic_pred_submissions.csv', header=True, index=False)
submission_data.head(10)

