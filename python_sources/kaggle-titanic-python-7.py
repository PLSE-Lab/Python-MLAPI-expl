# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:12:58 2017

@author: vicarizmendi
"""
# In this challenge, we ask you to complete the analysis of what sorts of people
#  were likely to survive. In particular, we ask you to apply the tools of 
# machine learning to predict which passengers survived the tragedy.

# Version 7 - Folloging example given by ZlatanKremonic in Kaggle Kernels
# Instead of using GridSearchCV, we use 

import numpy as np
import pandas as pd
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline


df_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},na_values=[""] )
df_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},na_values=[""] )

# Best way of checking out a dataframe
print(df_train.info())
print(df_train.describe())

#print(df_train.shape) # (891, 11)
#print(df_train.dtypes)

print(df_train.head())



# Survived 

#So we can see that 62% of the people in the training set died. This is slightly
# less than the estimated 67% that died in the actual shipwreck (1500/2224).

print(df_train['Survived'].value_counts(normalize=True))
plt.figure() # Creates a new figure
sns.countplot(df_train['Survived'])


# Pclass

#critical role in survival

print(df_train['Survived'].groupby(df_train['Pclass']).mean())
plt.figure() # Creates a new figure
sns.countplot(df_train['Pclass'], hue=df_train['Survived'])


# Name

# We can obtain meaningful information from the name by extracting the title:
# Mrs., Miss, Mr, Dr....
print(df_train['Name'].head())
df_train['Name_Title'] = df_train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
print(df_train['Name_Title'].value_counts())

print(df_train['Survived'].groupby(df_train['Name_Title']).mean())

# the survival rate appears to be either significantly above or below the 
# average survival rate, which should help our model.

# 'Name_len' is divided in 5 groups with the method pd.qcut, and then the 
# mean is calculated grouped by these groups
# Quantile-based discretization function pd.qcut:
    
df_train['Name_Len'] = df_train['Name'].apply(lambda x: len(x))
print(df_train['Survived'].groupby(pd.qcut(df_train['Name_Len'],5)).mean())



# Sex

print(df_train['Sex'].value_counts(normalize=True))
print(df_train['Survived'].groupby(df_train['Sex']).mean())


# Age

# There are 177 nulls for Age, and they have a 10% lower survival rate than 
# the non-nulls. Before imputing values for the nulls, we will include an 
# Age_null flag just to make sure we can account for this characteristic 
# of the data.

print(df_train['Age'].isnull().sum())  # 177, for those lets see the survival rate
print(df_train['Survived'].groupby(df_train['Age'].isnull()).mean())

print(df_train['Survived'].groupby(pd.qcut(df_train['Age'],5)).mean())
print(pd.qcut(df_train['Age'],5).value_counts())


# SibSp  - not of much interest from the numbers below

print(df_train['Survived'].groupby(df_train['SibSp']).mean())
print(df_train['SibSp'].value_counts())


# Parch  - not of much interest from the numbers below

print(df_train['Survived'].groupby(df_train['Parch']).mean())
print(df_train['Parch'].value_counts())


# Ticket

# at a first glance this alphanumeric feature seems not to have predictive value
# But as there are different types of tickets lets try to see if by using the length 
# of the string will give the

df_train['Ticket_Len'] = df_train['Ticket'].apply(lambda x: len(x))
print(df_train['Ticket_Len'].value_counts())
print(df_train['Survived'].groupby(df_train['Ticket_Len']).mean())

# And try to get some more information from the first letter of the ticket

df_train['Ticket_Lett'] = df_train['Ticket'].apply(lambda x: str(x)[0])
print(df_train['Ticket_Lett'].value_counts())
print(df_train['Survived'].groupby(df_train['Ticket_Lett']).mean())



# Fare

# There is a clear relationship between Fare and Survived, and I'm guessing that
# this relationship is similar to that of Class and Survived.

# Divide the fares in 3 quantiles for comparison purposes. We see the survival
# rate is higher for the higher fares
print(pd.qcut(df_train['Fare'], 3).value_counts())
print(df_train['Survived'].groupby(pd.qcut(df_train['Fare'], 3)).mean())

# relationship between Fare and Pclass

print(pd.crosstab(pd.qcut(df_train['Fare'], 5), columns=df_train['Pclass']))


# Cabin


# There are many nulls but we can try to extract information from the ones non empty
print(df_train["Cabin"].unique())

# New feature - Cabin_Letter
# Extrac only the first letter to see if there is any dependency 
df_train["Cabin_Letter"] = df_train["Cabin"].apply(lambda x:str(x)[0])
print(df_train["Cabin_Letter"].unique())
print(df_train['Cabin_Letter'].value_counts())
print(df_train['Cabin_Letter'].value_counts().sum())
print(df_train["Survived"].groupby(df_train["Cabin_Letter"]).mean())

# New Feature - Cabin Number, eliminating the letter
# This doesn't seem to have any relationship with survival 
df_train['Cabin_num'] = df_train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
print(df_train['Cabin_num'].head(20))
print(df_train['Cabin'].head(20))

print(df_train['Cabin_num'].value_counts())
print(df_train["Survived"].groupby(df_train["Cabin_num"]!="an").mean())

df_train['Cabin_num'].replace('an', np.NaN, inplace = True)
df_train['Cabin_num'] = df_train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

#As a whole the passengers with cabin have higher survival rate than the rest
print(pd.qcut(df_train['Cabin_num'],3).value_counts())
print(df_train['Survived'].groupby(pd.qcut(df_train['Cabin_num'], 3)).mean())


# No correlation between s"survived" and "Cabin_num"
print(df_train['Survived'].corr(df_train['Cabin_num']))

# Embarked

# Apparently the passengers embarking by C have higher survival rate than the rest
print(df_train['Embarked'].value_counts())
print(df_train['Embarked'].value_counts(normalize=True))
# Even though the proportion of passengers from C is much less than the S
print(df_train["Survived"].groupby(df_train['Embarked']).mean())

# high presence of upper-class passengers from that location C
plt.figure() # Creates a new figure
sns.countplot(df_train['Embarked'], hue=df_train['Pclass'])




# Feature Engineering

# Two new columns, one with the Name_Len and the othe with the Name_Title
def names(train, test):  
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test

# Substitute the empty Age fields in train and in test subsets by the mean Age
# of non null ages in training data, groupped by "Name_Title" and "Pclass", 
# instead of doing it with the general mean 
# This more granular approach to imputation should be more accurate than merely
# taking the mean age of the population
    
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test

# Combine SibSp and Parch into a new variable family size and group in three 
# categories

def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test
   
# Create from Ticket two new columns Ticket_Lett (first letter) and Ticket_Len
# with the smaller n values grouped based on survival rate ???? Don't know 
# how the letters are chosen based on the survival rate NO

# ['1', '2', '3', 'S', 'P', 'C', 'A'] are the ones with more number of passengers -> "Ticket letter"
# ['W', '4', '7', '6', 'L', '5', '8'] have less than 14 passengers and more than one -> "Low Ticket"
# '9' -> "Other Ticket"

def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['Worder(', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test
print(df_train['Ticket_Lett'].value_counts())
print(df_train['Survived'].groupby(df_train['Ticket_Lett']).mean().sort_values())


# The following two functions extract the first letter of the Cabin column and 
# its number, respectively.

def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test
     
def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test

# We fill the null values in the Embarked column with the most commonly occuring value, which is 'S.'

def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test
                                      
                                       
# Next, because we are using scikit-learn, we must convert our categorical columns
# into dummy variables. The following function does this, and then it drops the
# original categorical columns. It also makes sure that each category is present
# in both the training and test datasets.

def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                    'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

# Drop the PassengerId feature, not needed in this approach
def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test

# Having built our helper functions, we can now execute them in order to build 
# our dataset that will be used in the model  

df_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},na_values=[""] )
df_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},na_values=[""] )
  
# fill in the one missing value of Fare in our test set with the mean 
# value of Fare from the training set (transformations of test set data must 
# always be fit using training data).
df_test['Fare'].fillna(df_train['Fare'].mean(), inplace = True)
 
df_train, df_test = names(df_train, df_test)
df_train, df_test = age_impute(df_train, df_test)
df_train, df_test = cabin_num(df_train, df_test)
df_train, df_test = cabin(df_train, df_test)
df_train, df_test = embarked_impute(df_train, df_test)
df_train, df_test = fam_size(df_train, df_test)
df_test['Fare'].fillna(df_train['Fare'].mean(), inplace = True)
df_train, df_test = ticket_grouped(df_train, df_test)
df_train, df_test = dummies(df_train, df_test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])


df_train, df_test = drop(df_train, df_test)

print(len(df_train.columns)) # 45 columns

# Hyperparameter Tuning

#from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=1000,
                             min_samples_split=6,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(df_train.iloc[:, 1:], df_train.iloc[:, 0])
print("%.4f" % rf.oob_score_)

#param_grid = { "criterion" : ["gini", "entropy"], 
# "min_samples_leaf" : [1, 5, 10],
# "min_samples_split" : [2, 4, 10, 12, 16],
# "n_estimators": [50, 100, 400, 700, 1000]}

#param_dist = {"min_samples_split": sp_randint(2, 16),
#              "min_samples_leaf": sp_randint(1, 11),
#              "criterion": ["gini", "entropy"],
#              "n_estimators": [50, 100, 400, 700, 1000]}

#gs = RandomizedSearchCV(estimator=rf, param_distributions = param_dist, scoring='accuracy', cv=3, n_jobs=-1)
#gs = gs.fit(df_train.iloc[:, 1:], df_train.iloc[:, 0])
#print(gs.cv_results_)
#print(gs.best_estimator_)
#print("%.4f" % gs.best_score_)



#Looking at the results of the grid search:
#{'min_samples_split': 10, 'n_estimators': 700, 'criterion': 'gini', 'min_samples_leaf': 1}
#0.838383838384

#looking at the results of the Randomized Search:
#{'min_samples_split': 6, 'n_estimators': 1000, 'criterion': 'entropy', 'min_samples_leaf': 1}
# 0.8350

predictions = rf.predict(df_test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},na_values=[""] )
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('y_test_7.csv', sep=",", index = False)



