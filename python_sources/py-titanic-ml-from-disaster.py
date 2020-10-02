
# Python Data Analysis Library
import pandas as pd 

# Library for Scientific Computing
import numpy as np

# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import RandomForestClassifier

# Read CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# ---------------------------- #
#       Data Analysis          #
# ---------------------------- #

# AGE -------------------------------------------------------
# Discretize train['Age'] into 5 equal-sized buckets
# There are 177 nulls for Age we can populate the missing values with mean age.
train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()

# Pclass -------------------------------------------------------
# Pclass - overall
# 1    0.242424
# 2    0.206510
# 3    0.551066

# Pclass - survived
# 1    0.629630
# 2    0.472826
# 3    0.242363

# Pclass 3 has 55% of population and but only 24% of them are survived.
train['Survived'].groupby(train['Pclass']).mean()

# SEX -------------------------------------------------------
# Sex - overall
# male      0.647587
# female    0.352413

# Sex - survived
# female    0.742038
# male      0.188908

# Females to have a higher survival rate than males.
train['Survived'].groupby(train['Sex']).mean()

# NAME -------------------------------------------------------
# we create a Name_Title column to understand how the titles may affect chances of survivability
train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Survived'].groupby(train['Name_Title']).mean()

# Similarly, we create a Name_Len column to understand how long and 
# short names may affect chances of survivability
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()
# Name_Len
# [12, 19]    0.220588
# (19, 23]    0.301282
# (23, 27]    0.319797
# (27, 32]    0.442424
# (32, 82]    0.674556
# Clearly we see that long names had more chances of survival

# SibSp -------------------------------------------------------
train['Survived'].groupby(train['SibSp']).mean()

# Parch -------------------------------------------------------
train['Survived'].groupby(train['Parch']).mean()

# Ticket -------------------------------------------------------
# The Ticket column is an alphanumeric data
# Similar to what we did in Name, we can create ticket_length, ticket_letter
# try to find if length of ticket has an impact in survival
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
train.groupby(['Ticket_Lett'])['Survived'].mean()

# Fare -------------------------------------------------------
train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean()
# [0, 8.662]       0.198052
# (8.662, 26]      0.402778
# (26, 512.329]    0.559322
# Here chance of survival is higher for people who paid higher fare
# this also indicate, Pclass 1 has higher survival than Pclass 3

# Cabin -------------------------------------------------------
# This column has the most nulls (almost 700), but we can still extract information from it, like the first letter of each cabin, or the cabin number. The usefulness of this column might be similar to that of the Ticket variable.

# Cabin Letter 
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train['Survived'].groupby(train['Cabin_Letter']).mean()

# Cabin Number 
train['Cabin_num'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
train['Cabin_num'].replace('an', np.NaN, inplace = True)
train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

train['Survived'].groupby(pd.qcut(train['Cabin_num'], 3)).mean()
train['Survived'].corr(train['Cabin_num'])

# Embarked -------------------------------------------------------
# Embarked - overall
# C    0.188976
# Q    0.086614
# S    0.724409

# Embarked - Survivied
# C    0.553571
# Q    0.389610
# S    0.336957

# Embarked: 'C' has more chance of survival
train['Survived'].groupby(train['Embarked']).mean()

# ---------------------------- #
#  Functions for Data Cleanup  #
# ---------------------------- #

# Function to create two separate columns: a numeric column indicating the length of a passenger's Name field, and a categorical column that extracts the passenger's title.
def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test

# Function to populate age column with mean value.
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test

# We combine the SibSp and Parch columns into family size, and group the family size variable into three categories individual, small and big.
def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Individual',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Small', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test

# Using Ticket column we create two new columns: Ticket_Lett and Ticket_Len.
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test

# functions extract the first letter of the Cabin column and its number.
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

# We fill the null values with 'S' in the Embarked column as 'S' is more common
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test
 
# categorical columns into dummy variables.
def dummies(train, test, columns):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

# Drop the PassengerId column
def drop(train, test, droppedCols = ['PassengerId']):
    for i in [train, test]:
        for col in droppedCols:
            del i[col]
    return train, test

# Call our helper functions to clean the data for our model   
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']
train, test = dummies(train, test, columns)
train, test = drop(train, test)

# ------------------------------------------------------- #
# ------------------ Model Estimation  ------------------ #
# ------------------------------------------------------- #
rf = RandomForestClassifier(n_estimators=100,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:10]

# 12	Sex_female	    0.111215
# 11	Sex_male	    0.109769
# 33	Name_Title_Mr.	0.109746
# 1	    Fare	        0.088209
# 2	    Name_Len	    0.087904
# 0	    Age	            0.078651
# 8	    Pclass_3	    0.043268

# predict the target variable for our test data and generate an output file.
predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv('../input/test.csv')
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('result.csv', sep=",", index = False)


