#!/usr/bin/env python
# coding: utf-8

# # Importing required Libraries and datasets

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train.head()


# # EDA and Data Preparation

# In[ ]:


## Checking the dataset for completeness

train.info()

## Remove Cabin
## Substitute missing Age
## Remove column with missing Embarked


# In[ ]:


## Creating the first two characters of ticket as a variable

train['Ticket_First'] = train['Ticket'].apply(lambda x:x.replace('.','').replace('/','').split()[0][:2])


# In[ ]:


## Checking the survival frequency across Ticket_First variable

ticket_freq = train[['Ticket_First','Survived']].groupby(['Ticket_First']).agg([('Nos people', 'count'), ('Nos survived', 'sum')])
ticket_freq.columns = ticket_freq.columns.get_level_values(1)
ticket_freq = ticket_freq.reset_index(level = [0])
ticket_freq['Survival %'] = round(ticket_freq['Nos survived']*100/ticket_freq['Nos people'])
ticket_freq.sort_values(by = ['Nos people'], ascending = False)

## It does seem like there are too many variables with too little observations to reliably decide survival. So, grouping the 
## Ticket_First where # observations =< 10


# In[ ]:


## Grouping the minor Ticket_First as others

def Ticket_Grp(col):
    
    if col[0] in ticket_freq[ticket_freq['Nos people'] > 10]['Ticket_First'].to_list():
        return col[0]
    else:
        return 'Others'


# In[ ]:


train['Ticket_Grp'] = train[['Ticket_First']].apply(Ticket_Grp, axis =1)
train['Ticket_Grp'].value_counts()


# In[ ]:


## Extracting the salutation from name as a variable

train['Salute'] = train['Name'].apply(lambda x:x.split()[1])


# In[ ]:


pd.value_counts(train['Salute']).head()


# In[ ]:


## Grouping the minor salutations as others

def Salute_group(col):
    
    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:
        return col[0]
    else:
        return 'Others'


# In[ ]:


train['Salute_Grp'] = train[['Salute']].apply(Salute_group, axis =1)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Salute_Grp', data = train, hue = 'Survived')


# In[ ]:


##Missing Values

sns.heatmap(train.isnull())


# In[ ]:


# Treat Age


# In[ ]:



# Option4
#train[pd.isnull(train['Age'])]

sns.boxplot (x='Sex', y='Age', data = train, hue = 'Pclass')

# Age is not a huge differentiator here but going ahead to see difference in the model results


# In[ ]:



# Calculating medians

PclassXSex_med = train[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()


# In[ ]:


# Defining a function to impute median (using median since the data is skewed) for each PclassXSex.

## MUCH MORE EFFICIENT WAY TO WRITING FUNCTION THAN BEFORE ##

def age_PclassSex(cols):
    age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(age) == True:
        return PclassXSex_med.loc[Sex].loc[Pclass][0]
    else:
        return age


# In[ ]:


train['Age_PclXSex'] = train[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)


# In[ ]:


# Removing the unneeded and NA-dominated columns

train.drop(['Age', 'Cabin'], axis =1 , inplace = True)


# In[ ]:


# Drop the na rows

train.dropna(inplace = True)


# In[ ]:


# Check if all the null values are gone

sns.heatmap(pd.isnull(train))


# In[ ]:



## Now creating dummy variables for Sex and Embarked


Sex_Dumm = pd.get_dummies(train['Sex'], drop_first = True)
Embarked_Dumm = pd.get_dummies(train['Embarked'], drop_first = True)
Ticket_Grp = pd.get_dummies(train['Ticket_Grp'], drop_first = True, prefix = 'Ticket')
Salute_Group = pd.get_dummies(train['Salute_Grp'], drop_first = True)


# In[ ]:


train = pd.concat([train, Sex_Dumm, Embarked_Dumm, Ticket_Grp, Salute_Group], axis = 1)
train.head()


# In[ ]:


train.info()


# # Creating base random forest model

# In[ ]:


train.columns


# In[ ]:


## Creating test train dataset from 'train' dataframe only as we don't have the 'y' for test.

from sklearn.model_selection import train_test_split

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'SibSp', 'Parch', 'Fare',
       'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_13', 'Ticket_17',
       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',
       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',
       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',
       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',
        'Miss.', 'Mr.', 'Mrs.', 'Others']], y, test_size = 0.3, random_state = 143)



# In[ ]:


## Fitting into the base RF model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)


# In[ ]:


pred = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(accuracy_score(y_test, pred))


# In[ ]:


## Trying to tune the hyperparameters (the parameters which can be decided by us and not by the model)

## Using Cross Validation for the same

from sklearn.model_selection import RandomizedSearchCV


# Define the parameter sets
# int(x) for x in ... converts the array into list. 

n_estimators = [int(x) for x in np.arange(200, 2200, 200)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.arange(10,110,10)]
max_depth.append(None)
min_samples_leaf = [1,2,3,4,5]
min_samples_split = [2,4,6,8,10]
bootstrap = [True,False]

param_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_leaf' : min_samples_leaf,
              'min_samples_split' : min_samples_split,
              'bootstrap' : bootstrap}


rf1 = RandomForestClassifier()
rf1_random = RandomizedSearchCV(rf1, param_grid, n_iter = 100, cv = 3, verbose =2 , random_state = 143, n_jobs = -1)


# In[ ]:


rf1_random.fit(X_train, y_train)


# In[ ]:


rf1_random.best_params_


# In[ ]:


rf1_random.best_estimator_


# In[ ]:


pred2 = rf1_random.best_estimator_.predict(X_test)


# In[ ]:


print(accuracy_score(y_test, pred2))


# In[ ]:


### No change in accuracy after trying to tune hyperparamters (-_-). 


# # Preparing test data 

# In[ ]:


## Prepare the test dataset in the same way

test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test.info()


# In[ ]:


## Creating the first two characters of ticket as a variable

test['Ticket_First'] = test['Ticket'].apply(lambda x:x.replace('.','').replace('/','').split()[0][:2])


# In[ ]:


## Grouping the minor Ticket_First as others

def Ticket_Grp2(col):
    
    if col[0] in ['A5', 'PC', 'ST', '11', '37', '33', '17', '34', '23',
       '35', '24', '26', '19', 'CA', 'SC', '31', '29', '36', 'SO', '25',
       '28', '13']:
        return col[0]
    else:
        return 'Others'


# In[ ]:


test['Ticket_Grp'] = test[['Ticket_First']].apply(Ticket_Grp2, axis =1)


# In[ ]:


test['Salute'] = test['Name'].apply(lambda x:x.split()[1])


# In[ ]:


test['Salute'] = test['Name'].apply(lambda x:x.split()[1])
def Salute_group(col):
    
    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:
        return col[0]
    else:
        return 'Others'


# In[ ]:


test['Salute_Grp'] = test[['Salute']].apply(Salute_group, axis =1)


# In[ ]:



# Calculating medians

PclassXSex_med = test[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()
PclassXSex_med


# In[ ]:


test['Age_PclXSex'] = test[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)


# In[ ]:


# Removing the unneeded and NA-dominated columns

test.drop(['Cabin', 'Age'], axis =1 , inplace = True)


# In[ ]:


# Substituting the missing value of fare using mean fare of that "Passenger_class X Sex X Embarked" group

test[pd.isnull(test['Fare'])]


# In[ ]:


Fare_med = test[['Pclass','Fare','Sex', 'Embarked']].groupby(['Pclass','Sex', 'Embarked']).agg(['count', 'mean'])

Fare_med


# In[ ]:


test['Fare'].fillna(12.718, inplace = True)


# In[ ]:


test.info()


# In[ ]:


## Now creating dummy variables for Sex and Embarked


Sex_Dumm = pd.get_dummies(test['Sex'], drop_first = True)
Embarked_Dumm = pd.get_dummies(test['Embarked'], drop_first = True)
Ticket_Grp = pd.get_dummies(test['Ticket_Grp'], drop_first = True, prefix = 'Ticket')
Salute_Group = pd.get_dummies(test['Salute_Grp'], drop_first = True)


# In[ ]:


test = pd.concat([test, Sex_Dumm, Embarked_Dumm, Ticket_Grp, Salute_Group], axis = 1)
test.head()


# # Building final model

# In[ ]:


# Now using all the train dataset to fit the model and then predicting the test data

X = train[['Pclass', 'SibSp', 'Parch', 'Fare',
       'Age_PclXSex', 'male', 'Q', 'S',  'Ticket_13', 'Ticket_17',
       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',
       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',
       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',
       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',
        'Miss.', 'Mr.', 'Mrs.', 'Others']]
y = train['Survived']


# In[ ]:


## Fit the model with the best set of parameters

rf_fin = rf1_random.best_estimator_

rf_fin.fit(X,y)


# In[ ]:


test.set_index('PassengerId', inplace = True)


# In[ ]:


test_fin =test[['Pclass', 'SibSp', 'Parch', 'Fare',
       'Age_PclXSex', 'male', 'Q', 'S',  'Ticket_13', 'Ticket_17',
       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',
       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',
       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',
       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',
        'Miss.', 'Mr.', 'Mrs.', 'Others']]

test_fin


# In[ ]:


pred_fin = rf_fin.predict(test_fin)


pred_df = pd.DataFrame(pred_fin, columns = ['Survived'],index = test_fin.index)
pred_df


# In[ ]:


# Output Result
pred_df['Survived'].to_csv('My_Titanic_Predictions.csv', index = True, header = True)


# # Checking feature importance

# In[ ]:


importance = rf_fin.feature_importances_


# In[ ]:


feat_imp = pd.DataFrame(importance, index = X.columns, columns = ['Importance'])

feat_imp.sort_values(['Importance'], inplace = True)


# In[ ]:


plt.figure(figsize = (10,10))
sns.barplot(x=feat_imp['Importance'], y = feat_imp.index, data = feat_imp, palette = 'coolwarm')


# In[ ]:




