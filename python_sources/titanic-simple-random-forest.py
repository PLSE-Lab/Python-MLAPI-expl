#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# * [Load Data and Libraries](#load)
# * [Check Data](#check)
# * [Data Pre-Processing](#data-process)
# * [Data Visualizations](#visualization)
# * [Training and Estimation](#training)

# # Load Data and Libraries <a id='load'></a>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load train and test data
train = pd.read_csv("../input/titanic/train.csv")
test  = pd.read_csv("../input/titanic/test.csv")

# All data
data  = train.append(test, sort=False)


# # Check Data <a id="check"></a>

# There're 12 variables in dataset

# In[ ]:


# Columns
print(len(data.columns))
data.columns


# Check data example and types

# In[ ]:


# Example
data.sample(n=10)


# In[ ]:


types = pd.DataFrame(data.dtypes).rename(columns={0: 'type'}).sort_values(by=['type'],ascending=False)
types


# For data pre-processing, categorize variables
# ```
# # Categorical String variables
# Embarked        object
# Cabin           object
# Ticket          object
# Sex             object
# Name            object
# 
# # Categorical Integer variables (have order)
# Parch            int64
# SibSp            int64
# Pclass           int64
# 
# # Numerical variables
# Fare           float64
# Age            float64
# ```

# Check missing rate

# In[ ]:


# Check missing values
def check_missing(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    missing_table = pd.concat([null_val, percent], axis=1)
    col = missing_table.rename(columns = {0 : 'Num', 1 : 'Rate'})
    return col

# Display columns missing values are under 1%.
print("Data #"+str(len(data)))
cols = check_missing(data)

types.join(cols).sort_values(by="Rate", ascending=False)


# # Data Pre-Processing <a id="data-process"></a>

# Pre-process data by their variable types, Categorical, Numerical, Integer, String.  
# And we now process training and test set together. (data = training + test)  
# (because we can process values in same scale)

# ```
# - Categorical String variables
# Col         Type      Num     Rate
# Cabin       object    1014    77.463713
# Embarked    object    2       0.152788
# Name		object    0       0.000000
# Sex         object    0       0.000000
# Ticket      object    0       0.000000
# ```
# * **Cabin**
#   * Just drop this variable, because 77.4% of values aremissing.
# * **Embarked**
#   * Fill missing values as mode value.(S)
#   * Map strings into integers: S->0, C->1, Q->2
#   * One-Hot Encoding
# * **Name**
#   * Map strings into integers: Mr,Master->1, Dr,Don,Major,etc->2, Rev->3, Ms,Miss,Mrs,Mme->4
#   * One-Hot Encoding
# * **Sex**
#   * Map strings into integers: male->0, female->1
#   * One-Hot Encoding
# * **Ticket**
#   * Leave as it is.

# In[ ]:


# Drop Cabin
data.drop(['Cabin'], axis=1, inplace = True)

# "Embarked": Fill NA and map into integer
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Embarked"] = data["Embarked"].map({"S": 0, "C" : 1, "Q" : 2})

# "Name": Map passenger's name with their title
title_mapping    = {
    '(.+)Mr\.(.+)': 1, '(.+)Master\.(.+)': 1,
    '(.+)Dr\.(.+)': 2, '(.+)Don\.(.+)': 2, '(.+)Major\.(.+)': 2,
    '(.+)Sir\.(.+)':2, '(.+)Col\.(.+)': 2, '(.+)Jonkheer\.(.+)': 2,
    '(.+)Capt\.(.+)': 2,'(.+)Countess\.(.+)': 2, '(.+)Dona\.(.+)': 2,
    '(.+)Rev\.(.+)': 3,
    '(.+)Ms\.(.+)': 4, '(.*)Miss\.(.+)': 4, '(.+)Mrs\.(.+)': 4,
    '(.+)Mme\.(.+)': 4,'(.+)Lady\.(.+)': 4, '(.+)Mlle\.(.+)': 4 
}
data["Title"] = data["Name"].replace(title_mapping, regex=True).astype('int')

# "Sex": Map male and female
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})


# ```
# - Categorical Integer variables (have order)
# Col     Type    Num   Rate
# Pclass  int64   0     0.000000
# SibSp   int64   0     0.000000
# ```
# * **Ticket**
#   * Leave as it is.
# * **Ticket**
#   * Leave as it is.

# In[ ]:


# Do nothing.


# ```
# - Numerical variables
# Col     Type    Num   Rate
# Age     float64 263   20.091673
# Fare    float64 1     0.076394
# ```
# * **Age**
#   * Fill missing values from median of same name title (Mr,Mrs,etc)
#   * Standardize values
#   * Divide into 10 categories
# * **Fare**
#   * Fill missing values from median of same Pclass
#   * Standardize values
#   * Divide into 10 categories

# In[ ]:


# Estimate missing age from title
for i in range(1, 5):
    age_to_estimate = data.groupby('Title')['Age'].median()[i]
    data.loc[(data['Age'].isnull()) & (data['Title'] == i), 'Age'] = age_to_estimate

# Estimate missing fare from pclass
for i in range(1, 4):
    fare_to_estimate = data.groupby('Pclass')['Fare'].median()[i]
    data.loc[(data['Fare'].isnull()) & (data['Pclass'] == i), 'Fare'] = fare_to_estimate

# Standardize numerical values
ss = StandardScaler()
ss.fit_transform(data[['Age', 'Fare']])

# Cut Age into 10 categolies
data["AgeBin"]  = pd.qcut(data["Age"], 10, duplicates="drop", labels=False)

# Cut Fare into 10 categolies
data["FareBin"] = pd.qcut(data["Fare"], 10, duplicates="drop", labels=False)


# ```
# - Generate new variable information from other variables
# FamirySize
# IsFamily
# FamilySurvival
# ```
# * **FamilySize**
#   * Size of Family: Parch + SibSp
# * **IsFamily**
#   * 0 families->0
#   * 1 families->1
#   * 2 or more families->2
#   * One-Hot Encoding
# * **FamilySurvival**
#   * Regard ones have same lastname or same ticketname as family
#   * One of their family survived -> 1
#   * All of their family dead -> 0
#   * No data -> 0.5
#   * One-Hot Encoding

# In[ ]:


# Add FamilySize
data['FamilySize'] = data["Parch"] + data["SibSp"]

# Add IsFamily
data['IsFamily'] = data["Parch"] + data["SibSp"]
data.loc[data['IsFamily'] > 1, 'IsFamily']  = 2
data.loc[data['IsFamily'] == 1, 'IsFamily'] = 1
data.loc[data['IsFamily'] == 0, 'IsFamily'] = 0

# Add FamilySurvival
DEFAULT_SURVIVAL_VALUE = 0.5
data['FamilySurvival'] = DEFAULT_SURVIVAL_VALUE

# Get last name
data['LastName'] = data['Name'].apply(lambda x: str.split(x, ",")[0])

# Family has same Lastname and Fare
for grp, grp_df in data.groupby(['LastName', 'Fare']):
    if(len(grp_df) != 1):
        # A Family group is found
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'FamilySurvival'] = 1
            elif (smin==0.0):
                data.loc[data['PassengerId'] == passID, 'FamilySurvival'] = 0

# Family(or group) has same Ticket No
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['FamilySurvival'] == 0) | (row['FamilySurvival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'FamilySurvival'] = 1
                elif (smin==0.0):
                    data.loc[data['PassengerId'] == passID, 'FamilySurvival'] = 0


# Drop useless data and keep as Training and Test set.

# In[ ]:


# Drop Useless data
train_target = data[:891]["Survived"].values
data.drop(['Name', 'PassengerId', 'Age', 'Fare', 'Ticket', 'LastName'], axis = 1, inplace = True)

# One-Hot Encoding Categorical variables
data = pd.get_dummies(data, columns=["Embarked", "Title", "Sex", "IsFamily", "FamilySurvival"], drop_first=True)

# Set data
train = data[:891]
test  = data[891:]

# Data types
data.dtypes


# # Data Visualizations <a id="visualization"></a>

# In[ ]:


# Plot correlations between variables
plt.figure(figsize=(20, 10))
sns.heatmap(train.corr(), annot=True, fmt='.2f')


# Display feature importances

# In[ ]:


possible_features = train.columns.copy().drop('Survived')

# Check feature importances
selector = SelectKBest(f_classif, len(possible_features))
selector.fit(train[possible_features], train_target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Feature importances:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], possible_features[indices[i]]))


# Display all possible features with Survival

# In[ ]:


# Display all possible features with Survival
fig, axs = plt.subplots(8, 2, figsize=(20, 30))
for i in range(0, 16):
    sns.countplot(possible_features[i], data=train, hue=train_target, ax=axs[i%8, i//8])


# # Training and Estimation <a id="training"></a>

# This time, pick variables by their importances
# ```
# Feature importances:
# 68.85 Sex_1
# 67.32 Title_4
# 32.23 FamilySurvival_1.0
# 24.60 Pclass
# 22.45 FareBin
# 7.05 FamilySurvival_0.5
# 6.36 Embarked_1
# 6.01 IsFamily_1
# 2.11 IsFamily_2
# 1.83 Parch
# 1.28 Title_3
# 0.92 AgeBin
# 0.53 SibSp
# 0.21 FamilySize
# 0.18 Title_2
# 0.04 Embarked_2
# ```

# In[ ]:


# Feature params
fparams =     ['Sex_1', 'Title_4', 'FamilySurvival_1.0', 'Pclass', 'FareBin', 
     'FamilySurvival_0.5', 'Embarked_1', 'IsFamily_1', 'IsFamily_2', 
     'Parch', 'Title_3', 'AgeBin', 'SibSp']

# Get params
train_features = train[fparams].values
test_features  = test[fparams].values

# Number of Cross Validation Split
CV_SPLIT_NUM = 6

# Params for RandomForestClassifier
rfgs_parameters = {
    'n_estimators': [300],
    'max_depth'   : [2,3,4],
    'max_features': [2,3,4],
    "min_samples_split": [2,3,4],
    "min_samples_leaf":  [2,3,4]
}

rfc_cv = GridSearchCV(RandomForestClassifier(), rfgs_parameters, cv=CV_SPLIT_NUM)
rfc_cv.fit(train_features, train_target)
print("RFC GridSearch score: "+str(rfc_cv.best_score_))
print("RFC GridSearch params: ")
print(rfc_cv.best_params_)


# In[ ]:


# Predict and output to csv
survived = rfc_cv.best_estimator_.predict(test_features)
pred = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])
pred['Survived'] = survived.astype(int)
pred.to_csv("../working/submission.csv", index = False)

