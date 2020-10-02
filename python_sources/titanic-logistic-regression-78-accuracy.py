#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


# reading CSV
train_data = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


# shape of training data
train_data.shape


# In[ ]:


train_data.head()


# In[ ]:


# getting the type of each feature and number of non-null values
train_data.info()


# In[ ]:


# finding the missing values column
missing_values = 100 * train_data.isna().sum() / len(train_data)
missing_values[missing_values > 0]


# here we have missing values in 3 column <br/>
# Since missing values in Cabin is more than 70% we will be removing this column from our dataset <br/>
# For column Age we will be determining the missing values using the mean based on Gender ie. we group the data based on gender
# and will determing the mean of each gender and fill the null values <br/> <br/>
# 
# Embarked column is a categorical column we will be replacing the null value with mode of the column.<br />
# 

# In[ ]:


# dropping the feature Cabin
train_data = train_data.drop(['Cabin'], axis=1)


# In[ ]:


# lets calculate the missing value of column Embarked based on mode
train_data['Embarked'].value_counts()


# In[ ]:


# replacing the missing value of column Embarked with 'S' as occurence of S is more
train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[ ]:


# finding the mean age based on gender
train_data.groupby('Sex')['Age'].mean()


# mean age of gender male is 31 <br />
# mean age of female is 28

# In[ ]:


# for the age column we gonna replace the null values to mean of gender 
train_data['Age'] = train_data.groupby('Sex')['Age'].apply(lambda row: row.fillna(row.mean()))


# In[ ]:


# changing the Age column to int
train_data['Age'] = train_data['Age'].astype(int)


# In[ ]:


# lets retreive title from name column
train_data['Title'] = train_data['Name'].apply(lambda row: row.split(", ")[1].split(".")[0])


# In[ ]:


train_data['Title'].value_counts()


# In[ ]:


# let's combine the title Dr, Rev and other with value count < 10 to others
title = ['Mr', 'Miss', 'Mrs', 'Master']
def get_title(row):
    if row in title:
        return row
    else:
        return 'Others'

train_data['Title'] = train_data['Title'].apply(lambda row: get_title(row))
train_data['Title'].value_counts()


# In[ ]:


# lets see our changed dataframe
train_data.head()


# In[ ]:


# we can also make the bins for age columns
def get_age_binning(row):
    if row <= 18:
        return 'Children'
    elif row > 18 and row <= 26:
        return 'Youth'
    elif row > 26 and row <= 60:
        return 'Adult'
    elif row > 60:
        return 'Senior Citizen'
    else:
        ""

train_data['Age_Bins'] = train_data['Age'].apply(lambda row: get_age_binning(row))
train_data['Age_Bins'].value_counts()


# In[ ]:


# we can identify if the person was alone on ship or not by checking sibsp = 0 and sibsp = Parch
def is_alone(sibsp, parch):
    if parch == 0 and sibsp == parch:
        return "Yes"
    else:
        return "No"

train_data['Alone'] = train_data.apply(lambda row: is_alone(row.SibSp, row.Parch), axis=1)
train_data['Alone'].value_counts()


# In[ ]:


# lets view the distribution of column Fare
sns.distplot(train_data['Fare'])


# most fare value lies between 0-50

# In[ ]:


train_data.describe()


# In[ ]:


# lets create binning for fare
def get_fare_type(row):
    if row <=100:
        return "Low"
    elif row > 100 and row <=200:
        return "Mid"
    else:
        return "High"

train_data['Fare_Type'] = train_data['Fare'].apply(lambda row : get_fare_type(row))
train_data['Fare_Type'].value_counts()


# so very few people were there with high fare ticket

# In[ ]:


# we can get rid of below columns 
# PassengerId, Name, Age, Ticket, Fare
train_data = train_data.drop(['PassengerId', 'Name', 'Age', 'Fare', 'Ticket'], axis=1)


# In[ ]:


# changing the feature Pclass, SibSp, Parch to category  
train_data['Pclass'] = train_data['Pclass'].astype('category')
train_data['SibSp'] = train_data['SibSp'].astype('category')
train_data['Parch'] = train_data['Parch'].astype('category')


# In[ ]:


# lets view the final features datatype 
train_data.info()


# In[ ]:


# lets view our final dataframe 
train_data.head()


# ### lets perform some visualization

# <h3> a) univariate analysis </h3>

# In[ ]:


# count of people survived
sns.countplot(train_data.Survived)


# In[ ]:


# count of males and females
sns.countplot(train_data.Sex)


# In[ ]:


# count of people on ship based on fare_type
sns.countplot(train_data.Fare_Type)


# In[ ]:


# count of people on ship based on pclass
sns.countplot(train_data.Pclass)


# In[ ]:


# count of people on ship based on SibSp
sns.countplot(train_data.SibSp)


# In[ ]:


# count of people on ship based on Parch
sns.countplot(train_data.Parch)


# In[ ]:


sns.countplot(train_data.Alone)


# In[ ]:


sns.countplot(train_data.Age_Bins)


# <h3> b) bivariate analysis</h3>

# In[ ]:


sns.countplot(train_data.Sex, hue=train_data.Survived)


# number of females survived more as compared to male

# In[ ]:


sns.countplot(train_data.Fare_Type, hue=train_data.Survived)


# people who have taken high or mid fare ticket have more chances of survival

# In[ ]:


sns.countplot(train_data.Age_Bins, hue=train_data.Survived)


# In[ ]:


sns.countplot(train_data.Alone, hue=train_data.Survived)


# <h2>Columns encoding </h2>

# In[ ]:


train_data_copy = train_data.copy()


# In[ ]:


train_data_copy['Sex'] = train_data_copy['Sex'].apply(lambda row: 1 if row == 'Male' else 0)
train_data_copy['Alone'] = train_data_copy['Alone'].apply(lambda row: 1 if row == 'Yes' else 0)


# In[ ]:


cols_to_encode = ['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 'Fare_Type']
encoded_data = pd.get_dummies(train_data_copy[cols_to_encode])
encoded_data.head()


# In[ ]:


train_data_copy = pd.concat([train_data_copy, encoded_data], axis=1)
train_data_copy.columns


# In[ ]:


train_data_copy = train_data_copy.drop(['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 
                                                  'Fare_Type'], axis=1)


# In[ ]:


train_data_copy.head()


# In[ ]:


# lets find the correlation matrix 
plt.figure(figsize=(20,10))
sns.heatmap(train_data_copy.corr(), cmap='YlGnBu', annot=True)
plt.show()


# In[ ]:


X_train = train_data_copy.drop(['Survived'], axis=1)
Y_train = train_data_copy['Survived']


# ### choosing the best 8 features to create our model using RFE

# In[ ]:


log_reg_model = LogisticRegression()
rfe = RFE(log_reg_model, 8)
rfe = rfe.fit(X_train, Y_train)


# In[ ]:


cols = X_train.columns[rfe.support_]


# In[ ]:


# building the logistic regression model using the 8 best features selected using RFE
log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train[cols], Y_train)
y_pred = log_reg.predict(X_train[cols])


# In[ ]:


accuracy_score(Y_train, y_pred)


# In[ ]:


confusion_matrix(Y_train, y_pred)


# ### performing the EDA steps on the test data

# In[ ]:


# reading the test data
test_data = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


# getting the shape of test data
test_data.shape


# In[ ]:


# finding missing percentge for test data
missing_data = 100 * test_data.isna().sum() / len(test_data)
missing_data[missing_data > 0]


# We gonna get rid of Cabin feature as percentage of missing value is more than 70 <br>
# Fare missing value will be calculated using the mean of fare column <br/>
# For Age column we gonna consider mean age of gender

# In[ ]:


# finding mean age in test data
test_data.groupby('Sex')['Age'].mean()


# In[ ]:


# replacing the missing with mean value 
test_data['Age'] = test_data.groupby(['Sex'])['Age'].apply(lambda row: row.fillna(row.mean()))


# In[ ]:


test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# performing binning for the fare
test_data['Fare_Type'] = test_data['Fare'].apply(lambda row : get_fare_type(row))
test_data['Fare_Type'].value_counts()


# In[ ]:


# retrieving the title from the name
test_data['Title'] = test_data['Name'].apply(lambda row: row.split(", ")[1].split(".")[0])
test_data['Title'] = test_data['Title'].apply(lambda row: get_title(row))
test_data['Title'].value_counts()


# In[ ]:


# performing the binning for the age column
test_data['Age_Bins'] = test_data['Age'].apply(lambda row: get_age_binning(row))
test_data['Age_Bins'].value_counts()


# In[ ]:


# finding if the person was alone or not
test_data['Alone'] = test_data.apply(lambda row: is_alone(row.SibSp, row.Parch), axis=1)
test_data['Alone'].value_counts()


# In[ ]:


test_data['Pclass'] = test_data['Pclass'].astype('category')
test_data['SibSp'] = test_data['SibSp'].astype('category')
test_data['Parch'] = test_data['Parch'].astype('category')


# ### Encoding the test data

# In[ ]:


test_data['Sex'] = test_data['Sex'].apply(lambda row: 1 if row == 'Male' else 0)
test_data['Alone'] = test_data['Alone'].apply(lambda row: 1 if row == 'Yes' else 0)


# In[ ]:


test_data.head()


# In[ ]:


cols_to_encode = ['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 'Fare_Type']
test_encoded_data = pd.get_dummies(test_data[cols_to_encode])
test_encoded_data.head()


# In[ ]:


test_data = pd.concat([test_data, test_encoded_data], axis=1)


# In[ ]:


# saving the passender id to some dataframe as lateron we need to concat the passenger id and final predictions
passenger_id = test_data['PassengerId']


# In[ ]:


test_data = test_data.drop(['Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 
                              'Fare_Type', 'Title', 'Age_Bins', 'PassengerId'], axis=1)


# In[ ]:


y_test_pred = log_reg.predict(test_data[cols])


# In[ ]:


test_results = pd.concat([passenger_id, pd.DataFrame(y_test_pred)], axis=1)
test_results.columns = ['PassengerId','Survived']


# In[ ]:


test_results.to_csv('test_results_titanic.csv', index=None)

