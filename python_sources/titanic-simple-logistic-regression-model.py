#!/usr/bin/env python
# coding: utf-8

# # Simple Logistic Regression Approach

# ### Notebook Contents:
# 
# - Loading libraries and datasets
#     1. Exploring the data variables
# - Exploratory Data analysis
#     1. Exploring the spread and center of variables
#     2. Exploring the survival of passengers
# - Feature Engineering
#     1. Imputing missing variables
#     2. Converting Categorical variables
# - Training Model and predictions
#     1. Logistic Regression
#     
# ### To do:
#  - Improve the models

# ### Import Libraries

# In[ ]:


import numpy as np    # linear algebra
import pandas as pd   # data processing/feature engineering
import matplotlib.pyplot as plt       # Data visualization
import seaborn as sns                 # Enhanced Data Visualization

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression # Logistic Regression Model


# ### Load Data

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# ### Shape, head, Center

# #### Training data
# 

# In[ ]:


train_df.head(2)


# In[ ]:


train_df.describe()


# In[ ]:


train_df.info()


# #### Test Data

# In[ ]:


test_df.head(2)


# In[ ]:


test_df.describe()


# In[ ]:


test_df.info()


# The training data and test data have some missing values in several columns. Let's explore them further and some other properties/relations of variables with some visual plots.

# ## Exploratory Data Analysis

# ### Missing data

# In[ ]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# ### How many passengers survived ?

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(train_df['Survived'], palette='RdBu_r')


# ### Actual count of passengers survival

# In[ ]:


survived = train_df[train_df['Survived']==1]['Survived'].sum()
survived


# ### How many males and females amongst survivals ?

# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['Sex'], palette='rainbow')


# ### Survival according to the class

# In[ ]:


sns.countplot(train_df['Survived'], hue=train_df['Pclass'], palette='rainbow')


# ### Overall Age distribution of passengers on Titanic

# In[ ]:


train_df['Age'].hist(color='darkred', bins=30, alpha=0.6)


# ### Fare distribution among the passengers

# In[ ]:


train_df['Fare'].hist(color='purple', bins=30, figsize=(8,4))


# ### Age distribution according to class

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x=train_df['Pclass'], y=train_df['Age'], palette='winter')


# ### Fare distribution according to class

# In[ ]:


sns.boxplot(x=train_df['Pclass'], y=train_df['Fare'])


# ## Feature Engineering

# ### Fill Missing Values

# In[ ]:


train_df.isnull().sum()


# - 'Age' shall be imputed with mean values according to class since it is a numerical variable. 
# - 'Cabin' shall be removed as it is not really required.
# - 'Embarked' shall be imputed with mode values as it is a categorical variable. 

# #### Mean Age per class

# In[ ]:


meanAge = train_df.groupby('Pclass').mean()['Age']
meanAge


# In[ ]:


# Defining a function for calculating mean age
def imputeAge(cols):
    Age = cols[0]
    Class = cols[1]
    
    if pd.isnull(Age):
        
        if Class == 1:
            return meanAge[1]
        elif Class == 2:
            return meanAge[2]
        else:
            return meanAge[3]
    else:
        return Age        


# In[ ]:


# Applying above function in Age column
train_df['Age'] = train_df[['Age', 'Pclass']].apply(imputeAge, axis=1)


# #### Drop the 'Cabin' variable

# In[ ]:


train_df.drop('Cabin', axis=1, inplace=True)


# #### Impute 'Embarked' variable

# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])


# In[ ]:


train_df.isnull().sum()


# ### Test Data Imputations

# In[ ]:


test_df.isnull().sum()


# - 'Age' variable shall be imputed with mean values according to class.
# - 'Fare' variable shall be imputed with mean value according to class.
# - 'Cabin' and all variables which were removed from training data shall be dropped from test data as well.

# #### Mean Age according to class (Test data)

# In[ ]:


meanAge_test = test_df.groupby('Pclass').mean()['Age']
meanAge_test


# In[ ]:


def imputAge_test(cols):
    Age = cols[0]
    Class = cols[1]
    
    if pd.isnull(Age):
        if Class == 1:
            return meanAge_test[1]
        elif Class == 2:
            return meanAge_test[2]
        else:
            return meanAge_test[3]
    else:
        return Age


# In[ ]:


test_df['Age'] = test_df[['Age', 'Pclass']].apply(imputAge_test, axis=1)


# ### Fare imputation (Test data)

# In[ ]:


meanFare_test = test_df.groupby('Pclass').mean()['Fare']
meanFare_test


# In[ ]:


# Check the number of missing values
test_df['Fare'].isnull().sum()


# #### Check the class of passenger with missing 'Fare'

# In[ ]:


test_df[test_df['Fare'].isnull() == True]['Pclass']


# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(meanFare_test[3])


# #### Drop the 'Cabin' Variable

# In[ ]:


test_df.drop('Cabin',axis=1, inplace=True)


# In[ ]:


test_df.isnull().sum()


# ### Converting the Categorical Variables

# - Creating dummy variables for 'Sex' and 'Embarked'.
# - 'PassengerId' , 'Name', 'ticket' shall be dropped.

# In[ ]:


sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embark = pd.get_dummies(train_df['Embarked'], drop_first=True)


# In[ ]:


train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace=True)


# In[ ]:


train_df = pd.concat([train_df, sex, embark], axis=1)


# In[ ]:


train_df.head(2)


# ### Test data conversions

# In[ ]:


sex = pd.get_dummies(test_df['Sex'], drop_first=True)
embark = pd.get_dummies(test_df['Embarked'], drop_first=True)


# In[ ]:


test_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis= 1, inplace=True)


# In[ ]:


test_df.head(2)


# In[ ]:


test_df = pd.concat([test_df, sex, embark], axis =1)


# In[ ]:


test_df.head(2)


# ### Model preprocessing

# In[ ]:


X_train = train_df.drop(['Survived','PassengerId'], axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1)

X_train.shape, y_train.shape, X_test.shape


# # Model Building and Training

# ### Creating an instance of logistic model

# In[ ]:


logmodel = LogisticRegression(max_iter=150)


# #### Fitting the model on training data

# In[ ]:


logmodel.fit(X_train, y_train)


# ### Predicting Survival in test data

# In[ ]:


predictions = logmodel.predict(X_test)


# ### File Submission

# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index = False)
submission.head()

