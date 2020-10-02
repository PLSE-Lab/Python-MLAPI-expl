#!/usr/bin/env python
# coding: utf-8

# ### Titanic Disaster
# 

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Ignore warnings

# In[ ]:


import warnings

warnings.filterwarnings('ignore')


# ## 2. Import data

# In[ ]:


# Import train and test dataset

df_train = pd.read_csv('../input/titanic/train.csv')


df_test = pd.read_csv('../input/titanic/test.csv')


# ## 3. Exploratory Data Analysis

# ### Check the shape of dataframe

# In[ ]:


df_train.shape , df_test.shape


# We can see that there are 891 instances and 12 variables in the dataset.

# ### Preview the dataset

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### View summary of dataframe

# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# We can see that there are categorical and numerical variables in the dataset. Moreover several of the variables contain missing values. I will explore this later.
# 
# Now, I will view the summary statistics of numerical variables.

# ### View summary statistics of numerical variables

# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# ### Identify types of variables
# 
# 
# In this section, I will find out what types of variables there are in this dataset.

# In[ ]:


# check the type of variables in dataset

df_train.dtypes


# We can see that there are mixture of categorical and numerical variables in the dataset. Numerical are those of type int and float. Categorical those of type object.

# ### Find categorical variables
# 
# In this section, I will find the categorical variables.

# In[ ]:


# find categorical variables


categorical = [var for var in df_train.columns if df_train[var].dtype=='O']


print('There are {} categorical variables\n'.format(len(categorical)))


print('The categorical variables are :', categorical)


# In[ ]:


# view the categorical variables

df_train[categorical].head()


# - We can see that there 2 mixed type of variables: Cabin and Ticket
# 
# 
# - Cabin and Ticket variables contain both numbers and letters. I will extract the numerical part and then the non-numerical part and generate 2 variables out of them.
# 
# 

# ### Find numerical variables
# 
# In this section, I will find the numerical variables.

# In[ ]:


# find numerical variables

numerical = [var for var in df_train.columns if df_train[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[ ]:


# preview the numerical variables

df_train[numerical].head()


# - We can see that there are 3 discrete variables and 2 continuous variables in the dataset.
# 
# - The 3 discrete variables are - `Pclass`, `SibSp` and `Parch`.
# 
# - The 2 continuous variables are - `Fare` and `Age`.

# ### Summary - Types of variables
# 
# 
# - There are 5 categorical variables in the dataset. They are given by `Name`, `Sex`, `Ticket`, `Cabin` and `Embarked`.
# 
# - Out of the above 5 categorical variables, 2 are of mixed type variables - `Ticket` and `Cabin`
# 
# - There are 5 numerical variables in the dataset. They are given by `Pclass`, `Age`, `SibSp`, `Parch` and `Fare`.
# 
# - Out of the above 5 numerical variables - 3 are discrete variables and 2 are continuous variables in the dataset.
# 
# - The 3 discrete variables are - `Pclass`, `SibSp` and `Parch`.
# 
# - The 2 continuous variables are - `Fare` and `Age`.
# 

# ## 4. Data Exploration and Preprocessing 
# 
# 
# Now, I will explore the potential problems within variables. I will start by finding the missing values.
# 
# 
# ### Missing values in variables

# In[ ]:


# check missing values in variables in training data

df_train.isnull().sum()


# We can see that the `Age`, `Cabin` and `Embarked` variable contain missing values in training dataset. 

# In[ ]:


# check missing values in variables in test data

df_test.isnull().sum()


# In this case, We can see that the `Age`, `Fare` and `Cabin` variable contain missing values in test dataset. 

# ## Explore `Sex` variable

# In[ ]:


df_train['Sex'].value_counts()


# In[ ]:


df_test['Sex'].value_counts()


# The `Sex` variable are categorized into 2 categories -`Male` and `Female`. Males have relatively low probability of survival than females. 
# 
# Right ? - Wrong.
# 
# Male and female categories also have children. So, a male child have a relatively higher probability of survival than a male person. So, I will segregate the children from the passengers onboard at titanic.

# ### Feature extraction for child label

# In[ ]:


# label minors as child, and remaining people as female or male

def label_child(passenger):
    
    # take the age and sex
    age, sex = passenger
    
    # compare age, return child if under 16, otherwise leave sex
    if age < 16:
        return 'child'
    else:
        return sex


# In[ ]:


# create a new column `person` which specify the person as male, female or child

df_train['Person'] = df_train[['Age', 'Sex']].apply(label_child, axis = 1)

df_test['Person'] = df_test[['Age', 'Sex']].apply(label_child, axis = 1)


# In[ ]:


# check the distribution in `Person` variable in training data

df_train['Person'].value_counts()


# In[ ]:


# check the distribution in `Person` variable in test data

df_test['Person'].value_counts()


# So, I have created a new column `Person` which categorized the passengers as male, female and child.

# ## Explore `Pclass` variable

# In[ ]:


# print number of labels in Pclass variable

print('Pclass contains', len(df_train['Pclass'].unique()), 'labels')


# In[ ]:


# view labels in Pclass variable

df_train['Pclass'].unique()


# In[ ]:


# check frequency distribution of values in Pclass variable

df_train['Pclass'].value_counts()


# Now, I will segregate the Person by class.

# In[ ]:


# Person segregated by class in training set

sns.factorplot('Pclass', data = df_train, hue = 'Person', kind = 'count')


# ### Observations
# 
# 
# - We can see that there are large number of men travelling in Pclass 3. These men have lowest probability of survival amongst all combinations of `Person` and `Pclass`.
# 
# 
# - `Female` and `child` in Pclass 1 have highest probability of survival amongst all combinations of `Person` and `Pclass`.
# 
# 
# - The children in Pclass 3 have a higher probability of survival than men in Pclass 3.

# ## Explore `Age` variable

# In[ ]:


# distribution of age in training dataset

df_train['Age'].hist(bins=25, grid=False)


# In[ ]:


# distribution of age in test dataset


df_test['Age'].hist(bins=25, grid=False)


# We can see that `Age` is positively skewed. I will do some visualizations to explore more about the `Age` variable.

# In[ ]:


# age segregated by Person

fig = sns.FacetGrid(df_train, hue = 'Person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.add_legend()


# ### Observations
# 
# 
# - We can see that age is positively skewed in case of male, female and child. So, mean is not an appropriate measure for imputation in this case
# 
# 
# - Whenever we have positively skewed distribution, we should use median for missing values imputation. So, I will use median age to impute missing values.

# ### Imputation of `Age` variable
# 

# In[ ]:


# view the median age of people in training and test set

for df1 in [df_train, df_test]:
    print(df1.groupby('Person')['Age'].median())


# In[ ]:


# impute missing values with respective median values

for df1 in [df_train, df_test]:
    df1['Age'] = df1['Age'].fillna(df1.groupby('Person')['Age'].transform('median'))


# ### Check for missing values in `Age` variable

# In[ ]:


df_train['Age'].isnull().sum()        


# In[ ]:


df_test['Age'].isnull().sum()


# Now, we can see that there are no missing values in `Age` variable in training and test set.

# ## Explore `Cabin` variable

# In[ ]:


# print number of labels in Cabin variable

print('Cabin contains', len(df_train['Cabin'].unique()), 'labels in training set')

print('\nCabin contains', len(df_test['Cabin'].unique()), 'labels in test set')


# We have a problem here. There are large number of labels in `Cabin` variable in training and test set. 
# 
# To solve the problem, I will extract first letter of Deck from `Cabin`. The number of `Cabin` is irrelevant as the letter specifies the location in the boat.

# ### Extract first letter of Deck from the Cabin
# 
# 
# I will create a new variable `CabinLetter` that will extract first letter of `Cabin`.

# In[ ]:


df_train['CabinLetter'] = df_train['Cabin'].str.get(0)

df_test['CabinLetter'] = df_test['Cabin'].str.get(0)


# In[ ]:


# print number of labels in CabinLetter variable

print('CabinLetter contains', len(df_train['CabinLetter'].unique()), 'labels in training set\n')

print('CabinLetter contains', len(df_test['CabinLetter'].unique()), 'labels in test set')


# We can see that number of labels in `Cabin` training and test set have been significantly reduced from 148 to 9 and from 77 to 8.

# In[ ]:


# view labels in CabinLetter variable in training set

df_train['CabinLetter'].unique()


# In[ ]:


# view labels in CabinLetter variable in test set

df_test['CabinLetter'].unique()


# The above comparision shows that there is an extra label in `CabinLetter` variable in training set. So, I will take a look at the `Cabin` labels in training set. 

# In[ ]:


# view labels in Cabin variable in training set

df_train['Cabin'].unique()


# We can see that there is a label `T` in the `Cabin` variable in training set.

# ### Check for missing values in `CabinLetter` variable

# In[ ]:


df_train['CabinLetter'].isnull().sum()


# In[ ]:


df_test['CabinLetter'].isnull().sum()


# In[ ]:


sns.factorplot('CabinLetter', data = df_train, hue = 'Person', kind = 'count')


# In[ ]:


sns.factorplot('CabinLetter', data = df_test, hue = 'Person', kind = 'count')


# The above plots show number of people onboard in different types of cabin.

# I will impute missing values in `CabinLetter` by the most frequent values segregated by person.

# In[ ]:


# impute missing values in CabinLetter with respective mode values

for df1 in [df_train, df_test]:
    df1['CabinLetter'] = df1['CabinLetter'].fillna(df1['CabinLetter'].mode().iloc[0])


# ### Again check for missing values in `CabinLetter`

# In[ ]:


for df1 in [df_train, df_test]:
    print(df1['CabinLetter'].isnull().sum())


# Now, we can see that there are no missing values in the `CabinLetter` variable.

# ### Drop old variable `Cabin`

# In[ ]:


df_train.drop('Cabin', axis = 1, inplace = True)

df_test.drop('Cabin', axis = 1, inplace = True)


# ## Explore `Embarked` variable

# In[ ]:


# check distribution of `Embarked` variable in training set

df_train['Embarked'].value_counts()


# In[ ]:


# check distribution of `Embarked` variable in test set

df_test['Embarked'].value_counts()


# There are 3 ports of embarkation. They are `Cherbourg(C)`, `Queenstown(Q)` and `Southampton(S)`.

# In[ ]:


# where did people from different classes get on board

sns.factorplot('Embarked', data = df_train, hue= 'Pclass', kind = 'count')


# ### Check for missing values in `Embarked` variable

# In[ ]:


for df1 in [df_train, df_test]:
    print(df1['Embarked'].isnull().sum())


# We can see that there are 2 missing values in `Embarked` variable in training set and none in test set.

# ### Impute `Embarked` variable with the most frequent port (S)

# In[ ]:


df_train['Embarked'].fillna('S', inplace = True)


# ## Explore `Fare` variable

# ### Check for missing values in `Fare` variable

# In[ ]:


for df1 in [df_train, df_test]:
    print(df1['Fare'].isnull().sum())


# We can see that there is a null value in `Fare` variable in test set. I will replace it with the mean fare in test set.

# In[ ]:


df_test['Fare'].fillna(df_test.Fare.mean(), inplace =True)


# ## Check for missing values in the training and test data set

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# We can see that all the missing values have been removed from the dataset.

# ### Explore whether a passenger was with family or not

# In[ ]:


for df1 in [df_train, df_test]:
    df1['Familyman'] = df1.Parch + df1.SibSp
    df1['Familyman'].loc[df1['Familyman'] > 0] = 'Yes'
    df1['Familyman'].loc[df1['Familyman'] == 0] = 'No'


# In[ ]:


# check the frequency distribution of `Familyman` variable

for df1 in [df_train, df_test]:
    print(df1['Familyman'].value_counts())


# The above analysis shows that most people were travelling alone with no family. So, they have relatively low orobability of survival.

# ### Check whether a man is travelling with spouse

# In[ ]:


def man_with_spouse(passenger):
    sex, sibsp = passenger
    if sex == 'male' and sibsp > 0:
        return 1
    else:
        return 0


# In[ ]:


# create a new variable `man_and_spouse` to check whether a man is travelling with spouse

for df1 in [df_train, df_test]:
    df1['man_and_spouse'] = df1[['Sex', 'SibSp']].apply(man_with_spouse, axis = 1)


# ### Check whether a person is a mother travelling with children

# In[ ]:


def woman_with_child(passenger):
    age, sex, parch = passenger
    if age > 20 and sex == 'female' and parch > 0:
        return 1
    else:
        return 0


# In[ ]:


# create a new variable `is_mother` to check whether a woman is travelling with child

for df1 in [df_train, df_test]:
    df1['is_mother'] = df1[['Age', 'Sex', 'Parch']].apply(woman_with_child, axis = 1)


# ## Preview the dataset again

# In[ ]:


# Preview the train dataset again

df_train.head()


# In[ ]:


# preview the test dataset again

df_test.head()


# ## 5. Data Preprocessing

# ### Remove unimportant features - `Ticket` , `Name` and `PassengerId`
# 

# ### Drop `PassengerId` variable
# 
# 
# `PassengerId` is a unique identifier for each passenger. So, it is not a variable that contributes towards the predictive power of the model. Hence, I will remove this variable from the dataset. 

# In[ ]:


# drop PassengerId variable 

for df1 in [df_train, df_test]:
    df1.drop('PassengerId', axis=1, inplace=True)


# ### Drop `Ticket` variable
# 
# 
# `Ticket` variable contain too many labels. So, it doesn't have much predictive power. I will remove it from the dataset.

# In[ ]:


# drop Ticket variable 

for df1 in [df_train, df_test]:
    df1.drop('Ticket', axis=1, inplace=True)


# ### Drop `Name` variable

# In[ ]:


# drop Name variable 

for df1 in [df_train, df_test]:
    df1.drop('Name', axis=1, inplace=True)


# In[ ]:





# ## 6. Declare feature vector and target variable

# In[ ]:


X_train = df_train.drop(['Survived'], axis=1)

y_train = df_train.Survived

X_test = df_test


# ## 7. Feature engineering

# ### Encode categorical variables
# 
# 
# Now, I will encode the categorical variables.

# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# We can see that the categorical variables that are required to be encoded are `Sex`, `Embarked`, `Person`, `CabinLetter` and `Familyman`.

# ### encode sex variable

# In[ ]:


# encode sex variable

for df1 in [X_train, X_test]:
    df1['Sex']  = pd.get_dummies(df1.Sex, drop_first=True)


# In[ ]:


X_train.Sex.unique()


# In[ ]:


X_test.Sex.unique()


# ### encode remaining categorical variable with categorical encoder

# In[ ]:


# import category encoders

import category_encoders as ce


# In[ ]:


# encode categorical variables with ordinal encoding

encoder = ce.OneHotEncoder(cols=['Embarked', 'Person', 'CabinLetter', 'Familyman'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# ## 9. Feature scaling

# In[ ]:


X_train.head()


# In[ ]:


cols = X_train.columns


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[ ]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[ ]:


X_test = pd.DataFrame(X_test, columns=[cols])


# We now have `X_train` dataset ready to be fed into a classifier. 
# 

# ## 10. Machine Learning algorithm building
# 

# ### XGBoost Classifier

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb_final = XGBClassifier()


# In[ ]:


xgb_final.fit(X_train, y_train)


# In[ ]:


y_pred = xgb_final.predict(X_test)


# In[ ]:


test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


submission = pd.DataFrame({
                        "PassengerId": test_df['PassengerId'],
                        "Survived": y_pred
                          })


# In[ ]:


submission.head()


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook

filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:




