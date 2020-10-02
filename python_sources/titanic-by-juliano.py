#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival  
# ## Competition by Kaggle
# ### Juliano Garcia

# #### Competition Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## Step 1: Define the problem!

# As per the competition description we need to develop an algorithm to predict the survival of a list of passengers

# ## Step 2: Import data and basic understanding

# In[ ]:


# Importing data wrangling and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# importing the train dataset
df = pd.read_csv('../input/train.csv')

# importing the test dataset
df_pred = pd.read_csv('../input/test.csv')

# And let's check how the data looks like!
df.sample(10)


# In[ ]:


# Let's check the datatypes
df.info()


# Some observations of the data:
# - The ___Survived___ attribute is the target variable. Binary, 1 for Yes (Survived) and 0 for No (Not Survived)
# - ___PassangerId___ and _Ticket_ seems to be random, and won't have any relation to the outcome whatsoever
# - The ___Pclass___ is a ordinal for the ticket class, proxy for a socialeconomic class (1 - Upper, 2 - Mid and 3 - Lower)
# - ___Name___ could be used with feature engineering to give some details on gender, status, family size, etc
# - ___Sex___ and ___Embarked___ are nominal and therefore should be converted to dummy variables for our predictive model
# - ___Age___ and ___Fare___ are continuous quantitative data
# - ___SibSp___ represents the number of sibilings or spouse on board, while ___Parch___ represents the number of parents and children on board, could be used to create family size and a "is alone" dummy
# - ___Cabin___ is a nominal variable, might be useful to determine the location on the ship, however it appears we have a lot of missing values (204 values out of 891 rows)! Therefore should be excluded from the model in order to not create any bias

# Now let us create a copy of the dataset for messing arround it, and also a referencing variable so we can perform operations on both the train and test dataframes

# In[ ]:


# Copy dataframe
df1 = df.iloc[:, :]
df_pred1 = df_pred.iloc[:, :]

# Create a list of both dataframes
both = [df1, df_pred1]


# ## Step 3: Data Cleaning

# 1. Correct aberant values and outliers
# 2. Complete missing information
# 3. Create new features
# 4. Casting variables to correct format

# ### 1. Correct aberant values and outliers

# It does not appear we have any aberant values, however we might have some outliers on the ___Age___ and ___Fare___ attributes, both since we didn't see any absurd values on the .describe() method we will waint until we finish our exploratory analysis to tackle it

# ### 2. Complete missing information

# Let's check the count of null values

# In[ ]:


# For the Train Set
df1.isnull().sum()


# In[ ]:


# and the Test Set
df_pred1.isnull().sum()


# - ___Age___ we will complete with the mediam
# - ___Fare___ we will complete with the mediam
# - ___Cabin___ we will exclude from the dataframe
# - ___Embarked___ we will replace with the mode

# In[ ]:


for dataset in both:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

# Let's delete the cabin attribute and also PassengerId and Ticket, already stated above
df1.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)


# In[ ]:


# Let's check the null values again
print(df1.isnull().sum())
print('-'*20)
print(df_pred1.isnull().sum())


# ### 3. Create new features

# Let us create some new features:
# - Family Size
# - Is Alone
# - Title
# - Fare Bin
# - Age Bin

# In[ ]:


for dataset in both:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = dataset['FamilySize'] == 1
    dataset['IsAlone'] = dataset['IsAlone'].astype('int')
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['FareBin'] = pd.cut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype('int'), 5)


# In[ ]:


df1.info()


# In[ ]:


df1.sample(10)


# Le's see how the Title attribute is distributed

# In[ ]:


df1['Title'].value_counts()


# For statistical purpose, let's eliminate the titles with fewer than __10__ entries

# In[ ]:


# Selects the titles to delete in both datasets
title_del = (pd.concat([df1, df_pred1], sort=False)['Title'].value_counts() < 10)

# Replace them with "Other"
for dataset in both:
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Other' if title_del.loc[x] == True else x)

# Let's see how it looks like
df1['Title'].value_counts()


# ### 4. Correcting formats

# Let's convert the categorical data to dummies for our mathematical analysis

# In[ ]:


# Import the library
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[ ]:


# Crete the encoder object
encoder = LabelEncoder()

to_encode = [('Sex', 'Sex_Coded'), ('Embarked', 'Embarked_Coded'), ('Title', 'Title_Coded'), 
             ('FareBin', 'Fare_Coded'), ('AgeBin', 'Age_Coded')]

# Fit and transform using the Train set and transform the test set
for dataset in both:
    for a in to_encode:
        dataset[a[1]] = encoder.fit_transform(dataset[a[0]])


# In[ ]:


df1.sample(5)


# Now let's create our dummies!

# In[ ]:


df1_dummy = pd.get_dummies(df1[['Sex', 'Embarked', 'Title']])
df_pred1_dummy = pd.get_dummies(df_pred1[['Sex', 'Embarked', 'Title']])
df1_dummy.head()


# We will get back to it latter!

# ## Exploratory Data Analysis (EDA)

# In[ ]:


# List the features to be used on the analysis
features = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

# Set the target
target = ['Survived']


# In[ ]:


# Let's check the Survived mean for each non continuous attribute
for feature in features:
    if df1[feature].dtype != 'float64' :
        print(df1[[feature, target[0]]].groupby(feature, as_index=False).mean())
        print('-'*20, '\n')


# In[ ]:


# Let's get visual for the continuous attributes
plt.figure(figsize=[16,12])

# Plot Fare as boxplot to identify outliers
plt.subplot(231)
plt.boxplot(x=df1['Fare'])
plt.title('Fare')
plt.ylabel('Fare ($)')

# Also let's plot Age in a boxplot
plt.subplot(232)
plt.boxplot(df1['Age'])
plt.title('Age')
plt.ylabel('Age (Years)')

# How about family size?
plt.subplot(233)
plt.boxplot(df1['FamilySize'])
plt.title('Family Size')
plt.ylabel('Family Size (#)')

# Now how would Fare affect survivability?
plt.subplot(234)
plt.hist(x = [df1[df1['Survived']==1]['Fare'], df1[df1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

# Age and survivability
plt.subplot(235)
plt.hist(x = [df1[df1['Survived']==1]['Age'], df1[df1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

# Family size and survivability
plt.subplot(236)
plt.hist(x = [df1[df1['Survived']==1]['FamilySize'], df1[df1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


#Now let's visualize the 
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=df1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=df1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=df1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=df1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=df1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=df1, ax = saxis[1,2])


# In[ ]:


#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = df1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = df1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = df1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')


# In[ ]:


#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=df1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=df1, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=df1, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')


# In[ ]:


#more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))

#how does family size factor with sex & survival compare
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=df1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)


# In[ ]:


#how does embark port factor with class, sex, and survival compare
#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
e = sns.FacetGrid(df1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()


# In[ ]:


#histogram comparison of sex, class, and age by survival
h = sns.FacetGrid(df1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()


# In[ ]:


df1.head()


# In[ ]:


pred_atributes = ['Pclass', 'Embarked_Coded', 'IsAlone', 'Sex_Coded', 'Title_Coded', 'Age_Coded']
X = df1[pred_atributes]
Y = df1['Survived']
X_pred = df_pred1[pred_atributes]


# ## Building the model

# In[ ]:


# first let us split the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[ ]:


# Importing the library
from sklearn import ensemble


# In[ ]:


# Using random forest model anf fit with Train Set
forest = ensemble.RandomForestClassifier(n_estimators=20)
forest.fit(X_train, Y_train)


# In[ ]:


# Now let's predict the Test set
Yhat_test = forest.predict(X_test)
Yhat_test[:6]


# In[ ]:


# Now let us compare
compare = Y_test == Yhat_test
compare.mean()


# In[ ]:


# Now let's fit the whole dataset
forest.fit(X, Y)
Yhat = forest.predict(X)

final = Y == Yhat
final.mean()


# In[ ]:


# Now let us predict the Predict Dataset
Y_pred = forest.predict(X_pred)

# Create the output dataframe
output = pd.concat([df_pred[['PassengerId']], pd.Series(Y_pred)], axis=1)
output.rename(columns={0 : 'Survived'}, inplace=True)
output.head()


# In[ ]:


# Let's export the CSV
output.to_csv('output.csv', index=None)


# In[ ]:




