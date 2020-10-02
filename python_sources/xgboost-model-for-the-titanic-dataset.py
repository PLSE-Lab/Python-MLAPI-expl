#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival Rate from the Titanic Dataset - Preprocessing

# This kernel has been heavily influenced by other kernels, articles and discussions that I came across while figuring out how to proceed analysing this dataset.

# In[ ]:


# Import the packages required to perform data analysis
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Affects the appearence of matplotlib plots
sns.set_palette("husl")
sns.set_style("whitegrid")

pd.options.display.max_rows=None
pd.options.display.max_columns=None

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import the train dataset
df_train = pd.read_csv('../input/train.csv')
# Import the test dataset
df_test = pd.read_csv('../input/test.csv')
# Save PassengerId for final submission
passengerId = df_test.PassengerId
# Join the train and test datasets for preprocessing
df_full= pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
# Create indices to separate data later on
train_idx = len(df_train)
test_idx = len(df_full) - len(df_test)
# Make a copy of df_full
df = df_full.copy()
# Eyeball the data
df.tail()


# In[ ]:


# Study the features of the dataframe
df.info()


# In[ ]:


# Find missing values in each feature
df.isnull().sum()


# The features Age, Cabin and Embarked have missing values. Ignore the missing values in the feature Survived because it shows the missing values from the test dataset.

# ## Feature Engineering

# In[ ]:


# Get a list of the feature names
df.columns.values


# ## Create a feature named 'Title' from the categorical feature 'Name'

# In[ ]:


# Building the logic to write the get_title function
print(df['Name'][0])
print(df['Name'][0].split(',')[1].split('.')[0].strip())


# In[ ]:


# Function to extract titles from names
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'unknown'


# In[ ]:


# Applying the function get_title() on the feature 'Name' using a list comprehension
titles = pd.Series([x for x in df.Name.apply(lambda x : get_title(x))])
title_count = titles.value_counts().sort_index()
title_count


# In[ ]:


# Group the titles that have a low frequency of occurence into an array
rare_titles = title_count[title_count <= 10]
rare_titles.index.values


# In[ ]:


# Next, group the titles to be engineered
rare_titles_Royalty = ['Lady', 'the Countess', 'Don', 'Dona', 'Jonkheer', 'Sir']
rare_titles_Officer = ['Capt', 'Col', 'Major']
rare_titles_Mr = ['Rev']
rare_titles_Mrs = ['Mme']
rare_titles_Miss = ['Mlle', 'Ms']
rare_titles_Doctor = ['Dr']

# Create a function to Normalize the titles
def normalize_titles(df):
    title = df['Title']
    
    if title in rare_titles_Royalty:
        return 'Royalty'
    if title in rare_titles_Officer:
        return 'Officer'
    if title in rare_titles_Mr:
        return 'Mr'
    elif title in rare_titles_Mrs:
        return 'Mrs'
    elif title in rare_titles_Miss:
        return 'Miss'
    elif title in rare_titles_Doctor:
        if df['Sex'] == 'male':
            return 'Mr'
        else: 
            return 'Mrs'
    else: 
        return title


# In[ ]:


# Create a feature named Title in the train dataframe
df['Title'] = titles
df['Title'].describe()


# In[ ]:


# Apply the engineered titles to the Title column
df['Title'] = df.apply(normalize_titles, axis = 1)
df['Title'].describe()


# In[ ]:


# Eyeball the dataframe
df.head()


# ### Next, we analyse the feature 'Age'

# In[ ]:


# Analysing the Age column
print(df['Age'].describe(), "\n")
print("Number NaN values:", df['Age'].isnull().sum())


# The 'Age' feature has 263 missing values. These values have to be imputed before using this feature in any analyses. The best way to imputed these missing values is to do the following:

# In[ ]:


# Group by Sex, Pclass, and Title 
grouped = df.groupby(['Sex', 'Pclass', 'Title'])
grouped.Age.median()


# By grouping 'Age' with ['Sex', 'Pclass', 'Title'] and then taking the median value, we are able to get values that better reflect the original values missing rather than simply taking the median of the of the feature 'Age'. Then use the transform with groupby to apply the median values to the feature 'Age' as shown below.

# In[ ]:


# Apply the grouped median value on the Age NaN
df.Age = grouped.Age.transform(lambda x: x.fillna(x.median(), inplace=False))


# ### Moving on, we analyse the feature 'Cabin'

# In[ ]:


# Analysing the feature named 'Cabin'
print(df.Cabin.describe(), "\n")
print('Number of NaN values:', df.Cabin.isnull().sum())


# Cabin is a categorical feature and the total number of missing values in this feature is too high to be imputed successfully because the dataset does not provide sufficient information to impute this data. Therefore, we shall drop this column from our final preprocessed dataset.

# ### Analysing features 'Embarked' and 'Fare'

# In[ ]:


# Drop feature Cabin since it has few values and is unlikely to impact our analysis
df.drop(['Cabin'], axis=1, inplace=True)
# Fill Embarked NaN values with the most frequently embarked location
df.Embarked = df.Embarked.fillna(df.Embarked.mode().values[0])
# Fill Fare NaN values with median fare
df.Fare = round(df.Fare.fillna(df.Fare.median()))


# In[ ]:


# View changes
df.info()


# ### Create feature 'FamilySize'

# In[ ]:


# Create feature FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# ### Create a checkpoint-1

# In[ ]:


# Checkpoint-1
df_before_dummies = df.copy()
# Eyeball the dataframe
df.head()


# ## Exploratory Data Analysis

# In this section, we take a look at the relationship between the features in our dataset and their impact on survival.

# **1. Looking at Sex, Age and Survival**

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

survived = 'Survived'
died = 'Died'

women = df[df.Sex=='female']
men = df[df.Sex=='male']

ax = sns.distplot(women[women.Survived==1].Age.dropna(), ax=axes[0], bins = 20, label=survived, kde=False, color='c')
ax = sns.distplot(women[women.Survived==0].Age.dropna(), ax=axes[0], bins = 40, label=died, kde=False, color='r')
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men.Survived==1].Age.dropna(), ax=axes[1], bins = 20, label=survived, kde=False, color='b')
ax = sns.distplot(men[men.Survived==0].Age.dropna(), ax=axes[1], bins = 40, label=died, kde=False, color='r')
ax.legend()
ax.set_title('Male')


# The above histogram shows that a higher number of women survived when compared to men. Women between 15 - 35 and men between 25 - 35 have a higher chance of survival.

# **2. Analysing Pclass and Survival**

# In[ ]:


# Create a catplot
ax = sns.catplot(x='Pclass', hue='Survived', col='Title', data=df, palette='inferno',  
                 col_wrap=3, height=4, aspect=1, kind='count')


# The plot above shows that Pclass 1 and Pclass 2 passengers are more likely to survive than Pclass 3 passengers. Also, women survived more than men.

# ### Convert the feature 'Sex' from categorical data to numeric data

# In[ ]:


df.Sex = df.Sex.map({'male':0, 'female':1})


# ### pd .get_dummies()

# In[ ]:


# Create dummy variables for categorical features
Pclass_dummies = pd.get_dummies(df.Pclass, prefix='Pclass')
Title_dummies = pd.get_dummies(df.Title, prefix='Title')
Embarked_dummies = pd.get_dummies(df.Embarked, prefix='Embarked')


# In[ ]:


# Concatenate dummy columns with main dataset
df_dummies = pd.concat([df, Pclass_dummies, Title_dummies, Embarked_dummies], axis=1)
# Drop categorical features
df_dummies.drop(columns={'PassengerId', 'Name', 'Ticket', 'Title', 'Embarked', 'Pclass', 'SibSp', 'Parch'}, inplace=True)
df_dummies.columns.values


# ## Create a checkpoint-2

# In[ ]:


# Re-order the columns in the dataframe
cols_reordered = ['Fare', 'Sex', 'Age', 'FamilySize', 
                  'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Master',
                  'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer',
                  'Title_Royalty', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                  'Survived']

# Re-ordered columns and checkpoint-2
df_preprocessed = df_dummies[cols_reordered]

# Convert features 'Fare' and 'Age' to datatype int
df_preprocessed.Age = df_preprocessed.Age.astype(int)
df_preprocessed.Fare = df_preprocessed.Fare.astype(int)
df_preprocessed.tail()


# #### df_preprocessed is a cleaned dataframe which can be used to fit  machine learning models to predict the survival of passengers.

# # Predicting Survival Rate from the Titanic Dataset - Machine Learning

# In[ ]:


# Import the packages required for machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[ ]:


# Create the train dataset
train_df = df_preprocessed[:train_idx].copy()
print("Number of records:",len(train_df))
# Convert the feature Survived to int
train_df.Survived = train_df.Survived.astype(int)
train_df.tail()


# In[ ]:


# Create the test dataset
test_df = df_preprocessed[test_idx:].copy()
test_df.drop(['Survived'], axis=1, inplace=True)
print("Number of records:",len(test_df))
test_df.tail()


# ## Create the inputs and targets

# In[ ]:


# Create the targets
targets = train_df.Survived.values
print("Length of the target array:",len(targets))
print(targets.shape)

# Create the inputs
inputs = train_df.iloc[:, :-1]
print("Length of the input dataframe:",len(inputs))
print(inputs.shape)


# ## Split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=418, shuffle=True, 
                                                   random_state=20, stratify=targets)


# ## Model - XGBoost

# In[ ]:


# Instantiate the model
xgb = XGBClassifier()
# Fit the data
xgb.fit(X_train, y_train)
# Predict the values
xgb_predictions = xgb.predict(X_train)
# Display the accuracy score of the train dataset
print("Accuracy score: %.2f%%" % (round(xgb.score(X_train, y_train) * 100)))


# Next, check the accuracy score of the training model using cross-validation.

# In[ ]:


kfold = StratifiedKFold(n_splits=10, random_state=20, shuffle=True)
results = cross_val_score(xgb, X_train, y_train, cv=kfold)
print('Accuracy score \nmean: %.2f%% \nsd: %.2f%%' % (results.mean()*100, results.std()*100))


# In[ ]:


# Plot feature importance
fig, axes = plt.subplots(figsize=(10, 8))
plot_importance(xgb, ax=axes)
plt.show()


# ## Feature Selection with XGBoost Feature Importance Scores

# In[ ]:


# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold
thresholds = sorted(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# ## Implementing the model on test data

# In[ ]:


# instantiate the model
model = XGBClassifier()
model.fit(X_train, y_train)
# select features using threshold
selection = SelectFromModel(model, threshold=0.002, prefit=True)
selection_model = XGBClassifier()
select_X_train = selection.transform(X_train)
# train model
selection_model.fit(select_X_train, y_train)
# test model
select_X_test = selection.transform(test_df)
y_pred = selection_model.predict(select_X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_test.shape[1], accuracy*100.0))


# In[ ]:


test_df['PassengerId'] = passengerId
test_df.reset_index(inplace=True)
test_df.drop(['index'], axis=1, inplace=True)
submission_df = pd.concat([test_df.PassengerId, pd.DataFrame(y_pred)], axis=1)
submission_df.columns = ['PassengerId', 'Survived']
submission_df['PassengerId'] = passengerId
submission_df.tail()


# In[ ]:


submission_df.to_csv("../working/submission_titanic.csv", index = False)


# In[ ]:




