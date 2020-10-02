#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis and Feature Engineering

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_X = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')


# In[ ]:


print(train.columns); print(train.shape)
print(test_X.columns); print(test_X.shape)


# In[ ]:


features = list(set(train.columns) - {'Survived'})
target = 'Survived'
train_X, train_y = train[features], train[target]


# In[ ]:


train_X.head()


# In[ ]:


print(train_X.info())


# In[ ]:


print(test_X.info())


# ## Visualizing null values

# In[ ]:


plt.figure()
plt.title('Visualizing null values in train (nulls in white)')
sns.heatmap(train_X.isnull(), cbar=False)


# Features with far too many null values have low predictive power. It makes sense to remove such variables than try to impute and fill in so many null values.

# In[ ]:


null_percent = (train_X.isnull().sum() / train_X.shape[0]).values * 100
plt.figure()
plt.xlabel('Percentage of nulls')
sns.barplot(x=null_percent, y=features)


# In[ ]:


null_threshold = 0.5    # Features with more than null_threshold fraction of nulls will be dropped.
for feature in features:
    null_fraction = train_X[feature].isnull().sum() / train_X.shape[0]
    if(null_fraction > null_threshold):
        print(f'Dropped \'{feature}\' having {round(null_fraction * 100, 2)}% nulls.')
        train_X = train_X.drop(feature, axis=1)
        test_X = test_X.drop(feature, axis=1)
        features.remove(feature)


# In[ ]:


print('Number of unique values for each feature')
train_X.nunique()


# 'Age' and 'Fare' are continuous variables. 'Parch', 'SibSp' and 'Pclass' are ordinal categorical variables. The rest are nominal categorical variables.

# In[ ]:


numeric_features = ['Age', 'Fare', 'Parch', 'SibSp']
ordinal_features = ['Pclass']    # Label encoded
nominal_features = ['Sex', 'Ticket', 'Embarked']    # To one hot encode


# ## Distribution of features

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(8, 3))
plt.tight_layout()
for i, feature in enumerate(['Age', 'Fare']):
    ax[i].set_title(f'Histogram of {feature}')
    sns.distplot(train_X[feature], ax=ax[i], kde=False)


# 'Age' seems to be relatively normally distributed. 'Fare' seems to be rather skewed.

# In[ ]:


fig, ax = plt.subplots(1, 5, figsize=(18, 3))
plt.tight_layout()
for i, feature in enumerate(['Parch', 'SibSp', 'Pclass', 'Sex', 'Embarked']):
    sns.countplot(train_X[feature], ax=ax[i])


# Most were single travellers ('Parch' and 'SibSp'). A large number portion travelled in 3rd class. A majority were male. Most embarked from Southampton.

# ## Multivariate analysis

# In[ ]:


plt.figure()
sns.pairplot(pd.concat([train_y, train_X[numeric_features]], axis=1), 
             hue='Survived', 
             vars=numeric_features,
             markers=['+', 'x'],
             diag_kind=None, 
             dropna=True,
             plot_kws={'alpha': 0.4})


# ## Univariate analysis (effect of features on survival)

# ### Effect of age on survival

# In[ ]:


plt.title('Effect of age on survival')
sns.swarmplot(x=train_y, y=train_X['Age'])


# It looks like in the age bracket 0 to 5, there were more survivors than were killed. The trend is reversed in the middle age groups, where more were killed than survived. This makes sense, as it was ladies and children first. And there were more men than women, so despite ladies first, men would have dominated the not survived section, explaining the blue bulge among the middle-aged.

# ### Effect of fare on survival

# In[ ]:


plt.title('Effect of fare on survival')
sns.swarmplot(x=train_y, y=train_X['Fare'])


# It is quite apparent that among those passengers who paid a zero fare, the vast majority did not survive. Also among those who paid a high fare (Fare > 10), more passengers survived than were killed.

# ### Effect of sex on survival

# In[ ]:


survival_percent = dict(round(train.groupby(by='Sex').mean()['Survived'] * 100, 2))
print(survival_percent)


# A significantly higher proportion of females survived (74.2%) than males (18.89%). Sex seems to be an important feature determining survival.

# ### Effect of age and sex together on survival

# In[ ]:


survived_males_age = train_X['Age'].where((train_X['Sex'] == 'male') & (train_y == 1))
not_survived_males_age = train_X['Age'].where((train_X['Sex'] == 'male') & (train_y == 0))
survived_females_age = train_X['Age'].where((train_X['Sex'] == 'female') & (train_y == 1))
not_survived_females_age = train_X['Age'].where((train_X['Sex'] == 'female') & (train_y == 0))


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(9, 4))
plt.tight_layout()

ax[0].set_title('Male age hist (by survival)')
ax[0].set_xlim([0, 80])
sns.distplot(survived_males_age, ax=ax[0], kde=False, label='Survived')
sns.distplot(not_survived_males_age, ax=ax[0], kde=False, label='Not survived')
ax[0].legend()

ax[1].set_title('Female age hist (by survival)')
ax[1].set_xlim([0, 80])
sns.distplot(survived_females_age, ax=ax[1], kde=False, label='Survived')
sns.distplot(not_survived_females_age, ax=ax[1], kde=False, label='Not survived')
ax[1].legend()


# From the colour coding, it's quite apparent that more males died than survived, and more females survived than not.
# 
# From the graph on the left hand side, we can see that the males in the age range 20-40 had the highest chance of not surviving.
# 
# The spike in the age range 0-5 in both the male and female graphs indicates that children were probably given first preference to be saved.

# ### Effect of number of parents/children on survival

# In[ ]:


survival_percent = dict(round(train.groupby(by='Parch').mean()['Survived'] * 100, 2))
print(survival_percent)


# In[ ]:


plt.figure()
plt.xlabel('Number of parents/children')
plt.ylabel('% survived')
items = survival_percent.items()
parch = [item[0] for item in items]
survival_rates = [item[1] for item in items]
sns.barplot(x=parch, y=survival_rates)


# Highest survival rates were observed among those who had 1, 2 or 3 parents/children.

# ### Effect of number of siblings/spouses on survival

# In[ ]:


survival_percent = dict(round(train.groupby(by='SibSp').mean()['Survived'] * 100, 2))
print(survival_percent)


# In[ ]:


plt.figure()
plt.xlabel('Number of siblings/spouses')
plt.ylabel('% survived')
items = survival_percent.items()
parch = [item[0] for item in items]
survival_rates = [item[1] for item in items]
sns.barplot(x=parch, y=survival_rates)


# Highest survival rates were observed in those who had 1 or 2 siblings/spouses.

# ### Combining Parch and SibSp into one feature FamilyMembers and observing its effect on survival
# 
# It probably makes more sense to have one feature for the total number of family members than have separate counts for the number of parents/children and number of siblings/spouses.

# In[ ]:


train_X['FamilyMembers'] = train_X['Parch'] + train_X['SibSp']
test_X['FamilyMembers'] = test_X['Parch'] + test_X['SibSp']
numeric_features.append('FamilyMembers')


# In[ ]:


survival_percent = dict(round((pd.concat([train_X, train_y], axis=1)).groupby(by='FamilyMembers').mean()['Survived'] * 100, 2))
print(survival_percent)


# In[ ]:


plt.figure()
plt.xlabel('Number of family members')
plt.ylabel('% survived')
items = survival_percent.items()
parch = [item[0] for item in items]
survival_rates = [item[1] for item in items]
sns.barplot(x=parch, y=survival_rates)


# Number of family members clearly impacts survival rates, as seen from the above graph.

# ### Effect of class on survival

# In[ ]:


survival_percent = dict(round(train.groupby(by='Pclass').mean()['Survived'] * 100, 2))
print(survival_percent)


# In[ ]:


plt.figure()
plt.xlabel('Class')
plt.ylabel('% survived')
items = survival_percent.items()
parch = [item[0] for item in items]
survival_rates = [item[1] for item in items]
sns.barplot(x=parch, y=survival_rates)


# Survival rates were highest in 1st class and lowest in 3rd class.

# ### Effect of port of embarking on survival

# In[ ]:


survival_percent = dict(round(train.groupby(by='Embarked').mean()['Survived'] * 100, 2))
print(survival_percent)


# In[ ]:


plt.figure()
plt.xlabel('Port of embarking')
plt.ylabel('% survived')
items = survival_percent.items()
parch = [item[0] for item in items]
survival_rates = [item[1] for item in items]
sns.barplot(x=parch, y=survival_rates)


# Survival rates by class and survival rates by port of embarking show similar patterns. It's possible that class and port of embarking are correlated.

# ## Adding a title feature (Mr., Mrs., Dr., etc.)

# In[ ]:


train_titles = [name.split(',')[1].strip().split(' ')[0] for name in train_X['Name']]
test_titles = [name.split(',')[1].strip().split(' ')[0] for name in test_X['Name']]


# In[ ]:


train_X['Title'] = train_titles
test_X['Title'] = test_titles


# In[ ]:


print('Title occurrences in training set:')
print(train_X['Title'].value_counts())


# In[ ]:


print(train_X['Name'][train_X['Title'] == 'the'])    # Using logical indexing on train_X['Name']


# 'the' seems to be an odd title. But because we split on ' ', the whole title 'the Countess.' became 'the'. So the title should actually be 'Countess'.

# In[ ]:


print('Title occurrences in test set:')
print(test_X['Title'].value_counts())


# Apart from 'Mr.', 'Miss.', 'Mrs.' and 'Master.', the titles have very low occurrences.
# 
# Grouping the low occurrence titles according to the scheme below:
# * 'Major.', 'Col.' and 'Capt.' under 'Military'
# * 'Jonkheer.', 'Lady.', 'Sir.', 'Don.' and 'Dona.' under 'Royalty'
# * 'Mlle.' (Mademoiselle) under 'Miss.' and 'Mme.' (Madame) under 'Mrs.'
# * 'Ms.' under 'Miss.', even though Ms. could mean either Miss. or Mrs. (there's only one occurrence of Ms., so it shouldn't matter anyway)
# * 'the' under 'Royalty' (because of 'the Countess.')
# * 'Dr.' and 'Rev.' get their own categories

# In[ ]:


def group_titles(title):
    """ Function to group low occurrence titles using the scheme above. """
    
    if(title in {'Major.', 'Col.', 'Capt.'}):
        return 'Military'
    elif(title in {'Jonkheer.', 'Lady.', 'Sir.', 'Don.', 'Dona.', 'the'}):
        return 'Royalty'
    elif(title in {'Mlle.', 'Ms.'}):
        return 'Miss.'
    elif(title == 'Mme.'):
        return 'Mrs.'
    else:
        return title


# In[ ]:


train_X['Title'] = train_X['Title'].apply(group_titles)
test_X['Title'] = test_X['Title'].apply(group_titles)


# In[ ]:


print(f'Unique titles in training set:\t{train_X.Title.unique()}')
print(f'Unique titles in test set:\t{test_X.Title.unique()}')


# In[ ]:


nominal_features.append('Title')


# ## Dealing with nulls

# ### Nulls in 'Age'
# 
# Finding which feature best correlates with age (absolute magnitude). Then using it to impute the missing age values, along with using titles as an indicator of age.

# In[ ]:


corr_with_fare = train_X['Age'].corr(train_X['Fare'])
corr_with_family_members = train_X['Age'].corr(train_X['FamilyMembers'])
corr_with_pclass = train_X['Age'].corr(train_X['Pclass'])

print(f'Correlation of Age with Fare = {round(corr_with_fare, 3)}')
print(f'Correlation of Age with FamilyMembers = {round(corr_with_family_members, 3)}')
print(f'Correlation of Age with Pclass = {round(corr_with_pclass, 3)}')


# Age correlates highest (in absolute magnitude) with Pclass. The older a person is, the greater the chance of being able to afford a better class (1st > 2nd > 3rd).

# Instead of imputing age using mean across the age column, it might be a good idea to fill age by grouping primarily using title, and secondarily using class.
# 
# We can expect that age is correlated with title. It's reasonable to assume a 'Mr.' would be older than a 'Master.', and generally speaking, a 'Mrs.' to be older than a 'Miss.'
# 
# Imputing age grouping primarily by title and secondarily by class.

# Filling null values in 'Age' for the training set

# In[ ]:


null_ids_train = train_X['Age'].isnull()
null_ids_train = null_ids_train[null_ids_train != False].index.tolist()    # Passenger indices where age is null in training set
len(null_ids_train)


# Sanity check: len(null_ids_train) matches the number of nulls in the 'Age' column in train_X.

# In[ ]:


fill_values_train = train_X.groupby(by=['Title', 'Pclass']).mean()['Age'].astype(int)
print('Age values to fill classwise, training set:')
print(fill_values_train)


# In[ ]:


for index in null_ids_train:
    title = train_X.loc[index, 'Title']
    pclass = train_X.loc[index, 'Pclass']
    train_X.loc[index, 'Age'] = fill_values_train[(title, pclass)]


# In[ ]:


print(train_X.info())


# Filling null values in 'Age' for the test set

# In[ ]:


null_ids_test = test_X['Age'].isnull()
null_ids_test = null_ids_test[null_ids_test != False].index.tolist()    # Passenger indices where age is null in test set
len(null_ids_test)


# Sanity check: len(null_ids_test) matches the number of nulls in the 'Age' column in test_X.

# In[ ]:


fill_values_test = test_X.groupby(by=['Title', 'Pclass']).mean()['Age'].astype(int)
print('Age values to fill classwise, test set:')
print(fill_values_test)


# In[ ]:


for index in null_ids_test:
    title = test_X.loc[index, 'Title']
    pclass = test_X.loc[index, 'Pclass']
    test_X.loc[index, 'Age'] = fill_values_test[(title, pclass)]


# In[ ]:


print(test_X.info())


# ### Imputing nulls in all other columns

# Imputing numeric features using mean

# In[ ]:


imputer = SimpleImputer(strategy='mean')
train_X[numeric_features] = imputer.fit_transform(train_X[numeric_features])
test_X[numeric_features] = imputer.transform(test_X[numeric_features])


# Imputing ordinal and nominal categorical features using mode

# In[ ]:


imputer = SimpleImputer(strategy='most_frequent')

train_X[ordinal_features] = imputer.fit_transform(train_X[ordinal_features])
test_X[ordinal_features] = imputer.transform(test_X[ordinal_features])

train_X[nominal_features] = imputer.fit_transform(train_X[nominal_features])
test_X[nominal_features] = imputer.transform(test_X[nominal_features])


# In[ ]:


print(f'train_X null count = {train_X.isnull().sum().sum()}')
print(f'test_X null count = {test_X.isnull().sum().sum()}')


# Neither train nor test contain nulls now.

# ## Encoding categorical features

# Label/ordinal encoding ordinal categorical features.
# 
# One hot encoding nominal features.

# In[ ]:


print(f'Ordinal categorical features: {ordinal_features}')
print(f'Nominal categorical features: {nominal_features}')


# ### Dealing with 'Pclass'

# In[ ]:


print(train_X['Pclass'].unique())


# 'Pclass' is already label/ordinal encoded by default.

# ### Dealing with 'Sex', 'Embarked' and 'Title'

# In[ ]:


one_hot_encoder = OneHotEncoder(handle_unknown='error', sparse=False, drop='if_binary')

train_oe_matrix = one_hot_encoder.fit_transform(train_X[['Sex', 'Embarked', 'Title']]).astype('int')
test_oe_matrix = one_hot_encoder.transform(test_X[['Sex', 'Embarked', 'Title']]).astype('int')

print(train_oe_matrix.shape)
print(test_oe_matrix.shape)


# Sanity check: The number of columns in the one hot encoded matrices should be 12 each.
# * 'Sex' has 2 categories, and we've used drop='if_binary', so it contributes 1 column to the OHE matrix.
# * 'Embarked' has 3 categories, so it contributes 3 columns to the OHE matrix.
# * 'Title' has 8 categories, so it contributes 8 columns to the OHE matrix.

# Generating the column names for the one hot encoded features in the format (feature)_(category)

# In[ ]:


one_hot_encoder.categories_


# In[ ]:


features = ['Sex', 'Embarked', 'Title']
ohe_column_names = list()
for i, categories in enumerate(one_hot_encoder.categories_):
    for category in categories:
        ohe_column_names.append(features[i] + '_' + category)
# Dropped first column in 'Sex', so dropping it from ohe_column_names as well
ohe_column_names.pop(0)
print(ohe_column_names)


# In[ ]:


# Converting one hot encoded matrices to dataframe
train_oe = pd.DataFrame(train_oe_matrix, index=train_X.index, columns=ohe_column_names)
test_oe = pd.DataFrame(test_oe_matrix, index=test_X.index, columns=ohe_column_names)

# Dropping columns 'Sex', 'Embarked' and 'Title' as they have been one hot encoded
train_X = train_X.drop(columns=['Sex', 'Embarked', 'Title'])
test_X = test_X.drop(columns=['Sex', 'Embarked', 'Title'])

# Adding the one hot encoded columns to train_X and test_X
train_X = pd.concat([train_X, train_oe], axis=1)
test_X = pd.concat([test_X, test_oe], axis=1)


# 'Ticket' contains far too many categories to be one hot encoded. Maybe it could be label encoded or maybe we could glean some information from it. But for now, dropping the 'Ticket' column.
# 
# Also dropping 'Name' as we have extracted it from it as much useful information as we could.

# In[ ]:


train_X = train_X.drop(columns=['Ticket', 'Name'])
test_X = test_X.drop(columns=['Ticket', 'Name'])


# ## Scaling numeric features

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(7, 3))
fig.suptitle('Before scaling', y=1.1)
plt.tight_layout()
ax[0].set_title('Training')
ax[1].set_title('Test')
sns.scatterplot(train_X['Age'], train_X['Fare'], ax=ax[0], hue=train_y)
sns.scatterplot(test_X['Age'], test_X['Fare'], ax=ax[1])


# In[ ]:


mean = train_X[numeric_features].mean(axis=0)
stddev = train_X[numeric_features].std(axis=0)

for numeric_feature in numeric_features:
    train_X[numeric_feature] = (train_X[numeric_feature] - mean[numeric_feature]) / stddev[numeric_feature]
    test_X[numeric_feature] = (test_X[numeric_feature] - mean[numeric_feature]) / stddev[numeric_feature]


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(7, 3))
fig.suptitle('After scaling', y=1.1)
plt.tight_layout()
ax[0].set_title('Training')
ax[1].set_title('Test')
sns.scatterplot(train_X['Age'], train_X['Fare'], ax=ax[0], hue=train_y)
sns.scatterplot(test_X['Age'], test_X['Fare'], ax=ax[1])

