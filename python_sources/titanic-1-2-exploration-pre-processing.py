#!/usr/bin/env python
# coding: utf-8

# # Titanic (1/2): Exploration and Preprocessing

# - This is a Kernel (my first one on Kaggle) to explore and pre-process (cleaning, visualizing and feature engineering) the Titanic Dataset
# - The output will be used in my next Kernel to model, predict and score on the Test Dataset and submit my results.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ## Import Dataset

# In[ ]:


IN_CLOUD  = True
INPUT_DIR = '../input' if IN_CLOUD else './data'


# In[ ]:


train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')


# #### Print types of the columns and get an overview of the Dataset by print few random rows: 

# In[ ]:


# Overview of the Dataset

print(train_df.dtypes)
train_df.sample(8)


# ### Check for Missing Values in columns:

# In[ ]:


# Check Missing Values

print(f'Count of Missing Values for each Column (out of {len(train_df)}): ')
print(train_df.isnull().sum())


# ##### We notice that having 891 rows, we have:
#     - 687 missing values in the 'Cabin' column, which is about 77%
#     - 177 missing values in the 'Age' column
#     - 2 missing values in the 'Embarked' column
# 

# - We will drop the 'Cabin' column because it has too many missing values

# In[ ]:


train_df.drop('Cabin', axis=1, inplace=True)


# - Since we only have 2 missing values in 'Embarked', we can fill them with the most occured value.

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:



train_df.Embarked.fillna('S', inplace=True)
assert not train_df.Embarked.isnull().any()


# - We will deal with the Age column later, but for now let's fill the missing values with **the mean of all ages**.
#   
#   ( ***We will also save the rows that have missing age values to process them later again*** )

# In[ ]:


missing_age_rows = train_df[train_df['Age'].isnull()].copy() # Save for later processing
missing_age_rows.sample(3)


# In[ ]:


train_df.Age.fillna(train_df['Age'].mean(), inplace=True)
assert not train_df.Age.isnull().any()


# ## Some quick Visualizations & Processings on Features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


COLOR_SURVIVED='#57e8fc'
COLOR_DEAD='#fc5e57'


# ### Overall Survival Ratio
# <span style="color:red; font-weight: bold">#Data Visualisation</span>

# In[ ]:


# Survival Ratio

labels = ['Dead', 'Survived']
val_counts = train_df.Survived.value_counts()
#print(vals)
sizes = [val_counts[0], val_counts[1]]
colors = [COLOR_DEAD, COLOR_SURVIVED ]
#print(sizes)

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, explode=(0.1,0), autopct='%1.1f%%', colors=colors)
ax.axis('equal')
plt.title('Overall Survival Ratio')
plt.show()


# ### The "Sex" Feature:
# <span style="color:red; font-weight: bold">#Feature Encoding  #Data Visualisation</span>

# #### We will convert the values in 'Sex' column to integers because it's more practical and fast

# In[ ]:


def encode_sex(sex_col):
    return sex_col.map({'female': 0, 'male': 1}).astype('int')


# In[ ]:


train_df.Sex = encode_sex(train_df.Sex)
print(train_df.Sex.dtype)
train_df.Sex.unique()


# In[ ]:


COLOR_MALE   = '#6699ff'
COLOR_FEMALE = '#ff66ff'


# In[ ]:


val_counts = train_df.Sex.value_counts()
sizes  = [val_counts[0], val_counts[1]]
labels = ['Female', 'Male']
colors = [COLOR_FEMALE, COLOR_MALE]

print(val_counts, labels)
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, explode=(0.1, 0), autopct='%1.1f%%', colors= colors)
ax.axis('equal')
plt.title('Count of Passengers by Sex')
plt.show()


# In[ ]:


ct = pd.crosstab(train_df.Sex, train_df.Survived)

ind = np.arange(2)
survived_vals = [ct.loc[1][1], ct.loc[0][1]]
dead_vals = [ct.loc[1][0], ct.loc[0][0]]
print(ct)

width=0.3

plt.bar(ind, survived_vals, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, dead_vals, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, ('Men', 'Women'))
plt.yticks(np.arange(0, 600, 50))
plt.legend( loc='upper right')
plt.show()

#ax.bar(ct)


# ### The "Age Category" Feature:
# <span style="color:red; font-weight: bold">#Feature Engineering #Data Visualisation</span>

# #### We create a new Feature 'AgeCat' to assign each passenger an Age Category:
#     - less than 14 years old => 0 (Kids)
#     - 14 ~ 22 years old      => 1 (Teens)
#     - 22 ~ 35 years old      => 2 (Adults)
#     - 35 ~ 50 years old      => 3 (Bid Adults)
#     - more than 50 years old => 4 (Seniors)

# In[ ]:


def construct_age_cat_col(age_col):
    age_cat_col = pd.Series([-1] * len(age_col))
    for i, val in age_col.iteritems():
        if val < 14:                 # Kids
            age_cat_col[i] = 0
        elif val >= 14 and val < 22: # Teens
            age_cat_col[i] = 1
        elif val >= 22 and val < 35: # Adults
            age_cat_col[i] = 2
        elif val >= 35 and val < 50: # Big Adults
            age_cat_col[i] = 3
        elif val >= 50:              # Seniors
            age_cat_col[i] = 4
        else:
            raise ValueError('Preprocessing Age: Age Value unsupported ! ', val)
    return age_cat_col


# In[ ]:


print('Information about the ages of the passengers:')
#print(train_df.Age.describe())

train_df['AgeCat'] = construct_age_cat_col(train_df.Age)

train_df.sample(5)


# #### Proportion of each Age Category

# In[ ]:


labels = ['Kids', 'Teens', 'Adults', 'Big Adults', 'Seniors']

ct = pd.crosstab(train_df.AgeCat, train_df.Survived, margins=True)
cats = list(ct.index.values)
cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)
cats.sort()
print(cats)


# In[ ]:


sizes = list(ct.loc[cats, 'All'])
print(sizes)
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.title('Count of Passengers by Age Category')
plt.show()


# #### Survivants by Age Category

# In[ ]:


ind = np.arange(5)
width = 0.25

survivants_values = list(ct.loc[cats, 1])
deads_values = list(ct.loc[cats, 0])


plt.bar(ind,  survivants_values, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, ('Kids', 'Teens', 'Adults', 'Big Adults','Seniors'))
plt.yticks(np.arange(0, 300, 25))
plt.legend(loc='upper right')
plt.show()


# ### The "Passenger Class" Feature:
# <span style="color:red; font-weight: bold">#Data Visualisation</span>

# In[ ]:


ct = pd.crosstab(train_df.Pclass, train_df.Survived, margins=True)
cats = list(ct.index.values)
cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)
cats.sort()
print(cats)


# In[ ]:


sizes = list(ct.loc[cats, 'All'])
print(sizes)
labels = ['Class 1', 'Class 2', 'Class 3']
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.title('Count of Passengers by Class')
plt.show()


# In[ ]:



survivants_values = list(ct.loc[cats, 1])
deads_values = list(ct.loc[cats, 0])

ind = np.arange(3)
width = 0.2
plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, ('1', '2', '3') )
plt.yticks(np.arange(0, 500, 50))
plt.legend(loc='upper right')
plt.show()


# ### The "Embarked" Feature: 
# <span style="color:red; font-weight: bold">#Data_Visualisation</span>

# In[ ]:


ct = pd.crosstab(train_df.Embarked, train_df.Survived, margins=True)
cats = list(ct.index.values)
cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)
print(cats)


# In[ ]:


sizes = list(ct.loc[cats, 'All'])
labels=cats
print(sizes)
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.title('Count of passengers by Embarked')
plt.show()


# In[ ]:



survivants_values = list(ct.loc[cats, 1])
deads_values = list(ct.loc[cats, 0])

ind = np.arange(len(cats))
width = 0.2
plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, (cats) )
plt.yticks(np.arange(0, 500, 50))
plt.legend(loc='upper right')
plt.show()


# ### The "Number of Relatives" Feature:
# <span style="color:red; font-weight: bold">#Feature Engineering #Data_Visualisation</span>

# - We use the columns ***'SibSp'*** (number of Siblings and Spouses) and ***'Parch'*** (number of Parents and Children) to engineer a new Feature called ***'NbrRelatives'*** which is the total number of relatives for the passenger on board.

# In[ ]:


def construct_nbr_relatives_col(sibsp_col, parch_col):
    return sibsp_col+parch_col


# In[ ]:


train_df['NbrRelatives'] = construct_nbr_relatives_col(train_df['SibSp'], train_df['Parch'])
train_df.sample(3)


# In[ ]:


ct = pd.crosstab(train_df.NbrRelatives, train_df.Survived)

cats = list(ct.index.values)
print(cats)

survivants_vals = ct.loc[:, 1]
deads_vals = ct.loc[:, 0]


# In[ ]:


ind = np.arange(len(cats))
width = 0.2
plt.bar(ind, survivants_vals, width, label='Survived' , color=COLOR_SURVIVED)
plt.bar(ind+width, deads_vals, width, label='Dead', color=COLOR_DEAD)
plt.xticks(ind+width/2, cats)
plt.legend(loc='upper right')
plt.show()


# - Drop the **'SibSp'** and **'Parch'** Columns

# In[ ]:


train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# ### The "IsAlone" Feature ?
# <span style="color:red; font-weight: bold">#Feature Engineering #Data_Visualisation</span>

# - We continue by identifying the passengers who are alone (who don't have relatives on board) using a new Feature **'IsAlone'**

# #### Create the IsAlone Feature:

# In[ ]:


def construct_is_alone_col(nbr_relatives_col):
    return nbr_relatives_col.apply(lambda x: True if x == 0 else False)


# In[ ]:


train_df['IsAlone'] = construct_is_alone_col(train_df.NbrRelatives)
train_df.sample(3)


# #### Visualization of Survival counts by IsAlone  :

# In[ ]:


ct = pd.crosstab(train_df.IsAlone, train_df.Survived)
cats = ['Alone' if x is True else 'Not Alone' for x in list(ct.index.values)]

survivants_values = list(ct.loc[:, 1])
deads_values = list(ct.loc[:, 0])

ind = np.arange(len(cats))
width = 0.3
plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, (cats) )
plt.yticks(np.arange(0, 500, 50))
plt.legend(loc='upper right')
plt.show()


# ### The "Title" Feature: 
# 
# <span style="color:red; font-weight: bold">#Feature Engineering #Data_Visualization</span>

# #### Create and Explore the Title Feature: 
# 
# - In the **'Name'** Column, we notice that there is a pattern. We can extract the **Title** of each passenger into its own feature. It can be useful because it gives us some information about the socio-economic class of the passenger

# In[ ]:


train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]*)\.', expand=False)
print('Counts of different Titles:')
train_df['Title'].value_counts()


# - We can see that many Titles don't occur very much, so we can collect them under a new Title called ***'Rare'***

# In[ ]:


rare_titles = ['Jonkheer', 'Don', 'Sir', 'Countess', 'Capt', 'Jonkheer', 'Dona', 'Major', 'Dr', 'Rev', 'Col', 'Lady']
train_df['Title'].replace(rare_titles, 'Rare', inplace=True)
train_df[train_df['Title'] == 'Rare'].sample(3)


# - Gather the Female titles which aren't useful for prediction into one Title (Miss)

# In[ ]:


train_df['Title'].replace(['Ms', 'Mme', 'Mlle', 'Mrs'], 'Miss', inplace=True)


# - We make it into a function to use it later with Test Data:

# In[ ]:


def construct_title_col(name_col):
    title_col = name_col.str.extract(r'([A-Za-z]*)\.', expand=False)
    rare_titles = ['Jonkheer', 'Don', 'Sir', 'Countess', 'Capt', 'Jonkheer', 'Dona', 'Major', 'Dr', 'Rev', 'Col', 'Lady']
    title_col.replace(rare_titles, 'Rare', inplace=True)
    title_col.replace(['Ms', 'Mme', 'Mlle', 'Mrs'], 'Miss', inplace=True)
    return title_col


# In[ ]:


# Check if working by droping the Column and creating it again using the function
train_df.drop('Title', axis=1, inplace=True)
train_df['Title'] = construct_title_col(train_df['Name'])
train_df.sample(5)


# #### Replace Age of Master Passengers who had missing values :
# 
# - We notice below, that all passengers with title **'Master'** have age less than 12 (except for those that we edited earlier  which have the Age == 29~ which is the mean of all ages)
# 
# ( I just show 5 random rows here for better visualization )

# In[ ]:


train_df[train_df['Title']=='Master'].sample(5)


# - We will then replace the previously missing age values for the 'Master' passengers with the mean of (only) the 'Master' passengers ages, which is a better estimation than the previous value of mean of all passengers ages

# In[ ]:


train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId'])) , 'Age'] = np.NaN
mean_age_masters = train_df.loc[ (train_df['Title']=='Master') ].Age.mean()
print('Mean of Master passengers\'s ages :' , mean_age_masters)
print('Number of Master passengers who will be affected by the change: ', train_df[train_df['Title'] == 'Master']['Age'].isnull().sum())


# In[ ]:


train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId'])) , 'Age'] = mean_age_masters
train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId']))]


# In[ ]:


# Check
train_df[train_df['Title'] == 'Master']['Age'].max()


# #### Visualization of Survival Counts by Title:

# In[ ]:


ct = pd.crosstab(train_df.Title, train_df.Survived)
cats = list(ct.index.values)
print(cats)
#ct


# In[ ]:



survivants_vals = ct.loc[cats, 1]
deads_vals = ct.loc[cats, 0]

ind = np.arange(len(cats))
width = 0.25
plt.bar(ind, survivants_vals, width, label='Survived', color=COLOR_SURVIVED)
plt.bar(ind+width, deads_vals, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, cats)
plt.legend(loc='upper right')
plt.show()


# - Now we can drop the "Name" Columns:

# In[ ]:


train_df.drop('Name', axis=1, inplace=True)


# ## Preprocess the Test Data:
# 
# - Now that we got a good overview of the Training Data, we proceed to the final preprocessings on both Training and Test Data

# In[ ]:


test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')
test_df.sample(5)


# In[ ]:


print(f'Checking for Missing values in Test Dataset (out of {len(test_df)}): ')
test_df.isnull().sum()


# - The missing values in Cabin are not a problem, we will drop the column anyway
# - Regarding missing values in'Age' Column, we will impute them as we imputed them in Train Dataset:
#     - Replace all NaN values of passengers with Title Master with Mean of Ages of passengers that have 'Master' Title
#     - The Others replace them with the mean of all passengers
# - We also have one missing value in 'Fare' Column: we will replace it with the mean Fare of passengers which have the same Pclass, same Number of Relatives to get a better approximation

# ### Construct Title, NbrRelatives, IsAlone Columns for Test Dataset:

# In[ ]:


test_df['Title'] = construct_title_col(test_df['Name'])
test_df.sample(3)


# In[ ]:


test_df['NbrRelatives'] = construct_nbr_relatives_col(test_df['SibSp'], test_df['Parch'])
test_df.sample(3)


# In[ ]:


test_df['IsAlone'] = construct_is_alone_col(test_df['NbrRelatives'])
test_df.sample(3)


# ### Handle Missing Values in 'Age' Column of Test Dataset:

# In[ ]:


print('Nbr of Missing Age Values for passengers with Master Title', test_df[test_df['Title'] == 'Master'].Age.isnull().sum())


# In[ ]:


mean_ages_masters = test_df[test_df['Title'] == 'Master'].Age.mean()
print(mean_ages_masters)
test_df.loc[ (test_df['Title'] == 'Master') & (test_df['Age'].isnull()), 'Age'] = mean_ages_masters

assert not test_df[test_df['Title'] == 'Master'].Age.isnull().any()


# In[ ]:


print('Nbr of Missing Age Values for passengers except Master Title', test_df[test_df['Title'] != 'Master'].Age.isnull().sum())


# In[ ]:


mean_ages_all = test_df.Age.mean()
print(mean_ages_all)
test_df['Age'] = test_df.Age.fillna( mean_ages_all )

assert not test_df.Age.isnull().any()


# ### Handle the Missing Value in "Fare" column of Test Dataset:

# #### Check the Row concerned :

# In[ ]:


test_df.loc[test_df.Fare.isnull(), :]


# #### We can see that :
#     - It's a passenger in the Pclass 3
#     - It's a passenger with 0 Number of Relatives
#     - It's a passenger who embarked from 'S' (Southhampthon)
# - We will then check the passengers who meet this ( or most of ) these criterions: ( there are 92 rows, I just show 3 random ones for better visualization )

# In[ ]:


similar_passengers = test_df.loc[(test_df.Pclass == 3) & (test_df.IsAlone == True) & (test_df.Embarked == 'S'), :]
similar_passengers.sample(3)


# - We will use the Mean Fare of those passengers to fill the missing Fare value :

# In[ ]:


assert test_df.Fare.isnull().any()
similar_passengers_mean_fare = similar_passengers.Fare.mean()
print('Mean Fare of Similar Passengers: ', similar_passengers_mean_fare)
test_df.Fare.fillna(similar_passengers_mean_fare, inplace=True)
assert not test_df.Fare.isnull().any()
# Check the Passenger
test_df.loc[test_df.PassengerId == 1044,:]


# ### Construct AgeCat Column of Test Dataset:

# In[ ]:


test_df['AgeCat'] = construct_age_cat_col(test_df['Age'])
test_df.sample(3)


# ### One-Hot Encoding of the "Embarked" Feature:

# In[ ]:


def encode_embarked(embarked_col):
    #return embarked_col.map({'S': 2, 'Q': 1, 'C': 0}).astype('int')
    return pd.get_dummies(data=embarked_col, columns=['Embarked'], prefix='Embarked')


# In[ ]:


one_hot_embarked_cols = encode_embarked(train_df['Embarked'])
train_df = pd.concat([train_df, one_hot_embarked_cols], axis=1)
train_df.sample(3)


# In[ ]:


one_hot_embarked_cols = encode_embarked(test_df['Embarked'])
test_df = pd.concat([test_df, one_hot_embarked_cols], axis=1)
test_df.sample(3)


# - We can now drop the Embarked Column :

# In[ ]:


train_df.drop('Embarked', axis=1, inplace=True)
test_df.drop('Embarked', axis=1, inplace=True)


# ### Encoding The "Title" Feature

# In[ ]:


def encode_title(title_col):
    #return title_col.map({ 'Mr': 0, 'Miss': 1, 'Master': 2, 'Rare': 3 }).astype('int')
    return pd.get_dummies(data=title_col, prefix='Title')


# In[ ]:


one_hot_title_cols = encode_title(train_df['Title'])
train_df = pd.concat([train_df, one_hot_title_cols], axis=1)


# In[ ]:


one_hot_title_cols = encode_title(test_df['Title'])
test_df = pd.concat([test_df, one_hot_title_cols], axis=1)


# In[ ]:


train_df.sample(3)


# In[ ]:


test_df.sample(3)


# In[ ]:


train_df.drop('Title', axis=1, inplace=True)
test_df.drop('Title', axis=1, inplace=True)


# ### Encoding of the "Sex" Feature:

# In[ ]:


test_df.Sex = encode_sex(test_df.Sex)
test_df.sample(3)


# #### Some Last Column Drops and Checks :

# In[ ]:


train_df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
test_df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


print('Check columns of both datasets:')
train_df.columns, test_df.columns


# In[ ]:


OUTPUT = True
OUTPUT_DIR = '.' if IN_CLOUD else INPUT_DIR
if OUTPUT:
    train_df.to_csv(f'{OUTPUT_DIR}/train_clean.csv', index=False)
    test_df.to_csv(f'{OUTPUT_DIR}/test_clean.csv', index=False)
    print('Done Outputing to CSV')


# In[ ]:


train_df.sample(5)


# In[ ]:


test_df.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:




