#!/usr/bin/env python
# coding: utf-8

# **Initialization**
# Import libraries
# Read input data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import style

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#read input data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# **Missing data**
# 
# Look for missing data. 
# If data is missing, we have 2 options:
# * impute the missing values, if many samples are affected
# * choose to drop there samples, if only a few or if too many samples are affected
# 
# Observations:
# * training data is missing age from many samples, cabin data is missing from most of the samples and embarked info misses from 2 samples only.
# * test data is missing age from many samples, fare from only one sample and cabin from most of the samples.

# In[ ]:


print('Missing training data: \n',train_data.isnull().sum())
print('Missing testing data: \n',test_data.isnull().sum())


# **Missing values: Age**
# 
# Training data:
# 177 samples are missing the Age value => we must impute Age (we cannot drop this amount of samples)
# We will use the same values to impute Age for test data too.
# 
# Observation about the Name field:
# - We notice there is a "title" attribute (ex: Mr, Master etc) embedded in the "Name" column 
# - This title might be an indication of age, for females at least (Miss vs Mrs)
# - To explore this further, we extract the title attribute into its own separate column (which we name 'Title')
# 
# **Extract Title and verify assumption:**
# - The older a passenger is, the higher class (s)he is travelling in
# - For a more accurate imputation, we will look at the mean age in passengers grouped by (Sex,Title,Pclass). We suspect that passengers of the same sex, same title, will choose a higher class when they are older and we examine the data (below) to confirm this. We will use the mean age in each of these subgroups to replace missing values
# - We also look at the frequency of each of the subgroups obtained in the above step (because, for example, we don't want to replace 60 missing values in a subgroup with an average of 3 existing values in that subgroup)

# In[ ]:


# extract title from the name column
coltitle = train_data['Name'].apply(lambda s: pd.Series({'Title': s.split(',')[1].split('.')[0].strip(),
                                                   'LastName':s.split(',')[0].strip(), 'FirstName':s.split(',')[1].split('.')[1].strip()}))
# appaend title to the train_data
train_data = pd.concat([train_data, coltitle], axis=1) # add new column to our data
# we no longer need column 'Name' 
train_data.drop('Name', axis=1, inplace=True)

# print(train_data['Title'])
# some titles are in French. 
# we translate them to English (standardize our data)
train_data.loc[train_data['Title']=='Mlle', 'Title']='Miss'.strip()
train_data.loc[train_data['Title']=='Ms', 'Title']='Miss'.strip()
train_data.loc[train_data['Title']=='Mme', 'Title']='Mrs'.strip()

# For each sex, group by title and for each title group by ticket class
# The assumption is that for the same title, younger people travel in lower class (3rd is considered lowest class)
# Inspect data to confirm hypothesis
print('Inspect mean age in each subgroup as a function of ticket class')
print(pd.pivot_table(train_data, index=['Sex', 'Title', 'Pclass'], values=['Age'], aggfunc='mean'))

# For the same grouping method as above, inspect data to make sure we will not infer data based on a sample too small
print('Inspect number of available age data points in each subgroup and total number of individuals in each subgroup')
print(pd.pivot_table(train_data, index=['Sex', 'Title', 'Pclass'], values=['Age', 'FirstName'], margins=False, aggfunc='count'))


# **Impute Age for train and test data**

# In[ ]:


# We conclude it is safe to replace missing age data with the mean of each subgroup
def fillAges(row):
    
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 30
        elif row['Title'] == 'Mrs':
            return 40

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return 23
        elif row['Title'] == 'Mrs':
            return 34

    elif row['Sex']=='female' and row['Pclass'] == 3:
        
        if row['Title'] == 'Miss':
            return 16
        elif row['Title'] == 'Mrs':
            return 34

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Mr':
            return 42
        elif row['Title'] == 'Dr':
            return 42

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Mr':
            return 33

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return 5
        elif row['Title'] == 'Mr':
            return 29

train_backup = train_data.copy()                
train_data['Age'] = train_data.apply(lambda s: fillAges(s) if np.isnan(s['Age']) else s['Age'], axis=1)

#use the same values means from train_data to replace the missing values in test_data
test_backup = test_data.copy()
test_data['Age'] = train_data.apply(lambda s: fillAges(s) if np.isnan(s['Age']) else s['Age'], axis=1)


# **Examine distribution of Age in train data before and after imputation**

# In[ ]:


f, axes = plt.subplots(1, 2);
sns.distplot(train_backup.loc[~np.isnan(train_backup['Age']),'Age'], axlabel="Age distribution in train data", ax=axes[0])
sns.distplot(train_data['Age'], axlabel="Age distribution in train data", ax=axes[0])

sns.distplot(test_backup.loc[~np.isnan(test_backup['Age']),'Age'], axlabel="Age distribution in test data", ax=axes[1])
sns.distplot(test_data['Age'], axlabel="Age distribution in test data", ax=axes[1])


# **Missing values: Embarked**
# 
# Two samples are missing this value.
# We look at what is the most popular embarkment point and impute this value.

# In[ ]:


# visually inspect the two samples
print(train_data[train_data['Embarked'].isnull() == True])
print('\n')

print("--Frequency for point of embarkment--")
train_data.groupby('Embarked').Survived.count().plot(kind='bar')
plt.show();

# most of the passengers embarked at point 'S', so we impute this value to the two missing samples
train_data.loc[train_data['Embarked'].isnull() == True, 'Embarked']='S'.strip()


# **Missing values: Fare**
# 
# 1 sample in test data is missing Fare. We impute with the mean Fare from train data 

# In[ ]:


test_data.loc[np.isnan(test_data['Fare']),'Fare']=train_data['Fare'].mean()


# **Missing value: Cabin**
# 
# * Only 204 samples of the 891 in the training data have the 'Cabin' attribute filled in
# * Cabin has duplicates -> some passengers probably shared a cabin (e.g. families)
# * We will impute the Cabin as follows:
#     * samples with missing Cabin will get a 0
#     * samples with Cabin filled in will get a 1

# In[ ]:


print(train_data['Cabin'].describe())
train_data.loc[pd.isna(train_data['Cabin']), 'Cabin'] = 0
train_data.loc[train_data['Cabin']!=0,'Cabin'] = 1


# 
# 
# **Exploratory data analysis**
# 
# 1. Establish what data types we have -> will inform what exploratory analysis we can perform on each attribute
# 
# Pclass - categorical, ordinal, integer [1=highest,2,3]
# 
# Sex - categorical, nominal (and dychotomous), text ['male', 'female']
# 
# Age - numerical, continuous, ratio, int, many values
# 
# SibSp - numerical, discrete, ratio, 
# [0, 1, 2, 3, 4, 5, 8]
# 
# Parch - numerical, discrete, ratio,
# [0, 1, 2, 3, 4, 5, 6, 9]
# 
# Cabin - categorical, integer [0=missing, 1=existing]
# 
# Embarked - categorical, [S, C, Q]
# 
# Title - categorical, nominal, text ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady',
#        'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona']
# 
# 
# 2. Descriptive statistics for an overview

# In[ ]:


#1 data types
print("--Data types for training data--")
train_data.info()
print('_'*60)
print("--Data types for test data--")
test_data.info()

# generate descriptive statistics
print('_'*60)
print("--Descriptive statistics for numerical training data--")
print(train_data.describe()) # if no argument provided, the result includes all numeric columns
# We notice:
# 38% survival rate
# mean age is 29
# SibSp range [0:8]
# Parch range [0:6]

#2 descriptive statistics for categorical features
print('_'*60)
print("--Descriptive statistics for categorical training data--")
print(train_data.describe(include=['O']))
# Name values do not repeat -> confirmation that there were no mistakes where passengers are entered twice
# Sex has 2 possible values, 58% of test subjects are 'male'
# Ticket has duplicates -> families probably share a ticket
# Embarked has 3 values, most passengers embarked at point 'S'


# **3. Visualize data**
# 

# **Data visualization: Passenger class**
# 
# Most passengers are in third class. 
# First class > 60% survival rates
# 
# **Conclusion**: consider Pclass for model training

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2)
f.suptitle('Passenger class')
train_data.groupby('Pclass').Survived.mean().plot(kind='bar', ax=axes[0], legend="True", title="Survival rate")
train_data.groupby('Pclass').Pclass.count().plot(kind='bar', ax=axes[1], title="Frequency")


# **Data visualization: Sex**
# 
# Most passengers are males, but most survivors are females
# 
# **Conclusion**: consider Sex for model training
# 

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2)
f.suptitle('Sex')
train_data.groupby('Sex').Survived.mean().plot(kind='bar', ax=axes[0], legend="True", title="Survival rate")
train_data.groupby('Sex').Sex.count().plot(kind='bar', ax=axes[1], title="Frequency")


# **Data visualization: Age**
# 
# Some age 'bins' have better survival rates: most children under 15 survived.
# Other age groups have lower rates (people between 20 and 30 for example)

# In[ ]:


plt.figure(figsize=(8,3))
plt.hist(x = [train_data[train_data['Survived']==1]['Age'], train_data[train_data['Survived']==0]['Age']], 
         stacked=False, color = ['g','r'],label = ['Survived','Dead'],bins=26)
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.figure(figsize=(8,3))
ax = sns.kdeplot(train_data[train_data['Survived']==1]['Age'], color="darkturquoise", shade=True)
sns.kdeplot(train_data[train_data['Survived']==0]['Age'], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# **Age: binning**
# 
# We notice that:
# 1. children (approx 0 to 16 yo) have highest survival rates
# 2. next there is an age group with quite low survival rates (approx 16 to 30-something yo)
# 3. passengers over 60-something have low survival rates
# 
# **Conclusion**
# We bin the Age data into 5 bins, each spanning approx 16 years.

# In[ ]:


#let's have a quick look at survival rates for 5 age bands, to confirm our hypothesis
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
print(train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

#let's create the 
train_data['AgeBin'] = 0

train_data.loc[train_data['Age']<=16,'AgeBin']=0     
train_data.loc[(train_data['Age']>16)&(train_data['Age']<=32),'AgeBin']=1 
train_data.loc[(train_data['Age']>32)&(train_data['Age']<=48),'AgeBin']=2 
train_data.loc[(train_data['Age']>48)&(train_data['Age']<=64),'AgeBin']=3 
train_data.loc[(train_data['Age']>64)&(train_data['Age']<=80),'AgeBin']=4

plt.figure(figsize=(5,3));
train_data.groupby('AgeBin').Survived.mean().plot(kind='bar', legend="False", title="Survival rate for age bins")
s = pd.Series(["0-16","16-32","32-48","48-64","64-80"]);
plt.xticks([0,1,2,3,4],s)
plt.xlabel('Age Bins')
plt.ylabel('Survival Rate')
plt.show()


# **Data visualization: SibSp and Parch**
# 
# **Siblings or spouse**
# Most people have no siblings nor spouse.
# But highest survival rates are for those with 1 or 2.
# The population from 2 to 8 SibSp is weakly represented in the data
# 
# **Conclusion**: 
# 1. consider SibSp for model training
# 2. consider engineering 2 categories only: 0 if SibSp=0 and 1 if SibSp>0
# 
# **Parents or children**
# Most people have no parent or children.
# But they have less survival rates than those with 1, 2 or 3. 
# Passengers with 3-6 Parch have very low frequency in our train data.
# 
# **Conclusion**: 
# 1. consider Parch for model training
# 2. consider engineering 2 categories only: 0 if Parch=0 and 1 if Parch>0

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
f.suptitle('Total number of siblings and spouses')
train_data.groupby('SibSp').Survived.mean().plot(kind='bar', ax=axes[0], legend="True", title="Survival rate")
train_data.groupby('SibSp').SibSp.count().plot(kind='bar', ax=axes[1], title="Frequency")
axes[0].set_ylabel('Survival rate');
axes[1].set_ylabel('Frequency');

f, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
f.suptitle('Total number of parents and children')
train_data.groupby('Parch').Survived.mean().plot(kind='bar', ax=axes[0], legend="True", title="Survival rate")
train_data.groupby('Parch').Parch.count().plot(kind='bar', ax=axes[1], title="Frequency")
axes[0].set_ylabel('Survival rate');
axes[1].set_ylabel('Frequency');


# **Data engineering: Single from SibSp and Parch**
# 
# We notice in the plots from previous sections that:
# * singles (no SibSp and no Parch) have lower survival rates 
# 
# We therefore create a new attribute:
# * Single = categorial, 0 or 1
# * Single = 0 if (SibSp + Parch) = 0 and 1 otherwise
# 
# **Data engineering: FamilySize from SibSp and Parch**
# 
# FamilySize = SibSp + Parch

# In[ ]:


train_data['Single']=0
train_data.loc[(train_data['Parch'] + train_data['SibSp'] == 0),'Single']=1
s = pd.Series(["w/ family","single"]);

f, axes = plt.subplots(nrows=1, ncols=2)
f.suptitle('Single versus with family and survival')
train_data.groupby('Single').Survived.mean().plot(kind='bar', ax=axes[0], title="Survival rate")
plt.xticks([0,1],s);

train_data.groupby('Single').Single.count().plot(kind='bar', ax=axes[1], title="Frequency")
plt.xticks([0,1],s);

train_data['FamilySize']=0
train_data['FamilySize']=train_data['Parch'] + train_data['SibSp']+1
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,4))
f.suptitle('Family size and survival')
train_data.groupby('FamilySize').Survived.mean().plot(kind='bar', ax=axes[0], title="Survival rate")
train_data.groupby('FamilySize').FamilySize.count().plot(kind='bar', ax=axes[1], title="Frequency")


# **Data visualization: Ticket**
# 
# 681 unique tickets, so 210 passengers are sharing their ticket
# 
# **Data engineering: SharedTicket & SharedTicketCount from Ticket**
# 
# We create a new attribute: **SharedTicket**
# 1. SharedTicket is 0 for those with unique tickets
# 2. SharedTicket is 1 for those who share tickets
# 
# **Observations:**
# 1. We notice that most people are not sharing a ticket (they travel with unique tickets). But those who share a ticket have much higher survival rates.
# 
# We create another attribute: **SharedTicketCount** = number of passengers who hold the same particular ticket 
# 
# **Observations:**
# 1. We see that the ideal group sizes for highest survival chances are: 2, 3 and 4 passengers on the same ticket. 
# This confirms what we have previously seen for attribute "FamilySize"
# 
# **Conclusion**
# SharedTicket and SharedTicketCount seem to confirm the information we already got from Single and FamilySize and not to add anything new 

# In[ ]:


df = pd.DataFrame(train_data['Ticket'].value_counts())
train_data['SharedTicket'] = 0
train_data['SharedTicketCount'] = 0
for index, row in train_data.iterrows():
    ticketName = row['Ticket']
    #print('ticket name:',ticketName)
    #print('# of occurrences:',df.loc[ticketName,'Ticket'])
    train_data.loc[index,'SharedTicketCount'] = df.loc[ticketName,'Ticket']    
    if df.loc[ticketName,'Ticket'] > 1:
        train_data.loc[index,'SharedTicket'] = 1
    else:
        train_data.loc[index,'SharedTicket'] = 0 
        
f, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(7,7))
f.suptitle('Shared versus non shared ticket and survival')

s = pd.Series(["non-sharing","sharing"]);
train_data.groupby('SharedTicket').Survived.mean().plot(kind='bar', ax=axes[0,0], title="Survival rate")
plt.xticks([0,1],s);
train_data.groupby('SharedTicket').SharedTicket.count().plot(kind='bar', ax=axes[0,1], title="Frequency")
plt.xticks([0,1],s); 

train_data.groupby('SharedTicketCount').Survived.mean().plot(kind='bar', ax=axes[1,0], title="Survival rate")
train_data.groupby('SharedTicketCount').SharedTicketCount.count().plot(kind='bar', ax=axes[1,1], title="Frequency")


# **Data visualization: Cabin**
# 
# Passengers with existing information about Cabin have double survival rates than the ones missing Cabin data.
# Most of the travellers did not have Cabin data though.
# **Conclusion** Consider 'Cabin' attribute for model training 

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2)
f.suptitle('Passengers with missing or existing Cabin information')

s = pd.Series(["missing","existing"]);

train_data.groupby('Cabin').Survived.mean().plot(kind='bar', ax=axes[0], title="Survival rate")
axes[0].set_xticklabels(s)
axes[0].set_xlabel('Cabin data');
axes[0].set_ylabel('Survival rate');

train_data.groupby('Cabin').Cabin.count().plot(kind='bar', ax=axes[1], title="Frequency")
axes[1].set_xticklabels(s)
axes[1].set_xlabel('Cabin data');
axes[1].set_ylabel('Frequency');


# **Data visualization: Embarked**
# 
# Most passengers embarked at 'S'
# But passengers who embarked at 'C' have highest survival rates
# 
# **Conclusion**: consider Embarked for model training
# 

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2)
f.suptitle('Point of Embarkment')

train_data.groupby('Embarked').Survived.mean().plot(kind='bar', ax=axes[0], title="Survival rate")
axes[0].set_xlabel('Point of Embarkment');
axes[0].set_ylabel('Survival rate');

train_data.groupby('Embarked').Embarked.count().plot(kind='bar', ax=axes[1], title="Frequency")
axes[1].set_xlabel('Point of Embarkment');
axes[1].set_ylabel('Frequency');


# **Data visualization: Title**
# 
# There are some rare titles with weak representation in the data.
# Replace all rare titles with a single category name and anaylize again.
# 
# **Conclusion** 
# 1. The title confirms what we've seen for Sex: women have higher survival rates than men. 
# 2. Unmaried women have lower survival rates than maried women (as seen in the analysis of family size vs survival rates too)
# 3. Children have high survival rates (Title is 'Master')

# In[ ]:


f, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(7,7))
f.suptitle('Title and survival')

train_data.groupby('Title').Title.count()

train_data.groupby('Title').Survived.mean().plot(kind='bar', ax=axes[0,0], title="Survival rate")
axes[0,0].set_xlabel('Title');
axes[0,0].set_ylabel('Survival rate');

train_data.groupby('Title').Title.count().plot(kind='bar', ax=axes[0,1], title="Frequency")
axes[0,1].set_xlabel('Title');
axes[0,1].set_ylabel('Frequency');

train_data.loc[train_data['Title']=='Capt', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Col', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Don', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Dr', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Jonkheer', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Lady', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Major', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Rev', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='Sir', 'Title']='Rare'.strip()
train_data.loc[train_data['Title']=='the Countess', 'Title']='Rare'.strip()

train_data.groupby('Title').Survived.mean().plot(kind='bar', ax=axes[1,0], title="Survival rate")
axes[1,0].set_xlabel('Engineered Title');
axes[1,0].set_ylabel('Survival rate');

train_data.groupby('Title').Title.count().plot(kind='bar', ax=axes[1,1], title="Frequency")
axes[1,1].set_xlabel('Engineered Title');
axes[1,1].set_ylabel('Frequency');
plt.show()

#let's examine the Master population to confirm this is the group of children
print("--Who are the masters--")
masters = train_data[train_data['Title']=='Master']['Age']
masters.describe()


# **Correlation map**

# In[ ]:


def titanic_corr(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
   
plt.figure(figsize=(10,10))
titanic_corr(train_data)


# 4. Correcting the data: remove unnecesary information
# 
# After visual inspection, we observe there is unnecessary information. 
# 
# Passenger ID
# Ticket 
# Fare - or leave ?
# Cabin - too many samples lack this information****
# Name

# In[ ]:


titanic.drop('Ticket', axis=1, inplace=True)
titanic.drop('Fare', axis=1, inplace=True)


# # To save the pre-processed data

# In[ ]:


train_data.head()
train_data.to_csv('train_preprocessed.csv')


# **Conveting categorical features** 
# 
# #Pclass - 2
# #Sex - 3
# #Age - 4
# #Sibsp - 5
# #Parch - 6
# #Embarked - 7
# #Title - 8

# In[ ]:


label=titanic.iloc[0:890,1]
data=titanic.iloc[0:890,[2,3,4,5,6,7,8]]
testdata=titanic.iloc[891:1309,[2,3,4,5,6,7,8]]
titanic=[data,testdata]

#print(label)
#print(data)
#print(testdata)

label = label.astype(int)

for change in titanic:
    change['Sex']=change['Sex'].map({'female':0,'male':1}).astype(int)
    
for change in titanic:
    change['Embarked']=change['Embarked'].map({'S':0,'C':1, 'Q':2}).astype(int)
    
for change in titanic:
    change['Title']=change['Title'].map({'Master':0,'Miss':1, 'Mr':2, 'Mrs':3, 'Rare':4, 'Sir':5}).astype(int)

print('Final data after wrangling')
print(data)


# **KNeighborsClassifier**
# 
# With default model parameters

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(data,label)
predictions=clf.predict(testdata)
print(predictions)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('kneighbors.csv', index=False)
print("Your submission was successfully saved!")


# **LogisticRegression**
# 
# With default model parameters

# In[ ]:


regr = LogisticRegression()
regr.fit(data, label)

# Predicted Values for Survived
predict_r = regr.predict(testdata)
print (predict_r)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predict_r})
output.to_csv('regression.csv', index=False)
print("Your submission was successfully saved!")

