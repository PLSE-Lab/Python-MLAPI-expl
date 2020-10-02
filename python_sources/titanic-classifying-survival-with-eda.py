#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary tools that we might need

# In[ ]:


import numpy as np
import pandas as pd
import os
from keras import models, layers, Sequential
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ## Importing datasets

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


## A merged dataset for common manipulations
merged_df = pd.concat((train_df.drop(['Survived'], axis = 1), test_df))
merged_df.head()


# In[ ]:


print(train_df.shape)
print(test_df.shape)
print(merged_df.shape)


# ## EDA

# In[ ]:


train_df.info()


# Our target class is 'Survived'. I will check individual correlation of the independent variables with our target class. After that I will try some mutated versions of the independent variables.

# In[ ]:


import seaborn as sb
from matplotlib import pyplot as plt


# In[ ]:


train_df.groupby(['Pclass'])['Survived'].sum() / train_df.groupby(['Pclass'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'Pclass', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("Pclass")
plt.show()


# So, there is a good correlation between passenger class and survival.
# 
# I will get to the 'Name' later, as it might not be very straightforward.

# In[ ]:


train_df.groupby(['Sex'])['Survived'].sum() / train_df.groupby(['Sex'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'Sex', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("Sex")
plt.show()


# Sex is also showing a very significant correlation.
# 
# Age has some missing values. We will get to it later.

# In[ ]:


train_df.groupby(['SibSp'])['Survived'].sum() / train_df.groupby(['SibSp'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'SibSp', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("SibSp")
plt.show()


# The relation between survival and the number of siblings or spouses is showing an interesting relation than the people with none. Those who had their spouse or their 1 or 2 siblings with them, had a better odds of survival. More siblings (one is considered to not have more than one spouse with them :p ) actually decreases the chance of survival.

# In[ ]:


train_df.groupby(['Parch'])['Survived'].sum() / train_df.groupby(['Parch'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'Parch', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("Parch")
plt.show()


# This graph only indicates that, minors with their parents or people with few number of children survived more than people with no child or parents. But the relationship is definitely not very strong.
# 
# I am going to merge them together. Let's see what happens.

# In[ ]:


train_df['Family'] = train_df['SibSp'] + train_df['Parch']


# In[ ]:


train_df.groupby(['Family'])['Survived'].sum() / train_df.groupby(['Family'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'Family', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("Family")
plt.show()


# Apart from people with 5 more members in their family, there's a steady relationship between the number of family members and the survival. People with small family survived more than those who were alone or who had more family members.

# In[ ]:


train_df.drop(['Family'], axis = 1, inplace = True) #Dropping it here. We will use it as a custom feature later


# In[ ]:


ticket_num_records = train_df.groupby(['Ticket']).size().sort_values(ascending=False).to_dict()
train_df.groupby(['Ticket']).size().sort_values(ascending=False).head()


# In[ ]:


train_df['Companion'] = train_df['Ticket'].apply(lambda x: ticket_num_records[x])


# In[ ]:


train_df.groupby(['Companion'])['Survived'].sum() / train_df.groupby(['Companion'])['Survived'].count()


# In[ ]:


sb.barplot(x = 'Companion', y = 'Survived', data = train_df)
plt.ylabel("Survived")
plt.xlabel("Companion")
plt.show()


# In[ ]:


train_df.drop(['Companion'], axis = 1, inplace = True)


# What I did here is, mapped people with how many of their companions embarked on the ship with same ticket. We are assuming the ticket numbers are unique.
# 
# This relation is very similar to the one with family size. I think I will skip one of them.

# In[ ]:


train_df.groupby(pd.cut(train_df["Fare"], np.arange(0, 350, 25)))['Survived'].sum() / train_df.groupby(pd.cut(train_df["Fare"], np.arange(0, 350, 25)))['Survived'].count()


# I have distributed the prices in ranges to have a more clear idea about this column's relation with survival. One thing is clear from here, people who paid ticket prices over 75 dollars had better odds of survival. I think I will have a binary feature (more than \$75, less than or eq. $75) in place of 'Fare', as that will be more meaningful.

# In[ ]:


train_df["Cabin"].unique()


# In[ ]:


train_df['CabinId'] = train_df['Cabin'].apply(lambda x: 'None' if pd.isna(x) else x[0])
train_df.groupby(['CabinId'])['Survived'].sum() / train_df.groupby(['CabinId'])['Survived'].count()


# In[ ]:


train_df.drop(['CabinId'], axis = 1, inplace = True)


# People who had cabins, have much higher survival ratio. A binary feature (1 if has cabin, 0 otherwise) should suffice.

# In[ ]:


train_df['Embarked'].unique()


# In[ ]:


train_df.groupby(['Embarked'])['Survived'].sum() / train_df.groupby(['Embarked'])['Survived'].count()


# Seems like this information does not represent much correlation with survival. It should not harm the model either.

# In[ ]:


import re

train_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0]).unique()


# In[ ]:


train_df['Title'] = train_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0])
train_df.groupby(['Title'])['Survived'].sum() / train_df.groupby(['Title'])['Survived'].count()


# In[ ]:


train_df.groupby(['Title'])['Survived'].count()


# In[ ]:


train_df.drop(['Title'], axis = 1, inplace = True)


# Apart from Master, Miss, Mr and Mrs, others are not that much statistically signifigant. I will get that to later.

# Now, there are a lot of missing values in Age column. We could have used mean age from the training dataset.

# In[ ]:


np.nanmean(train_df['Age'])


# But, for some of them, we can use a more targeted approach. The Master title is used for minors. Data proves it too.

# In[ ]:


np.nanmean(train_df[train_df['Name'].str.contains('Master')]['Age'])


# Same goes with the title Miss

# In[ ]:


np.nanmean(train_df[train_df['Name'].str.contains('Miss')]['Age'])


# In preprocessing stage, I will fill the NaN values accordingly.

# ## Preprocessing
# 
# I will work with the merged data for this step.

# In[ ]:


#Adding Title as a feature
merged_df['Title'] = merged_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0])


# In[ ]:


merged_df.head()


# In[ ]:


#We will take the mean values from training data, as per convention

boymean = np.nanmean(train_df[train_df['Name'].str.contains('Master.')]['Age'])
girlmean = np.nanmean(train_df[train_df['Name'].str.contains('Miss.')]['Age'])
meanage = np.nanmean(train_df['Age'])


# In[ ]:


merged_df['Age'] = np.where(np.isnan(merged_df['Age']) & (merged_df['Title'] == 'Master'), boymean, merged_df['Age'])
merged_df['Age'] = np.where(np.isnan(merged_df['Age']) & (merged_df['Title'] == 'Miss'), girlmean, merged_df['Age'])
merged_df['Age'] = merged_df['Age'].fillna(meanage)


# In[ ]:


# Replacing the only missing Fare value with mean Fare. Then converting it to a binary feature

merged_df['Fare'] = merged_df['Fare'].fillna(np.nanmedian(merged_df['Fare']))
merged_df['Fare'] = merged_df['Fare'].apply(lambda x: 1 if x > 75.0 else 0)


# In[ ]:


# Reformatting the Cabin column

merged_df['Cabin'] = merged_df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)


# In[ ]:


# Filling two empty Embarked data with an arbitrary character

merged_df['Embarked'] = merged_df['Embarked'].fillna('N')


# In[ ]:


merged_df['Family'] = merged_df['Parch'] + merged_df['SibSp']


# In[ ]:


merged_df.info()


# In[ ]:


merged_df.head()


# In[ ]:


# Removing features that I don't think has any significance

merged_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis = 1, inplace = True)


# In[ ]:


merged_df.head()


# We want to either normalize or make Age column as ranged data. Let's see if the ranged data makes sense.

# In[ ]:


train_df.groupby(pd.cut(train_df["Age"], np.arange(0, 100, 20)))['Survived'].sum() / train_df.groupby(pd.cut(train_df["Age"], np.arange(0, 100, 20)))['Survived'].count()


# As it shows, the survival ratio is not showing any significant correlation. We will just perform min-max normalization.

# In[ ]:


maxAge = train_df['Age'].max()
minAge = train_df['Age'].min()
merged_df['Age'] = (merged_df['Age'] - minAge)/(maxAge - minAge)


# In[ ]:


merged_df.head()


# In[ ]:


merged_df['Age'].max()


# Now, we will one hot encode all categorical features

# In[ ]:


dummiesPclass = pd.get_dummies(merged_df['Pclass'], prefix = 'Pclass')
merged_df = pd.concat([merged_df, dummiesPclass], axis=1)
merged_df.head()


# In[ ]:


dummiesFare = pd.get_dummies(merged_df['Fare'], prefix = 'Fare')
merged_df = pd.concat([merged_df, dummiesFare], axis=1)
merged_df.head()


# We need to label encode the Sex, Cabin and Title column first. But before that, let's get rid of some statistically insignificant categorie.

# In[ ]:


merged_df.groupby(['Title'])['PassengerId'].count()


# In[ ]:


merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Miss' if (x in ['Mlle', 'Mme', 'Ms']) else x)
merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Mrs' if (x in ['Dona', 'Lady']) else x)
merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Mr' if (x == 'Rev') else x)
merged_df['Title'] = merged_df['Title'].apply(lambda x: x if (x in ['Master', 'Mr', 'Mrs', 'Miss']) else 'Other')


# In[ ]:


merged_df.groupby(['Title'])['PassengerId'].count()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
merged_df['Title'] = le.fit_transform(merged_df['Title'])


# In[ ]:


dummiesTitle = pd.get_dummies(merged_df['Title'], prefix = 'Title')
merged_df = pd.concat([merged_df, dummiesTitle], axis=1)
merged_df.head()


# In[ ]:


merged_df['Sex'] = le.fit_transform(merged_df['Sex'])
dummiesSex = pd.get_dummies(merged_df['Sex'], prefix = 'Sex')
merged_df = pd.concat([merged_df, dummiesSex], axis=1)
merged_df.head()


# In[ ]:


dummiesCabin = pd.get_dummies(merged_df['Cabin'], prefix = 'Cabin')
merged_df = pd.concat([merged_df, dummiesCabin], axis=1)
merged_df.head()


# In[ ]:


merged_df['Embarked'] = le.fit_transform(merged_df['Embarked'])
dummiesEmbarked = pd.get_dummies(merged_df['Embarked'], prefix = 'Embarked')
merged_df = pd.concat([merged_df, dummiesEmbarked], axis=1)
merged_df.head()


# In[ ]:


merged_df.head()


# Let's categorize the Family column into none (N), small(S) and large(L).

# In[ ]:


merged_df['Family'] = merged_df['Family'].apply(lambda x: 'N' if x == 0 else ('S' if x < 4 else 'L'))


# In[ ]:


merged_df['Family'] = le.fit_transform(merged_df['Family'])
dummiesFamily = pd.get_dummies(merged_df['Family'], prefix = 'Family')
merged_df = pd.concat([merged_df, dummiesFamily], axis=1)
merged_df.head()


# In[ ]:


merged_df.drop(['Pclass', 'Sex', 'Fare', 'Cabin', 'Embarked', 'Title', 'Family'], axis = 1, inplace = True)


# In[ ]:


merged_df.head()


# ## Classification
# 
# I will try some classifiers, along with a deep learning model. First, we need to split our dataset again.

# In[ ]:


train_df_x = merged_df[:891]
test_df_x = merged_df[891:]


# In[ ]:


print(test_df_x.shape)
print(test_df.shape)


# In[ ]:


train_df_y = train_df['Survived']


# In[ ]:


train_df = train_df_x.copy()
train_df['Survived'] = train_df_y
train_df.drop(['PassengerId'], axis = 1, inplace = True)
train_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split

def train_and_test(model_specific_tasks, df, it = 20):
    accsum = 0
    minacc = 1.0
    maxacc = 0
    
    for i in range(it):
        print('Iteration: ', (i + 1), end = '\r')
        train, test = train_test_split(df, test_size=0.2)

        train_x = train.drop(['Survived'], axis=1)
        test_x = test.drop(['Survived'], axis=1)

        train_y = train['Survived']
        test_y = test['Survived']

        train_x = np.asarray(train_x).astype('float32')
        train_y = np.asarray(train_y).astype('float32')

        acc = model_specific_tasks(train_x, train_y, test_x, test_y)
        accsum += acc
        minacc = acc if acc < minacc else minacc
        maxacc = acc if acc > maxacc else maxacc
        
    print('Avg. accuracy: ', (accsum / it))
    print('Min. accuracy: ', minacc)
    print('Max. accuracy: ', maxacc)


# ### Logistic Regression

# In[ ]:


def logistic_reg(train_x, train_y, test_x, test_y):
    model = LogisticRegression(solver='sag', max_iter=1000)
    model.fit(train_x, train_y)
    return model.score(test_x, test_y)


# In[ ]:


train_and_test(logistic_reg, train_df.copy(), it = 50)


# ### Random Forest Classifier

# In[ ]:


def rfc(train_x, train_y, test_x, test_y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model.score(test_x, test_y)


# In[ ]:


train_and_test(rfc, train_df.copy())


# ### Neural network

# In[ ]:


def nn(train_x, train_y, test_x, test_y):
    model = Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape = (22,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_x, train_y, epochs=150, batch_size=16, verbose = 0)
    
    return model.evaluate(test_x, test_y, verbose = 0)[1]


# In[ ]:


train_and_test(nn, train_df.copy(), it = 10)

