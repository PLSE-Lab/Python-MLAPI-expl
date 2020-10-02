#!/usr/bin/env python
# coding: utf-8

# Hello, there!
# 
# This notebook presents a solution to the Titanic competition using nothing but **Name** feature to predict whether a passenger survived or died! I tried to comment as much as possible to ease your understanding about each step of the proposed solution.
# 
# The ideas to perform this notebook were developed by Chris Deotte and you can check his notebook in https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818, which is written in R language. 
# 
# Hope you enjoy it reading it as much as I enjoyed making this solution! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1.0 Pre-processing

# In[ ]:


#Train and test data
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

# To make the final submission
holdout_ids = test["PassengerId"] 


# In[ ]:


train.head()


# # 2.0 Variable Engineering

# In this section we will process some of the features in this dataset as well as adding new variables.
# 
# Using **only passenger names**, gender is deduced from title (Mr. Mrs. etc), children are determined from title (Master.), and family groups are identified by duplicate surnames (last names).
# 
# If we just predict females survive and males perish, we cross validate at 78.6% and receive a public leaderboard score of 76.6%.
# To improve upon the gender model, we need to determine which **males survive** and which **females perish**.

# In[ ]:


# Get all the young (under 16) male and store in "male"   
male = train.loc[(train.Sex=='male') & (train.Age < 16)]

# Get all the Pclass 3 female and store in "female"
female = train.loc[(train.Sex=='female') & (train.Pclass==3)]

print(male['Survived'].value_counts())
print(female['Survived'].value_counts())


# By analyzing the Titanic data, we find that many male survivors are among the youth, namely 52.5% = 21/40 of males under 16 years old survive. And most females who perish are among Pclass==3 passengers, namely 50% = 72/144 of Pclass==3 females perish.
# 
# So let's focus on identifying surviving males and perishing females among these 184 passengers. The data indicates that woman and children were prioritized in rescue. Furthermore it appears that woman and children from the same family survived or perished together. 
# 
# **Let's engineer a new feature to identify surviving males and perishing females!**

# In[ ]:


# First, let's extract the Title of each passenger
train['Title'] = 0
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.') #lets extract the title

titles = {
    "Mr":       "man",
    "Mme":      "woman",
    "Ms":       "woman",
    "Mrs":      "woman",
    "Master":   "boy",
    "Mlle":     "woman",
    "Miss":     "woman",
    "Capt":     "man",
    "Col":      "man",
    "Major":    "man",
    "Dr":       "man",
    "Rev":      "man",
    "Jonkheer": "man",
    "Don":      "man",
    "Sir":      "man",
    "Countess": "woman",
    "Dona":     "woman",
    "Lady":     "woman"
}

train["Title"] = train["Title"].map(titles)


# In[ ]:


train.head()


# Now, let's extract the **Surname** of each passenger and use this information along the **Title** feature to set five different options:
# 
# * 1) the passenger is a man - **DeadMan**
# * 2) the passenger is a woman/boy, has no other family member (without considering men) and survived - **LivedPassenger**
# * 3) the passenger is a woman/boy, has no other family member (without considering man) and died - **DeadPassenger**
# * 4) the passenger is a woman/boy and its family has mainly lived ( >= 50% lived) - **LivedFamily**
# * 5) the passenger is a woman/boy and its family has mainly died ( 50% < died) - **DeadFamily** 

# In[ ]:


# Let's extract the Surname of each passenger
train['Surname'] = 0
train['Surname'] = train.Name.str.extract('([A-Za-z]+)\,')


# All the men will be labelled as DeadMan (we only want to test the boys and the woman)
train.loc[(train.Title=='man'), 'Surname'] = 'DeadMan'


# Get women/boys that are alone (no repeated Surname and without considering Man)
unique_surname = (train['Surname'].value_counts().index)[ (train['Surname'].value_counts() == 1) ]

# List that will have all the SINGLE passengers that survived and died
passenger_lived = []
passenger_died = []

for surname in unique_surname:
  single_df = train.loc[(train.Surname == surname)]
  lived = single_df['Survived'].loc[(single_df.Survived == 1)].sum()
  if(lived == 1):
    passenger_lived.append(surname)
  else:
    passenger_died.append(surname)


# Get women/boys that are NOT alone.
non_unique_surname = (train['Surname'].value_counts().index)[ train['Surname'].value_counts() != 1 ].drop('DeadMan')

# List that will have all the FAMILIES that survived and died.
families_lived = []
families_dead = []

# Test if the majority of the family members (> 0.5) lived or died
for surname in non_unique_surname:
  surname_df = train.loc[(train.Surname == surname)]
  number_family = len(surname_df)
  number_lived = surname_df['Survived'].loc[(surname_df.Survived == 1)].sum()
  #print("Family: %s, Size: %d, Lived: %d " % (surname,number_family,number_lived))
  if(number_lived/number_family >= 0.5):
    families_lived.append(surname)
  else:
    families_dead.append(surname)


## Replace the 'Surname' feature according to the classification just made
train['Surname'] = train['Surname'].replace(passenger_lived, 'LivedPassenger')
train['Surname'] = train['Surname'].replace(passenger_died, 'DeadPassenger')
train['Surname'] = train['Surname'].replace(families_lived, 'LivedFamily')
train['Surname'] = train['Surname'].replace(families_dead, 'DeadFamily')


# In[ ]:


train['Surname'].value_counts()


# As it can be seen, all the **Surname** features were mapped into one of the five categories previously defined.
# 
# Finally, let's create a new feature called **SurnameSurvival** which will set if a passenger lived depending upon the **Surname** class:
# * 1) **DeadMan**, **DeadFamily** and **DeadPassenger** means that  **SurnameSurvival** = 0.
# * 2) **LivedPassenger** and **LivedFamily** means that **SurnameSurvival** = 1   

# In[ ]:


# Define new feature
train['SurnameSurvival'] = 0

train.loc[(train.Surname == 'LivedPassenger'), 'SurnameSurvival'] = 1
train.loc[(train.Surname == 'LivedFamily'), 'SurnameSurvival'] = 1


# In[ ]:


family_lived = train.loc[(train.Surname == 'LivedFamily')] 
print(family_lived['Survived'].value_counts())                  


# In[ ]:


prediction_train = train.SurnameSurvival
true_train = train.Survived

print(accuracy_score(true_train, prediction_train))


# By applying the definitions proposed in this notebook, we get an *Accuracy* of **89.7%** considering only the train data!

# # 3.0 Test data

# Now, let's apply what we defined in section 2.0 to the test dataset

# In[ ]:


# Combine train/test data to apply transformations simultaneously
X = pd.read_csv("../input/titanic/train.csv").drop('Survived', axis=1)
df = pd.concat([test,X],ignore_index=True,sort=False)

# Get the index to later separate train and test data.
test_index = test.index


# In[ ]:


# Let's extract the Title of each passenger
df['Title'] = 0
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') #lets extract the title

titles = {
    "Mr":       "man",
    "Mme":      "woman",
    "Ms":       "woman",
    "Mrs":      "woman",
    "Master":   "boy",
    "Mlle":     "woman",
    "Miss":     "woman",
    "Capt":     "man",
    "Col":      "man",
    "Major":    "man",
    "Dr":       "man",
    "Rev":      "man",
    "Jonkheer": "man",
    "Don":      "man",
    "Sir":      "man",
    "Countess": "woman",
    "Dona":     "woman",
    "Lady":     "woman"
}

df["Title"] = df["Title"].map(titles)


# In[ ]:


df.head()


# Now, we will use the **Surname** information processed in the previous section (through the vectors passenger_lived, passenger_died, families_lived and families_died) to define if the passenger **survived** or **died** for both train and test dataset.

# In[ ]:


# Extract the Surname of each passenger
df['Surname'] = 0
df['Surname'] = df.Name.str.extract('([A-Za-z]+)\,')

# All the men will be labelled as DeadMan (we only want to test the boys and the woman)
df.loc[(df.Title=='man'), 'Surname'] = 'DeadMan'


# All previously single passengers that were labeled as lived will be considered lived.
for surname in passenger_lived:
  df['Surname'] = df['Surname'].replace(surname, 'LivedPassenger')


# All previously single passengers that were labeled as dead will be considered dead.
for surname in passenger_died:
  df['Surname'] = df['Surname'].replace(surname, 'DeadPassenger')


# All previously families that were labeled as lived will be considered lived.
for surname in families_lived:
  df['Surname'] = df['Surname'].replace(surname, 'LivedFamily')


# All previosly families that were labeled as dead will be considered dead.
for surname in families_dead:
  df['Surname'] = df['Surname'].replace(surname, 'DeadFamily')


# In[ ]:


df['Surname'].value_counts()


# It is important to notice that there are **95 surnames** in the **test** dataset that **DID NOT** appear in the **train** dataset. 
# 
# In this case, we do not have information if any family member or the person survived or not. Therefore, we have to make some assumptions about who survived.
# 
# We'll split the analysis in two cases: whether there is a single surname occurrence or more than one occurrence.

# In[ ]:


unique_surname = (df['Surname'].value_counts().index)[ df['Surname'].value_counts() == 1 ] 
unique_surname


# There still **88 passengers** that has no other female/boy relative in the test dataset.
# 
# For this group of people, we will consider two cases:
# * If it is a boy, he will be labeled as **DeadPassenger**.
# * If it is a female, she will be labeled as **LivedPassenger**.

# In[ ]:


# Get the single passenger information
single_passenger = df[df['Surname'].isin(unique_surname.to_list())]

# Separate into single_woman and single_boy DataFrames
single_woman = single_passenger['Surname'].loc[(single_passenger.Title=='woman')]
single_boy = single_passenger['Surname'].loc[(single_passenger.Title=='boy')]

# All the women alone will be labelled as LivedPassenger
df['Surname'] = df['Surname'].replace(single_woman, 'LivedPassenger')

# All the boys alone will be labelled as DeadPassenger
df['Surname'] = df['Surname'].replace(single_boy, 'DeadPassenger')


# In[ ]:


df['Surname'].value_counts()


# Now, there still **7 families** that did not appear previously in the training dataset. 
# 
# Before performing any assumption, let's see the information about these passengers.

# In[ ]:


new_families = df.loc[(df.Surname != 'DeadMan') & (df.Surname != 'LivedPassenger') & (df.Surname != 'DeadPassenger') & (df.Surname != 'LivedFamily') & (df.Surname != 'DeadFamily')]
new_families_list = (df['Surname'].value_counts().index)[ df['Surname'].value_counts() < 5 ]
new_families


# 1. The majority of passengers are **women** and there are only **3 boys**.
# 
# As seen in section 2, the great majority of families considering only **women** and **boys** survived. Therefore, we will also apply this consideration in these samples, so the families above will be labeled as **LivedFamily**.
# 
# However, it is interesting to notice that both members of the **Billiard** family are boys and there is no **woman** with them. Therefore we will set both of them as **DeadPassenger**.
# 

# In[ ]:


df['Surname'] = df['Surname'].replace('Billiard', 'DeadFamily')
df['Surname'] = df['Surname'].replace(new_families_list, 'LivedFamily') #the highest value is this one where everyone here is dead!

df['Surname'].value_counts()


# Now, we will make the predictions based on these **Surname** feature. We will consider the same strategy developed in the training section.

# In[ ]:


# Define new feature
df['SurnameSurvival'] = 0

df.loc[(df.Surname == 'LivedPassenger'), 'SurnameSurvival'] = 1
df.loc[(df.Surname == 'LivedFamily'), 'SurnameSurvival'] = 1


# Finally, we will consider this **SurnameSurvival** as our prediction variable.

# In[ ]:


prediction = df['SurnameSurvival'].loc[test_index]
prediction


# # 4.0 Creating a submission file

# In[ ]:


submission_df = {"PassengerId": holdout_ids,
                 "Survived": prediction}

submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv",index=False)

