#!/usr/bin/env python
# coding: utf-8

# This is my first attempt to create an analytical notebook. It involves Exploratory Data Analysis on the "Titanic" dataset .
# Constructive feedback will be appreciated :)
# 
# ## STEPS
# 
# 1)Importing necessary Libraries
# 
# 2)Importing and understanding the required data
# 
# 3)Exploratory data analysis and data visualization
# 
# 4)Data cleaning
# 
# 

# ## 1)Importing necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting data
import seaborn as sns #for plotting data

from sklearn.preprocessing import LabelEncoder #to convert categorical variables into numerical values

#for ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# ## 2)Importing and understanding the data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.describe(include = 'all')


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# ### Some initial conclusions(by looking at raw data)
# 
# -  Three features __'Age'__ , __'Cabin'__ and __'Embarked'__ have missing values.These need to be adressed.
# -  'Age' seems to be an important feature for predicting survival rates and hence we should not discard it.
# -  'Cabin' has a lot of NULL values and may not provide much information so it may be discarded.
# -  'Embarked' only has 2 NULL values so they  will be filled with the  maximum ocurring category.

# ## 3)Exploratory data analysis and data visualization

# ### a. Survival based on the passenger class

# In[ ]:


print("Percentage of Pclass = 1 who survived:", round(train["Survived"][train["Pclass"] == 1]
                                                      .value_counts(normalize = True)[1]*100),"%")

print("Percentage of Pclass = 2 who survived:", round(train["Survived"][train["Pclass"] == 2]
                                                      .value_counts(normalize = True)[1]*100),"%")

print("Percentage of Pclass = 3 who survived:", round(train["Survived"][train["Pclass"] == 3]
                                                      .value_counts(normalize = True)[1]*100),"%")

sns.catplot(x = 'Pclass' , y = 'Survived',kind = 'point',data = train);



# __This shows that the Upper class are the ones with the best survival percentage.__

# ### b. Survival based on Sex

# In[ ]:


print("Survival % of Male:", round(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100),"%")

print("Survival % of Female:", round(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100),"%")
sns.catplot(x = 'Sex' , y = 'Survived',kind = 'point',data = train);


# __Females have a significant higher chance of surviving than males__

# ### c) Survival based on SibSp and Parch

# In[ ]:


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
g1=sns.catplot(x = 'SibSp',kind = 'count',data = train,ax = ax1);
g2=sns.catplot(x = 'SibSp' , y = 'Survived',kind = 'bar',data = train,ax = ax2);
plt.close(g1.fig)
plt.close(g2.fig)
plt.show()


# In[ ]:


for i in range(0,max(train["SibSp"])+1):
    if i in (6,7):
        continue
    else:
        print("Total passengers with", i , "siblings and/or spouse:" ,train["SibSp"].value_counts(sort = False)[i])


# -  __Since the sample size of passengers with SibSp > 2 is very less(less than 20),hence the survival rate trends can be misleading for them.__
# -  __But it can clearly be seen that those with 1 or 2 SibSp had a higher rate of survival than those with 0 SibSp.__
# 

# In[ ]:


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
g1=sns.catplot(x = 'Parch',kind = 'count',data = train,ax = ax1);
g2=sns.catplot(x = 'Parch' , y = 'Survived',kind = 'bar',data = train,ax = ax2);
plt.close(g1.fig)
plt.close(g2.fig)
plt.show()


# In[ ]:


for i in range(0,max(train["Parch"])+1):
    print("Total passengers with", i , "parent or child:" ,train["Parch"].value_counts(sort = False)[i])


# -  __Again since the sample size of passengers with Parch > 2 is very less(less than 10),hence the survival rate trends can be misleading for them.__
# -  __But it can clearly be seen that those with 1 or 2 Parch had a higher rate of survival than those with 0 SibSp.__

# ### NOTE: 
# I could not decide whether to perform analysis on __"Embarked"__ and __"Age"__ features before or after filling the missing values.
# Is there any rule on whether we should perform data cleaning  first followed by data analysis or vice versa???? If there is then please explain in the comments.
# I have decided to first clean the data and then perform analysis on above two features. 

# ## 4) Data cleaning

# __At the beginning we looked at our training dataset. Now let's take a look at our test dataset before cleaning them.__

# In[ ]:


test.head(10)


# In[ ]:


test.isnull().sum()


# -  __'Cabin'__ feature has a lot of NULL values and does not give much useful information so it may be dropped
# -  __'Fare' and 'Ticket'__ features may  also be dropped.
# -  __'Age'__ feature is important so missing values has to be filled.
# 
# __So we need to fill 'Age' and 'Embarked' missing values in our training dataset and 'Age' missing values in our test dataset__
# __Unnecessary features will be dropped once the cleaning part  is done__

# ### Working with 'Age' feature
# 
# Since there are a lot of missing values so rather than just filling them with average age I will first calculate the average age corresponding to each __title__(Title will be extracted out from the __'Name'__ feature) and then missing age values will be filled with avg age value corresponding  to  the title of the person whose age is missing.  

# In[ ]:


# combining both train and test datasets because both have missing Age values
TrainTest = [train, test]

#extract a title for each Name in the train and test datasets
for data in TrainTest:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


#To ease the analysis combining the titles into fewer categories
for data in TrainTest:
    data['Title'] = data['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    data['Title'] = data['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


test.head()


# In[ ]:


#Calculating the average age according to each title
title= ['Mr','Miss','Mrs','Master','Royal','Rare'];
mr_age = round(train[train["Title"] == 'Mr']["Age"].mean()) 
print('Average age of title Mr: ',mr_age)
miss_age = round(train[train["Title"] == 'Miss']["Age"].mean())
print('Average age of title Miss: ',miss_age)
mrs_age = round(train[train["Title"] == 'Mrs']["Age"].mean())
print('Average age of title Mrs: ',mrs_age)
master_age = round(train[train["Title"] == 'Master']["Age"].mean())
print('Average age of title Master: ',master_age)
royal_age = round(train[train["Title"] == 'Royal']["Age"].mean())
print('Average age of title Royal: ',royal_age)
rare_age = round(train[train["Title"] == 'Rare']["Age"].mean())
print('Average age of title Rare: ',rare_age)
avg_age = [mr_age,miss_age,mrs_age,master_age,royal_age,rare_age]


# In[ ]:


#Filling the missing values in train dataset
n_rows= train.shape[0]   
n_titles= len(title)
for i in range(0, n_rows):
    if np.isnan(train.Age[i])==True:
        for j in range(0, n_titles):
            if train.Title[i] == title[j]:
                train.Age[i] = avg_age[j]

train['Age'].isnull().sum()


# In[ ]:


#Filling the missing values in test dataset  
n_rows= test.shape[0]   
n_titles= len(title)
for i in range(0, n_rows):
    if np.isnan(test.Age[i])==True:
        for j in range(0, n_titles):
            if test.Title[i] == title[j]:
                test.Age[i] = avg_age[j]

test['Age'].isnull().sum()


# In[ ]:


#Creating different AgeGroups
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# -  __Babies have the greatest chance of survival__

# ### 'Embarked' feature
# 
# 
# Since there are only 2 missing values in training set and none in test set hence we will directly fill those 2 missing values with the maximum occuring option.

# In[ ]:


print("Number of people embarking in S:",train[train["Embarked"] == "S"].shape[0]) 


print("Number of people embarking in C:",train[train["Embarked"] == "C"].shape[0])


print("Number of people embarking in Q:",train[train["Embarked"] == "Q"].shape[0])

sns.catplot(x='Embarked',kind = 'count',data=train)
plt.show()


# __Clearly most of the people embarked at 'S' so we will fill the missing values with the same.__

# In[ ]:


train = train.fillna({"Embarked": "S"})
train['Embarked'].isnull().sum()


# __Now that all the missing values of important features have been filled ,the two things left to do are:__  
# 1.  to map the non-numerical feature values into numerical values before using them to train the model
# 2.  to drop the unnecessary features
# 
# But first let us take a look at our training and test dataset.

# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# ## Mapping the features 'AgeGroup' ,'Embarked' and 'Sex'

# In[ ]:


#Age Group
labelEncoder = LabelEncoder()
train.AgeGroup=labelEncoder.fit_transform(train.AgeGroup)
test.AgeGroup=labelEncoder.fit_transform(test.AgeGroup)


# In[ ]:


#Sex
train.Sex=labelEncoder.fit_transform(train.Sex)
test.Sex=labelEncoder.fit_transform(test.Sex)


# In[ ]:


train.Embarked=labelEncoder.fit_transform(train.Embarked)
test.Embarked=labelEncoder.fit_transform(test.Embarked)


# ## Dropping features which are not required

# In[ ]:


train=train.drop(['Name','Age','Ticket','Fare','Cabin','Title'],axis=1)
test=test.drop(['Name','Age','Ticket','Fare','Cabin','Title'],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# Now our data is clean and ready.

# __Creating this notebook was really fun and I got to learn a lot of things.No doubt this is the best way to learn  for newbies like me. 
# I hope this notebook will be useful in someway for people new to this.__

# ### Since this was my first attempt at this, I had to take some ideas  from other amazing  sources present here...
# ### Here are the link to these amazing notebooks
# 
# -  https://www.kaggle.com/startupsci/titanic-data-science-solutions
# -  https://www.kaggle.com/omarelgabry/a-journey-through-titanic
# -  https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
# -  https://www.kaggle.com/rochellesilva/simple-tutorial-for-beginners
# 
# 
# # Please provide your valuable feedback on this and ways to improve it!!!
# # Thank You
