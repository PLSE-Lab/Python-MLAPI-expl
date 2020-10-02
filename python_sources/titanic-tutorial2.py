#!/usr/bin/env python
# coding: utf-8

# # Titanic Tutorial for Beginners[Accuracy: 86.64(Training Data)]-
# 
# 
# * This is my second tutorial on this problem statement. Do point out my mistakes in comment section.
# * Do upvote if you find this notebook interesting.
# * In previous tutorial(https://www.kaggle.com/rishabhdhyani4/titanic-tutorial), I ignored Name feature which I have corrected in this tutorial and achieved better accuracy.

#  This is default first cell in any kaggle kernel. They import **NumPy** and **Pandas** libraries and it also lists the available Kernel files.** NumPy** is the fundamental package for scientific computing with Python. **Pandas** is the most popular python library that is used for data analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data Loading
# 
# 
# Our first step is to extract train and test data. We will be extracting data using pandas function read_csv. Specify the location to the dataset and import them.

# In[ ]:


# Reading data
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_test_copy=df_test.copy()


# In[ ]:


# Have a first look at train data
print(df_train.shape)


# By using df_train.shape we get to know that train data has 891 rows and 12 columns.

# In[ ]:


# Now, lets explore first five data from training set.
df_train.head()


# We got 12 features in our training data. From https://www.kaggle.com/c/titanic/data, we have:
# 
# * Survival = Survival
# * Pclass = Ticket class
# * Sex = Sex
# * Age = Age in years
# * Sibsp = # of siblings / spouses aboard the Titanic
# * Parch = # of parents / children aboard the Titanic
# * Ticket = Ticket number
# * Fare = Passenger fare
# * Cabin = Cabin number
# * Embarked = Port of Embarkation
# 
# Qualitative Features (Categorical) : PassengerId , Pclass , Survived , Sex , Ticket , Cabin , Embarked.
# 
# Quantitative Features (Numerical) : SibSp , Parch , Age , Fare.
# 
# It is obvious from the problem statement that we have to predict **Survival** feature.

# In[ ]:


# We will use describe function to calculate count,mean,max and other for numerical feature.
df_train.describe().transpose()


# In[ ]:


# The feature survived contain binary data which can also be seen from its max(1) and min(0) value.


# **Our next step is to examine NULL values.**

# In[ ]:


# Have a look for possible missing values
df_train.info()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


# We see that Age, Cabin and Embarked feature have NULL values.


# In[ ]:


# Have a first look at test data
print(df_test.shape)


# In[ ]:


# Have a look at train and test columns
print('Train columns:', df_train.columns.tolist())
print('Test columns:', df_test.columns.tolist())


# It looks OK, the only additional column in train is 'Survived', which is our target variable, i.e. the one we want to actually predict in the test dataset.

# In[ ]:


# Let's look at the figures and Understand the Survival Ratio
df_train.Survived.value_counts(normalize=True)


# In[ ]:


# We observe that less people survived.


# In[ ]:


# To get better understanding of count of people who survived, we will plot it.


# # Load our plotting libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot(x='Survived',data=df_train)


# So, out of 891 examples only 342 (38%) survived and rest all died.

# # Feature Examining-

# ## ** Pclass**
#  Come, let's examine Survival based on Pclass.

# In[ ]:


sns.countplot(x='Pclass',data=df_train,hue='Survived')


# **On examining the chart above, wer can clearly say that people belonging to third class died in large numbers.**

# ##  **Sex**
# 
# Come, let's examine Survival based on gender.

# In[ ]:


sns.countplot(x='Sex',data=df_train,hue='Survived')


# In[ ]:


sns.catplot(x='Sex' , y='Age' , data=df_train , hue='Survived' , kind='violin' , palette=['r','g'] , split=True)


# **On examining the chart above, we can clearly say that male are more likely to die in comparision to female.**

# In[ ]:


# Creating function to convert Sex attribute to numerical feature.
def mappy(frame):
    
    frame['Sex'] = frame.Sex.map({'female': 1 ,  'male': 0}).astype(int)
    


# In[ ]:


mappy(df_train)
df_train.head()


# In[ ]:


mappy(df_test)
df_test.head()


# ## **Age**
# 
# Come, let's examine Survival based on gender.

# **We have noticed earlier that column Age has some null values. So. first we will complete the Age column and then we will analyze it.**

# In[ ]:


sns.kdeplot(df_train.Age , shade=True , color='r')


# **Fill the Age with it's Median, and that is because, for a dataset with great Outliers, it is advisable to fill the Null values with median.**

# In[ ]:


#Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.
guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


# Now we will be filling missing age values. This is used to fill the age according to Pclass and Sex.
combine = [df_train, df_test]
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

df_train.head()


# In[ ]:


print(df_train.Age.count())  # Null values filled


# In[ ]:


sns.factorplot(x='Sex',y='Age' , col='Pclass', data=df_train , hue='Survived' , kind = 'box', palette=['r','g'])


# In[ ]:


# Understanding Box Plot :

# The bottom line indicates the min value of Age.
# The upper line indicates the max value.
# The middle line of the box is the median or the 50% percentile.
# The side lines of the box are the 25 and 75 percentiles respectively.


# ## **Fare**
# 
# Come, let's examine Survival based on Fare.

# In[ ]:


plt.figure(figsize=(20,30))
sns.factorplot(x='Embarked' , y ='Fare' , kind='bar', data=df_train , hue='Survived' , palette=['r','g'])


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Embarked',y='Fare',data=df_train,hue='Survived')


# **We observe that people who paid more are more likely to survive.**

# In[ ]:


df_test.info()


# In[ ]:


# We notice that fare has one missing value.
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
df_test.head()


# ## **Embarked**
# 
# Come, let's examine Survival based on Embarked.

# **We have noticed earlier that column Embarked has some null values. So. first we will complete this column and then we will analyze it.**

# In[ ]:


# The best way to fill it would be by most occured value
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0] ,inplace=True)


# In[ ]:


df_train.Embarked.count() # filled the values with Mode.


# ## **Cabin**
# 
# Come, let's examine Survival based on Embarked.

# In[ ]:


#Since Cabin has so many missing value, we will remove that column.


# In[ ]:


df_train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


df_test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.violinplot(x='Embarked' , y='Pclass' , data=df_train , hue='Survived' , palette=['r','g'])


# **We can see that those who embarked at C with First Class ticket had a good chance of Survival. Whereas for S, it seems that all classes had nearly equal probability of Survival. And for Q, third Class seems to have Survived and Died with similar probabilities.**

# In[ ]:


df_train.isnull().sum()


# In[ ]:


# None of the columns are empty.


# ## **SibSp**
# 
# Now lets analyze SibSp column.

# In[ ]:


sns.countplot(data=df_train,x='SibSp',hue='Survived')


# In[ ]:


df_train[['SibSp','Survived']].groupby('SibSp').mean()


# **It seems that there individuals having 1 or 2 siblings/spouses had the highest Probability of Survival, followed by individuals who were Alone.**

# ## **Parch**
# 
# Now lets analyze Parch column.

# In[ ]:


df_train[['Parch','Survived']].groupby('Parch').mean()


# **It seems that individuals with 1,2 or 3 family members had a greater Probability of Survival, followed by individuals who were Alone.**

# **Now let us perform some feature engineering to get informative and valuable attributes.**

# # **Feature Engineering:**

# **We need to analyze Name feature.  Based on it we will retain the new Title feature for model training.**

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# **We can replace many titles with a more common name or classify them as Rare.**

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# We will be converting categorical titles to numerical.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train.head()


# **Now let us create an attribute 'Alone' so that we could know whether the passenger is travelling alone or not.**

# In[ ]:


df_train['Alone'] = 0
df_train.loc[(df_train['SibSp']==0) & (df_train['Parch']==0) , 'Alone'] = 1

df_test['Alone'] = 0
df_test.loc[(df_test['SibSp']==0) & (df_test['Parch']==0) , 'Alone'] = 1


# In[ ]:


df_train.head()


# *  Now we are going to drop features which are not contributing much.
# * Names, PassengerId and Ticket Number doesn't help in finding Probability of Survival.
# * We have created Alone feature and therefore I'll be Dropping SibSp and Parch.

# In[ ]:


drop_features = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' ]

df_train.drop(drop_features , axis=1, inplace = True)


# In[ ]:


df_test.info()


# In[ ]:


df_test.head()


# In[ ]:


drop_featuress = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' ]

df_test.drop(drop_featuress , axis=1 , inplace = True)


# In[ ]:


df_test.head()


# ### Lets convert categorical feature into numerical value.
# 
# * Divide Age into 5 categories and Map them with 0/1/2/3/4.
# * Divide Fare into 4 categories and Map them to 0/1/2/3.
# * Embarked Attribute has (S/C/Q) , which will be mapped to 0/1/2.

# In[ ]:


def mapping(frame):
    
    
    
    frame['Embarked'] = frame.Embarked.map({'S' : 0 , 'C': 1 , 'Q':2}).astype(int)
    
    
    
    frame.loc[frame.Age <= 16 , 'Age'] = 0
    frame.loc[(frame.Age >16) & (frame.Age<=32) , 'Age'] = 1
    frame.loc[(frame.Age >32) & (frame.Age<=48) , 'Age'] = 2
    frame.loc[(frame.Age >48) & (frame.Age<=64) , 'Age'] = 3
    frame.loc[(frame.Age >64) & (frame.Age<=80) , 'Age'] = 4
    
    
    frame.loc[(frame.Fare <= 7.91) , 'Fare'] = 0
    frame.loc[(frame.Fare > 7.91) & (frame.Fare <= 14.454) , 'Fare'] = 1
    frame.loc[(frame.Fare > 14.454) & (frame.Fare <= 31) , 'Fare'] = 2
    frame.loc[(frame.Fare > 31) , 'Fare'] = 3


# In[ ]:


mapping(df_train)
df_train.head()


# In[ ]:


mapping(df_test)
df_test.head()


# **We can also create an artificial feature combining Pclass and Age.**

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

df_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# # **Now, it's right time to choose best model.**

# In[ ]:


# Importing some algorithms from sklearn.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:


x_train = df_train.drop("Survived", axis=1)
y_train = df_train["Survived"]
x_test=df_test.copy()


# In[ ]:


models = [LogisticRegression(),
        DecisionTreeClassifier(),SVC(),RandomForestClassifier()]

model_names=['LogisticRegression','DecisionTree','SVM','RandomForestClassifier']

accuracy = []


for model in range(len(models)):
    clf = models[model]
    clf.fit(x_train,y_train)
    accuracy.append(round(clf.score(x_train, y_train) * 100, 2))
    
compare = pd.DataFrame({'Algorithm' : model_names , 'Accuracy' : accuracy})
compare


# **We achieved same accuracy(86.64) for both RandomForestClassifier and DecisionTree Classifier. We choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.**

# In[ ]:


RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
y_test = RFC.predict(x_test)


# In[ ]:



d = {'PassengerId' : df_test_copy.PassengerId , 'Survived' : y_test}
answer = pd.DataFrame(d)
# Generate CSV file based on DecisionTree Classifier
answer.to_csv('predio.csv' , index=False)


# # Thank you
# 
# Guys,do put your query in comment section and if you like the implementation method, do upvote it. 
