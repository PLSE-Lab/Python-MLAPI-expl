#!/usr/bin/env python
# coding: utf-8

# # # Simple Logistic with Titanic - 81.3% test score: Quick yet effective For Beginners
# 
# 

# Hi there! 
# I am totally new to data science and machine learning, and it is my very first kernal!
# 
# This kernal is really straightforward yet still get a pretty decent score, currently ranking top8%.
# 
# Feel free to point out any mistakes or places to improve -- I am thrilled to learn more!

# # **Content

# 1. Import libraries and data
# 2. Exploratory data analysis(EDA)
# 3. Data cleaning 
# 4. Feature engineering
# 5. Data preparation for modelling
# 6. Modelling: Logistic regression
# 7. Model evaluation

# # 1. Import libraries and data

# Some common python libraries for data analysis and plotting are imported - you already can tell how straightforward the model is by the number of libraries imported.
# 
# A data_cleaner is created so that it will be more convenient to clean and fill the data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
sns.set() #make the graphs prettier


# In[ ]:


test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

data_cleaner = [train, test]


# # 2. Exploratory data analysis(EDA)

# Since there are already tons of brilliant kernals for EDA about this dataset, so this kernel will not go into much depth. Instead, we will focus more on the missing values and other useful relationships among data.

# Let's first take a look at the train dataset and then learn the correlations between 'Survived' column and other columns 

# In[ ]:


train


# In[ ]:


sns.countplot('Survived',data=train)


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.countplot('Sex',hue='Survived',data=train)


# In[ ]:


sns.countplot('Pclass',hue='Survived',data=train)


# #  3. Data cleaning

# In this dataset, we have some missing values. Some of them only misses a small quantity, such as Embarked, while others are almost completely lost, such as Cabin. 
# Given the situation and for the sake of simplicity, we will fill in reasonable values for those missing a few and just drop others.

# In[ ]:


for data in data_cleaner:
    print(data.isnull().sum())
    print('\n')


# Let's visualize those missing values in a heatmap.

# In[ ]:


for data in data_cleaner:
    plt.figure(figsize=(8,6))
    sns.heatmap(data.isnull(),cmap='viridis')


# First, let's deal with the Age column.
# From the EDA above and boxplot below, we can tell that Age is correlated with Pclass, which makes sense, since most rich people (in first class) tend to be older. So we will fill the age based on the class that person is in.

# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


age_ref = pd.DataFrame(data=[train.groupby('Pclass')['Age'].mean()],columns=train['Pclass'].unique())
age_ref


# In[ ]:


def fill_age(pclass,age):
    if pd.isnull(age):
        return float(age_ref[pclass])
    else:
        return age

for data in data_cleaner:
    data['Age'] = train.apply(lambda x: fill_age(x['Pclass'],x['Age']), axis=1)


# Next, let's deal with Fare and Embark. Since they both only miss a few, so we may simply fill the fare based on the mean and Embark on the mode.

# In[ ]:


def fill_fare(fare):
    if pd.isnull(fare):
        return train['Fare'].mean()
    else:
        return fare
    
def fill_embark(embark):
    if pd.isnull(embark):
        return train['Embarked'].mode().iloc[0]
    else:
        return embark
    
for data in data_cleaner:
    data['Fare'] = train.apply(lambda x: fill_fare(x['Fare']), axis=1)
    data['Embarked'] = train.apply(lambda x: fill_embark(x['Embarked']), axis=1)


# As for the Cabin, it's too much to fill, so we can just remove them all.

# In[ ]:


for data in data_cleaner:
    data.drop(['Cabin'],axis=1,inplace=True)


# In[ ]:


for data in data_cleaner:
    print(data.isnull().sum())
    print('\n')


# Okay, we've roughly finished cleaning our data!

# # 4. Feature engineering

# Revisiting our training data, we will try to extract some features based on the existing coloums, a.k.a, feature engineering.
# 
# **1. Title**
# 
# For the categorial column, it seems that we gain some useful information from the Name column, but for the Ticket column, there is really not much to do(you may explore it if you want, but here, let's just drop it)
# 
# 

# In[ ]:


train


# Taking a closer look at the Name, it can be easily observed that each name has a special title encapsulated within it, which can tell a great deal about a person, like his or hers social status, mattering a lot about hers or his survival.

# In[ ]:


train['Name']


# Let's extract some grab the title out by using the quick and dirty split() method and then add a column called Title.

# In[ ]:


title_list = list()
for data in data_cleaner:
    for title in data['Name']:
        title = title.split('.')[0].split(',')[1]
        title_list.append(title)
    
    data['Title'] = title_list
    title_list = list()


# Some titles only have a handful of occurences, so we can replace them as Other.

# In[ ]:


for data in data_cleaner:
    print(data['Title'].value_counts())
    print('\n')


# In[ ]:


train['Title'] = train['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',
       ' the Countess', ' Jonkheer'], 'Others')
train['Title'].value_counts()


# In[ ]:


test['Title'] = test['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',
       ' the Countess', ' Jonkheer',' Dona'], 'Others')
test['Title'].value_counts()


# **2. Family size**
# 

# It seems that most travellers travel as single.

# In[ ]:


sns.catplot(x="SibSp",kind="count", data=train, height=4.7, aspect=2.45)


# Presented in the visualiztion below, the survival chance of a passenger with 1 or 2 siblings/spouses and 1,2 or 3 parents/children is significantly higher than than for a single passenger or a passenger with a large family.

# In[ ]:


sns.catplot(x="SibSp", y="Survived", kind="bar", data=train, height=4, aspect=3).set_ylabels("Survival Probability")


# In[ ]:


sns.catplot(x="Parch", y='Survived',kind="bar", data=train, height=4.5, aspect=2.5)


# Inspired by such features, we can add another FamilySize column and IsAlone column.

# In[ ]:


def get_size(df):
    if df['SibSp'] + df['Parch'] + 1 == 1:
        return 'Single'
    if df['SibSp'] + df['Parch'] + 1 > 1:
        return 'Small'
    if df['SibSp'] + df['Parch'] + 1 > 4:
        return 'Big'
    
for data in data_cleaner:
    data['FamilySize'] = data.apply(get_size,axis=1)

for data in data_cleaner:
    data['IsAlone'] = 1 
    data['IsAlone'].loc[data['FamilySize'] != 'Single'] = 0


# # 5. Data preparation for modelling

# First, let's convert the categorical data into dummy variables, split the data into train set and test set, and scale the data using MinMaxScaler to feed our model.

# **Get dummies

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
title = pd.get_dummies(train['Title'],drop_first=True)
Pclass = pd.get_dummies(train['Pclass'],drop_first=True)
FamilySize = pd.get_dummies(train['FamilySize'],drop_first=True)

sex2 = pd.get_dummies(test['Sex'],drop_first=True)
embark2 = pd.get_dummies(test['Embarked'],drop_first=True)
title2 = pd.get_dummies(test['Title'],drop_first=True)
Pclass2 = pd.get_dummies(test['Pclass'],drop_first=True)
FamilySize2 = pd.get_dummies(test['FamilySize'],drop_first=True)

for data in data_cleaner:
    data.drop(['Sex','Embarked','Name','Ticket','PassengerId','Title','FamilySize'],axis=1,inplace=True)
    
train = pd.concat([sex,embark,train,title,FamilySize],axis=1)
test = pd.concat([sex2,embark2,test,title2,FamilySize2],axis=1)


# **Train Test Split

# Such process isn't really necessary, since titanic has a relatively small dataset. But we will still include it so that we can better validate our model.

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# **Scale our data
# 
# Remember not to fit our X_test and final test set, in order to prevent data leakage.

# In[ ]:


scaler = MinMaxScaler()

scaler.fit(X_train)

scaler.transform(X_train)
scaler.transform(X_test)
scaler.transform(test)


# # 6. Modelling: Logistic regression

# After trying several different models and even Artificial Neural Network(ANN) with extensive hyperparameter analysis, it turns out that the logistic regression performs the best, which is extremely simple but effective.
# 
# You may consider to tune the parameter of the logistic regression, but for now, we just leave it default.

# In[ ]:


logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)


# # 7. Model evaluation

# For this part, we will evaluate our model using classification report and confusion matrix. Then, let's save our prediction into a csv. file and be ready for the submission!
# 
# As you can see, this model performs amazingly and surprisingly well for such amount of code, about 82% accuracy, which leads us to top 4%!!!

# In[ ]:


print(classification_report(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='plasma')


# In[ ]:


predictions = logistic_model.predict(test)
pred_list = [int(x) for x in predictions]

test2 = pd.read_csv("../input/titanic/test.csv")
output = pd.DataFrame({'PassengerId': test2['PassengerId'], 'Survived': pred_list})
output.to_csv('Titanic_with_logistic.csv', index=False)


# Okay, that's pretty much it.
#  
# Undoubtedly, there are still tons of places to improve, from feature engineering to tuning the parameters. I will really appreciate for any comment below for advice and suggestions, or just a thanks.

# # Please upvote this kernal if you find it helpful!! 
# # Thank you so much!!!
