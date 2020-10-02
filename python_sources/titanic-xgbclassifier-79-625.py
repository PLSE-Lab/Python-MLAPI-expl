#!/usr/bin/env python
# coding: utf-8

# # You can use this notebook as a reference for feature engineering and imputing missing values
# 
# **If you like it, do upvote it**

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor,XGBClassifier
#from sklearn.impute import SimpleImputer


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# combined_products = pd.concat([gaming_products, movie_products])
# if to combine two dataset refer above code_line






# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
train_data


# In[ ]:


test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data


# **Combining test and train data for better imputing**

# In[ ]:


train_results = train_data["Survived"].copy()
train_data.drop("Survived", axis=1, inplace=True, errors="ignore")
titanic = pd.concat([train_data, test_data])
traindex = train_data.index
testdex = test_data.index


# In[ ]:


titanic


# In[ ]:


titanic[titanic['Cabin']=='B51 B53 B55']
#the 2nd person is probably a false one


# Reindexing the data

# In[ ]:


titanic.index=range(len(titanic))


# In[ ]:


titanic


# In[ ]:


# titanic=titanic.drop(['Cabin'],axis=1)
#cabin has 600+ null entries , so you can drop it as it increases noise


# In[ ]:


titanic.info()


# Lets fill only missing value in Fare

# In[ ]:


titanic[pd.isnull(titanic['Fare'])]


# In[ ]:


mean=titanic[titanic['Pclass']==3][titanic['Embarked']=='S'][titanic['Sex']=='male'][titanic['Age']>=40][titanic['SibSp']==0][titanic['Parch']==0]
mean['Fare'].describe()


# In[ ]:


titanic[['Fare']]=titanic[['Fare']].fillna(value=7.69)


# In[ ]:


titanic.isnull().sum()


# Extracting Title from Name

# In[ ]:




titanic["Title"] = titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic['Title'], titanic['Sex'])


# Optimizing the Title column

# In[ ]:




titanic['Title'] = titanic['Title'].replace('Mlle', 'Miss')
titanic['Title'] = titanic['Title'].replace('Ms', 'Miss')
titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')


# In[ ]:


titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess','Capt','Col','Don', 'Dr', 'Major','Rev', 'Sir', 'Jonkheer', 'Dona'], 'Not married')


# In[ ]:


titanic['Title'] = titanic['Title'].replace(['Mr', 'Mrs'], 'Married')


# In[ ]:


pd.crosstab(titanic['Title'], titanic['Sex'])


# In[ ]:


titanic


# Extracting Surname from name column

# In[ ]:


titanic["Surname"] = titanic.Name.str.split(',').str.get(0)


# In[ ]:


titanic


# Droping name column as we have extracted all important information

# In[ ]:


titanic=titanic.drop(['Name'],axis=1)


# In[ ]:


titanic


# In[ ]:


titanic.Surname.value_counts()


# In[ ]:


titanic['SurnameFreq']=titanic.groupby('Surname')['Surname'].transform('count')


# In[ ]:


titanic


# Lets create a new column Customized Fare as the there are duplicate tickets present in the data. So a person has bought tickets for his family also.

# In[ ]:


titanic.Ticket.value_counts()
#unique Values


# Obtaining TicketFreq

# In[ ]:


titanic['TicketFreq']=titanic.groupby('Ticket')['Ticket'].transform('count')


# In[ ]:


titanic


# I have also used Pclass for customized fare to optimize it

# In[ ]:


titanic['customizedFare']=titanic.Fare/(titanic.TicketFreq*titanic.Pclass)


# In[ ]:


titanic


# In[ ]:


titanic.isnull().sum()


# In[ ]:


titanic.Age.describe()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(titanic.corr(),annot=True)


# **Now Let us deal with missing values in Age column**
# As majority of kids and female survived so I'll break age column in kids, females and males section for dealinng with null values

# In[ ]:


titanic[titanic['Title']=='Boy'].Age.median()


# Assigning median of children to empty age values of title=master or kids

# In[ ]:


titanic.loc[(titanic.Age.isnull()) & (titanic.Title=='Master'),'Age']=int(4.0)


# In[ ]:


titanic.loc[(titanic.Age.isnull()) & (titanic.Sex=='male'),'Age']=int(30.0)


# In[ ]:


titanic.loc[(titanic.Age.isnull()) & (titanic.Sex=='female'),'Age']=int(27.0)


# In[ ]:


# age_by_pclass_sex = titanic.groupby(['Sex', 'Pclass']).median()['Age']

# for pclass in range(1, 4):
#     for sex in ['female', 'male']:
#         print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
# print('Median age of all passengers: {}'.format(titanic['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
# titanic['Age'] = titanic.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


titanic.info()


# In[ ]:


titanic.isnull().sum()


# Find median of age of females

# In[ ]:


titanic['Embarked'].fillna('S',inplace=True)

#to rename columns refer below
#renamed = reviews.rename(columns=dict(region_1='region', region_2='locale'))


# In[ ]:


titanic.isnull().sum()


# Combining Sibsp and Parch to Family

# In[ ]:


titanic['Family']=titanic['SibSp']+titanic['Parch']+1
titanic=titanic.drop(['SibSp','Parch'],axis=1)


# In[ ]:


titanic


# In[ ]:


titanic.SurnameFreq=titanic.TicketFreq


# In[ ]:


titanic


# In[ ]:


# titanic['Deck'] = titanic['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
# idx = titanic[titanic['Deck'] == 'T'].index
# titanic.loc[idx, 'Deck'] = 'A'


# In[ ]:


# titanic['Deck'] = titanic['Deck'].replace(['A', 'B', 'C'], 'ABC')
# titanic['Deck'] = titanic['Deck'].replace(['D', 'E'], 'DE')
# titanic['Deck'] = titanic['Deck'].replace(['F', 'G'], 'FG')
titanic['Deck']=titanic['Cabin'].notnull().astype(str).str[0]
titanic['Deck'].value_counts()


# In[ ]:


titanic=titanic.drop(['Cabin'],axis=1)


# In[ ]:


titanic.info()


# In[ ]:


titanic


# In[ ]:




# titanic['Age'] = pd.qcut(titanic['Age'], 10)


# Engineering another feature Familygroup to optimize data

# In[ ]:


def FamilyGroup(family):
    a=''
    if family<=1:
        a='Single'
    elif family<=4:
        a='Small'
    else:
        a='Large'
    return a
titanic['FamilyGroup']=titanic['Family'].map(FamilyGroup)
titanic=titanic.drop(['Family'],axis=1)  


# In[ ]:


# titanic['customizedFare'] = pd.qcut(titanic['customizedFare'], 13)


# In[ ]:


titanic.head()


# In[ ]:


titanic


# In[ ]:


def AgeGroup(age):
    a=''
    if age<=15:
        a='Child'
    elif age<=30:
        a='Young'
    elif age<=50:
        a='Adult'
    else:
        a='Old'
    return a
titanic['AgeGroup']=titanic['Age'].map(AgeGroup)
titanic=titanic.drop(['Age'],axis=1)  


# In[ ]:


titanic


# **Handling Categorical Data**

# In[ ]:


titanic.corr()


# In[ ]:


titanic=titanic.drop(['PassengerId','TicketFreq','Ticket','Fare','Title','Surname'], axis=1)


# In[ ]:


titanic_data=pd.get_dummies(titanic,columns=['Embarked','AgeGroup','Sex','Deck','FamilyGroup'])


# In[ ]:


titanic_data.head()


# In[ ]:


print(traindex)


# In[ ]:


print(testdex)


# In[ ]:


titanic_data.shape


# **Dividing total dataset into train and test*

# In[ ]:


titanic_data.loc[0:890]


# In[ ]:


titanic_data.loc[891:1308]


# In[ ]:


# Train
train_df = titanic_data.loc[0:890]
train_df['Survived'] = train_results

# Test
test_df = titanic_data.loc[891:1308]


# In[ ]:


train_df.corr()


# In[ ]:



print(train_df)


# In[ ]:


print(test_df)


# **Modeling Part**

# In[ ]:


X=train_df.drop(['Survived'],axis=1)
X.head()
y=train_df.Survived
y.head()


# In[ ]:





# In[ ]:


X.head()


# In[ ]:


# X_train , X_test, y_train, y_test=train_test_split(X,y, test_size=0.3,random_state=1)

#use it to test your model on training data


# In[ ]:


xgbr=XGBClassifier(n_estimators=2800,
    min_child_weight=0.1,
    learning_rate=0.002,
    max_depth=2,
    subsample=0.47,
    colsample_bytree=0.35,
    gamma=0.4,
    reg_lambda=0.4,
    random_state=42,
    n_jobs=-1,)
xgbr.fit(X,y)

predicts=xgbr.predict(test_df)


# In[ ]:






submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predicts
    })
submission.Survived = submission.Survived.round().astype("int")
submission.to_csv('titanic.csv', index=False)
print("Submitted Successfully")



