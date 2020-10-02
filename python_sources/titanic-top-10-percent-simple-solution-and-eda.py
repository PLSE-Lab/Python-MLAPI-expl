#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler

from sklearn.model_selection import  train_test_split , cross_val_score



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import os
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('../input/train.csv' , index_col = 'PassengerId')
label = train['Survived']

test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')
index = test.index


# In[ ]:


train.head(3)


# In[ ]:


train.info()


# > <h3>Survived</h3>
# Target variable for this dataset is Survived . So let us do some analysis on this field first.

# In[ ]:


sns.countplot(label)


# More than half (around 60%) of the passengers died. 

# <h3> Male and Female</h3>

# In[ ]:


fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')
a = sns.countplot(x = 'Sex' , data=train , ax = ax[0] , order=['male' , 'female'])
b = sns.countplot(x = 'Sex' , data= train[label == 1] , ax = ax[1] , order=['male' , 'female'])
c = sns.countplot(x = 'Sex' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=['male' , 'female'])
ax[0].set_title('All passenger')
ax[1].set_title('Survived passenger')
ax[2].set_title('Survived passenger under age 21')


# 1. Majority of passengers were male on Titanic.<br>
# 2. Most of the female survived.<br>

# <h3>Passanger Class</h3>

# In[ ]:


fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')
a = sns.countplot(x = 'Pclass' , data=train , ax = ax[0] , order=[1 ,2,3])
b = sns.countplot(x = 'Pclass' , data= train[label == 1] , ax = ax[1] , order=[1 ,2,3])
c = sns.countplot(x = 'Pclass' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=[1,2,3])
ax[0].set_title('All passanger')
ax[1].set_title('Survived passanger')
ax[2].set_title('Survived passanger under age 21')


# 1. Most of the poor people died (ie From passenger class 3) .
# 2. Most poor people who survived were under age 21

# > <h3>Embarked</h3>

# In[ ]:


fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')
a = sns.countplot(x = 'Embarked' , data=train , ax = ax[0] , order=['S' ,'Q','C'])
b = sns.countplot(x = 'Embarked' , data= train[label == 1] , ax = ax[1] , order=['S' ,'Q','C'])
c = sns.countplot(x = 'Embarked' , data= train[ ((train['Age'] < 21) & (label == 1)) ] , order=['S' ,'Q','C'])
ax[0].set_title('All passanger')
ax[1].set_title('Survived passanger')
ax[2].set_title('Survived passanger under age 21')


# 1. Most people boarded from Southampton since it the starting port of Titanic.
# 2. Most of the people who boarded from Southampton died.

# <h3> Feature Engineering</h3>

# <h4>Deck </h4>
# A deck is a permanent covering over a compartment or a hull of a ship. On a boat or ship, the primary or upper deck is the horizontal structure that forms the "roof" of the hull, strengthening it and serving as the primary working surface.<br>
# It also gives information in which part of the ship a particular passenger might be when the ship was shinking. 
# More information can be found here [here](https://en.wikipedia.org/wiki/RMS_Titanic) in Dimensions and layout section<br>
# We can get this information from the first letter of Cabin name if it not NaN

# In[ ]:


train['Deck'] = train.Cabin.str.get(0)
test['Deck'] = test.Cabin.str.get(0)
train['Deck'] = train['Deck'].fillna('NOTAVL')
test['Deck'] = test['Deck'].fillna('NOTAVL')
#Replacing T deck with closest deck G because there is only one instance of T
train.Deck.replace('T' , 'G' , inplace = True)
train.drop('Cabin' , axis = 1 , inplace =True)
test.drop('Cabin' , axis = 1 , inplace =True)


# <h4>Lets count the missing values in train and test</h4>

# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In training set there is missing value in **Embarked** and **Age**<br>
# In training set there is missing value in **Fare** and **Age**

# <h4>Let's fill the missing values in Embarked with the most frequent value in train set</h4>

# In[ ]:


train.loc[train.Embarked.isna() , 'Embarked'] = 'S'


# <h4>In the above bar graph we saw that Pclass ,sex , Embarked were the determing factor for the servival of a passenger we will group them using these features and fill the median age in the corresponding missing values in the group</h4>

# In[ ]:


age_to_fill = train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()
age_to_fill


# In[ ]:


for cl in range(1,4):
    for sex in ['male' , 'female']:
        for E in ['C' , 'Q' , 'S']:
            filll = pd.to_numeric(age_to_fill.xs(cl).xs(sex).xs(E).Age)
            train.loc[(train.Age.isna() & (train.Pclass == cl) & (train.Sex == sex) 
                    &(train.Embarked == E)) , 'Age'] =filll
            test.loc[(test.Age.isna() & (test.Pclass == cl) & (test.Sex == sex) 
                    &(test.Embarked == E)) , 'Age'] =filll


# Lets check if the above for loop is correct or not.<br>
# There shouldn't be any difference between the previous median of groups and after filling its median in place of NaN

# In[ ]:


train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()


# YAY ! There isn't any difference 

# Fare is string with number at the end , Two consecutive ticket number means they are bougth from same place or they got same deck on the ship...

# In[ ]:


train.Ticket = pd.to_numeric(train.Ticket.str.split().str[-1] , errors='coerce')
test.Ticket = pd.to_numeric(test.Ticket.str.split().str[-1] , errors='coerce')


# Lets fill the missing Ticket value in train data with median Ticket value and one missing fare value in test data with median fare in train

# In[ ]:


Ticket_median = train.Ticket.median()
train.Ticket.fillna(Ticket_median , inplace =True)
test.Fare.fillna(train.Fare.median() , inplace =True)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# Lets create one feature variable **Status** in the society . This features can be derive from the name features like 'Dr' , 'Rev' , 'Col' , 'Major' etc

# In[ ]:


train['Status'] = train['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()
test['Status'] = test['Name'].str.split(',').str.get(1).str.split('.').str.get(0).str.strip()
importan_person = ['Dr' , 'Rev' , 'Col' , 'Major' , 'Mlle' , 'Don' , 'Sir' , 'Ms' , 'Capt' , 'Lady' , 'Mme' , 'the Countess' , 'Jonkheer' , 'Dona'] 
for person in importan_person:
    train.Status.replace(person, 'IMP' , inplace =True)
    test.Status.replace(person, 'IMP' , inplace =True)


# In[ ]:


train.Status.unique()


# In[ ]:


test.Status.unique()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.drop(['Name' , 'Ticket' ] ,axis = 1, inplace = True)
train.drop(['Survived','Ticket' ,'Name' ], inplace =True , axis =1)


# In[ ]:


cat_col = ['Pclass' , 'Sex' , 'Embarked' , 'Status' , 'Deck']
train.Pclass.replace({
    1 :'A' , 2:'B' , 3:'C'
} , inplace =True)
test.Pclass.replace({
    1 :'A' , 2:'B' , 3:'C'
} , inplace =True)
train = pd.get_dummies(train , columns=cat_col)
test = pd.get_dummies(test , columns=cat_col)
print(train.shape , test.shape)


# Lets scale the data

# In[ ]:


scaler = MinMaxScaler()

train= scaler.fit_transform(train)
test = scaler.transform(test)


# <h3>Machine Learning</h3>

# In[ ]:


model = RandomForestClassifier(bootstrap= True , min_samples_leaf= 3, n_estimators = 500 ,
                               min_samples_split = 10, max_features = "sqrt", max_depth= 6)
cross_val_score(model , train , label , cv=5)


# In[ ]:


model = LogisticRegression()
cross_val_score(model , train , label , cv=5)


# In[ ]:


from sklearn.svm import SVC
model = SVC(C=4)
cross_val_score(model , train , label , cv=5)


# In[ ]:


model.fit(train , label)
pre = model.predict(test)


# In[ ]:


ans = pd.DataFrame({'PassengerId' : index , 'Survived': pre})
ans.to_csv('submit.csv', index = False)
ans.head()


# <h3>If you liked this kernel please consider upvoting it.<h3>
