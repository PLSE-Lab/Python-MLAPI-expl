#!/usr/bin/env python
# coding: utf-8

# **load modules**

# In[ ]:


import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# **load data**

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# **join sets**

# In[ ]:


train['type_set'] = 'train'
test['type_set'] = 'test'
sets = pd.concat([train,test],sort=False)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
print('sets.shape:', sets.shape)
sets.head()


# In[ ]:


sets.info()


# **there're voids in Age and Cabin and 1 in Fare. we need to fill it**

# In[ ]:


sets[sets.Embarked.isnull()==True]


# **let's set the most frequent value**

# In[ ]:


sets.Embarked.value_counts()


# In[ ]:


sets.Embarked.fillna('S', inplace = True)
sets[sets.Embarked.isnull()==True]


# In[ ]:


sets[sets.Fare.isnull()==True]


# **let's see average fare in 3 Pclass**

# In[ ]:


sets[sets.Pclass == 3].mean()['Fare']


# **set this value**

# In[ ]:


sets.Fare.fillna(13.30, inplace = True)


# **we have a lot missing value in Cabin.
# Maybe passengers didn't have cabin.
# let's see surviving, but first fill null with 'N'**

# In[ ]:


sets.Cabin.fillna('N', inplace = True)
sets.Cabin.unique()


# **separate type cabin from number**

# In[ ]:


sets['Cabin_type'] = [i[0] for i in sets.Cabin]
sets.Cabin_type.unique()


# **let's see surviving**

# In[ ]:


a_table = sets[['Survived','Cabin_type']].groupby(['Cabin_type']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **so, withuot cabin average surviving lower.
# Make new feature Has_cabin**

# In[ ]:


def feature(f):
    if f == 'N': return 0
    else: return 1
sets['Has_cabin'] = sets['Cabin_type'].apply(lambda x: feature(x))

a_table = sets[['Survived','Has_cabin']].groupby(['Has_cabin']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **last feature with missing  value is age.
# Let's see depends age and Survived.
# group first**

# In[ ]:


def feature(f):
    if f < 8: return 'infant'
    #if f < 18: return 'child'
    if f < 24: return 'young'
    #if f < 45: return 'Adult'
    if f < 60: return 'senior'
    if f >= 60: return 'old'    

sets['Age_group'] = sets['Age'].apply(lambda x: feature(x))

a_table = sets[['Survived','Age_group']].groupby(['Age_group']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **OK, Age is vary impotent, we need to fill it careful.
# Let's see names of passengers
# Get the title from the name**

# In[ ]:


sets["title"] = [i.split('.')[0] for i in sets.Name]
sets["title"] = [i.split(', ')[1] for i in sets.title]


# In[ ]:


sets["title"].value_counts()


# **let's see average age**

# In[ ]:


av_age = sets[['title','Age']].groupby(['title']).agg(['mean', 'count'])
av_age.columns = av_age.columns.droplevel(0)
av_age.reset_index(inplace = True)
av_age.sort_values(['count'], ascending=False)


# **lets see how many empty Age values groupping by title**

# In[ ]:


nul_age = sets[(sets.Age.isnull()==True)]

a_table = nul_age[['title','PassengerId']].groupby(['title']).agg(['count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.reset_index(inplace = True)
a_table.sort_values(['count'], ascending=False)


# **fill a empty Age values average Age in group**

# In[ ]:


for i in list(a_table.title.unique()):
    m1 = (sets['title'] == i) & (sets['Age'].isnull()==True)
    sets.loc[m1,'Age'] = sets.loc[m1,'Age'].fillna(round(av_age[av_age.title == i]['mean'].iloc[0]))


# **rewright Age_group feature**

# In[ ]:


sets['Age_group'] = sets['Age'].apply(lambda x: feature(x))


# In[ ]:


sets.info()


# **OK, all voids filled. let's see other feature**

# In[ ]:


a_table = sets[['Survived','Pclass']].groupby(['Pclass']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Pclass very paverfull feature.**
# 
# **We can't group Name, maybe len of Name have value**

# In[ ]:


sets['len_name'] = [len(i) for i in sets.Name]

def feature(x):
    if (x < 20): return 'short'
    if (x < 27): return 'medium'
    if (x < 32): return 'good'
    else: return 'long'

sets['len_name_g'] = sets['len_name'].apply(lambda x: feature(x))


a_table = sets[['Survived','len_name_g']].groupby(['len_name_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **very interesting depending fron len Name**
# 
# **Sex**

# In[ ]:


a_table = sets[['Survived','Sex']].groupby(['Sex']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Sex is too strong feature.
# Maybe better separate all data on two sets female and infant and others**
# 
# **SibSp**

# In[ ]:


a_table = sets[['Survived','SibSp']].groupby(['SibSp']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **good enough. let's group it**

# In[ ]:


def feature(x):
    if (x < 1): return '0'
    if (x < 2): return '1'
    #if (x < 32): return '1+'
    else: return '1+'

sets['SibSp_g'] = sets['SibSp'].apply(lambda x: feature(x))

a_table = sets[['Survived','SibSp_g']].groupby(['SibSp_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Parch**

# In[ ]:


a_table = sets[['Survived','Parch']].groupby(['Parch']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **good enough. let's group it**

# In[ ]:


def feature(x):
    if (x < 1): return '0'
    else: return '1+'

sets['Parch_g'] = sets['Parch'].apply(lambda x: feature(x))

a_table = sets[['Survived','Parch_g']].groupby(['Parch_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Ticket**

# In[ ]:


#sets.Ticket.unique()


# **maybe try len**

# In[ ]:


sets['len_Ticket'] = [len(i) for i in sets.Ticket]

a_table = sets[['Survived','len_Ticket']].groupby(['len_Ticket']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['len_Ticket'], ascending=False)


# **it seems len = 5 and 8 is better **

# In[ ]:


def feature(x):
    if (x == 5 or x == 5): return '5and8'
    else: return 'other'

sets['len_Ticket_g'] = sets['len_Ticket'].apply(lambda x: feature(x))


a_table = sets[['Survived','len_Ticket_g']].groupby(['len_Ticket_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **I dont like this groupping, its can be selection and our model will overfit**

# In[ ]:


sets['Ticket_type'] = [i.split()[0] for i in sets.Ticket]
sets['Ticket_type'].value_counts()

a_table = sets[['Survived','Ticket_type']].groupby(['Ticket_type']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['count'], ascending=False).head(10)


# **if Ticket_type is PC it is mach better**

# In[ ]:


def feature(x):
    if (x == 'PC'): return 'PC'
    else: return 'other'

sets['Ticket_type_g'] = sets['Ticket_type'].apply(lambda x: feature(x))


a_table = sets[['Survived','Ticket_type_g']].groupby(['Ticket_type_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Fare. grouppin first**

# In[ ]:


def feature(x):
    if (x < 10): return 'under10'
    if (x < 50): return '10-50'
    else: return '50+'

sets['Fare_g'] = sets['Fare'].apply(lambda x: feature(x))


a_table = sets[['Survived','Fare_g']].groupby(['Fare_g']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Embarked**

# In[ ]:


a_table = sets[['Survived','Embarked']].groupby(['Embarked']).agg(['mean', 'count'])
a_table.columns = a_table.columns.droplevel(0)
a_table.sort_values(['mean'], ascending=False)


# **Embarked no so good.
# That is all. Lets see what feature we generate**

# In[ ]:


list(sets)


# **to build model we try use groupping feature:**

# In[ ]:


train_set = sets[sets.type_set == 'train']
train_set.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)
print(train_set.shape)
train_set.head() 


# In[ ]:


list(train_set)


# **make dummies**

# In[ ]:


train_set = pd.get_dummies(train_set, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)


# In[ ]:


train_set.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

y = train_set["Survived"]
X = train_set.drop(['Survived'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)


# In[ ]:


logreg = LogisticRegression()

# fit the model with "train_x" and "train_y"
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

round(accuracy_score(y_pred, y_test),4)


# **Do the same on test set:**

# In[ ]:


test_set = sets[sets.type_set == 'test']
test_set.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)
test_set = pd.get_dummies(test_set, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)


# **refit model on all train set**

# In[ ]:


logreg.fit(X,y)

#predict test set
set_pred = logreg.predict_proba(test_set)[:, 1]


# **save result**

# In[ ]:


submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':set_pred})
submission.to_csv('submission.csv')


# **this model give us 0.78468 score and 4129 place from 11185 (top 37%).**
# ![image.png](attachment:image.png)

# **let's separate sets on women and children and all others**

# In[ ]:


sets_W_I = sets.loc[(sets.Sex == 'female') | (sets.Age_group == 'infant')]
sets_Otr = sets.loc[(sets.Sex != 'female') & (sets.Age_group != 'infant')]


# In[ ]:


sets_W_I["Survived"].mean()


# In[ ]:


sets_Otr["Survived"].mean()


# **build 2 model for this set**

# In[ ]:


#sets_W_I
train_set_W_I = sets_W_I[sets_W_I.type_set == 'train']
train_set_W_I.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

train_set_W_I = pd.get_dummies(train_set_W_I, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)

y = train_set_W_I["Survived"]
X = train_set_W_I.drop(['Survived'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)

logreg_W_I = LogisticRegression()

# fit the model with "train_x" and "train_y"
logreg_W_I.fit(X_train,y_train)

y_pred = logreg_W_I.predict(X_test)

round(accuracy_score(y_pred, y_test),4)


# **refit model on all train set**

# In[ ]:


logreg_W_I.fit(X,y)


# In[ ]:


#sets_Otr
train_set_Otr = sets_Otr[sets_Otr.type_set == 'train']
train_set_Otr.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

train_set_Otr = pd.get_dummies(train_set_Otr, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)

y = train_set_Otr["Survived"]
X = train_set_Otr.drop(['Survived'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)

logreg_Otr = LogisticRegression()

# fit the model with "train_x" and "train_y"
logreg_Otr.fit(X_train,y_train)

y_pred = logreg_Otr.predict(X_test)

round(accuracy_score(y_pred, y_test),4)


# **refit model on all train set**

# In[ ]:


logreg_Otr.fit(X,y)


# **preditc test sets and save results**

# In[ ]:


#sets_W_I
test_set_W_I = sets_W_I[sets_W_I.type_set == 'test']
test_set_W_I.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)
test_set_W_I = pd.get_dummies(test_set_W_I, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)

#sets_Otr
test_set_Otr = sets_Otr[sets_Otr.type_set == 'test']
test_set_Otr.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',
               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)
test_set_Otr = pd.get_dummies(test_set_Otr, columns=[ 'Pclass',
 'Sex',
 'Embarked',
 'Has_cabin',
 'Age_group',
 'len_name_g',
 'SibSp_g',
 'Parch_g',
 'len_Ticket_g',
 'Ticket_type_g',
 'Fare_g'], drop_first=True)


#predict test sets
set_pred_W_I = logreg_W_I.predict_proba(test_set_W_I)[:, 1]
set_pred_Otr = logreg_Otr.predict_proba(test_set_Otr)[:, 1]


# In[ ]:


submission_W_I = pd.DataFrame({'PassengerId': sets_W_I[sets_W_I.type_set == 'test']['PassengerId'], 'Survived':set_pred_W_I})
submission_Otr = pd.DataFrame({'PassengerId': sets_Otr[sets_Otr.type_set == 'test']['PassengerId'], 'Survived':set_pred_Otr})

submission_2models = pd.concat([submission_W_I, submission_Otr], ignore_index=True)

submission_2models.to_csv('submission_2models.csv')


# the result is not better ![image.png](attachment:image.png)
# and this with more then 0.70 probability
