#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#data processing

import numpy as np
import pandas as pd

#data visualization

import matplotlib.pyplot as plt
import seaborn as sns

#data predications

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[2]:


database = "../input/train.csv"
titanic = pd.read_csv(database)


# In[3]:


titanic.columns


# In[4]:


titanic.head()


# # Plotting and checking
# 

# In[5]:


#for getting data of surviving people
sns.countplot(x = 'Survived' ,data = titanic)


# In[6]:


sns.countplot(x = 'Survived' , hue = 'Sex' , data = titanic)


# In[7]:


sns.countplot( x = 'Survived' , hue = 'Pclass' , data= titanic)


# In[8]:


sns.distplot(titanic['Age'].dropna(),bins = 50)


# In[9]:


titanic['Fare'].hist(bins = 50)


# In[10]:


titanic['Age'].hist(bins=50)


# In[11]:


#Cleaning the Data


# In[12]:


titanic.columns


# In[13]:


titanic.tail()


# In[14]:


#sns.boxplot(x='SibSp',y='Age',data=titanic,palette='winter')
sns.boxplot(x = 'Pclass' , y = 'Age' , data = titanic)


# In[15]:


#sns.lmplot(x='Age', y='Parch', data=titanic)
#i am doing this according to answer but i will trying again
def defineage(cols):
    age= cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        elif pclass==3:
            return 24
    else:
        return age


# In[16]:


def definecatage(cols):
    age = cols[0]
    if age < 5:
        return 0
    elif age < 10:
        return 1
    elif age < 15:
        return 2
    elif age < 20:
        return 3
    elif age < 25:
        return 4
    elif age < 30:
        return 5
    elif age < 35:
        return 6
    elif age < 40:
        return 7
    elif age < 45:
        return 8
    elif age < 50:
        return 9
    elif age < 55:
        return 10
    elif age < 60:
        return 11
    elif age < 65:
        return 12
    elif age < 70:
        return 13
    elif age < 75:
        return 14
    elif age < 80:
        return 15
    else:
        return 16


# In[17]:


titanic['Age'] = titanic[['Age' , 'Pclass']].apply(defineage , axis = 1)


# In[18]:


titanic.info()


# In[19]:


def defcabin(cols):
    cabin = cols[0]
    if type(cabin)==str:
        return 1
    else:
        return 0


# In[20]:


titanic['Cabin'] = titanic[['Cabin']].apply(defcabin , axis = 1)


# In[21]:


titanic['Embarked'].unique()


# In[22]:


titanic_new = titanic.dropna(axis=0)


# In[23]:


titanic_new.info()


# In[24]:


sex = pd.get_dummies(titanic_new['Sex'],drop_first = True)
embarked = pd.get_dummies(titanic_new['Embarked'], drop_first = True)
titanic_new.drop(['Sex','Embarked', 'Ticket', 'Name'],axis =1,inplace= True)


# In[25]:


titanic_new = pd.concat([titanic_new,sex,embarked] ,axis =1)


# In[26]:


titanic_new['Age'] = titanic_new[['Age']].apply(definecatage , axis = 1)


# In[27]:


titanic_new.columns


# In[28]:


t_train, t_test ,s_train ,s_test = train_test_split(titanic_new.drop('Survived',axis=1),titanic_new['Survived'] , test_size = 0.20 , random_state=101)


# In[29]:


model = LogisticRegression()
model.fit(t_train , s_train)
#model.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])


# In[30]:


pred = model.predict(t_test)


# In[31]:


print(accuracy_score(s_test, pred))


# In[32]:


#Trying tree Model 
from sklearn import tree
generalized_tree = tree.DecisionTreeClassifier(random_state =70 , max_depth =7 , min_samples_split = 4)
#generalized_tree = tree.DecisionTreeClassifier()
#generalized_tree.fit(t_train , s_train )
generalized_tree.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])


# In[33]:


pred_tree = generalized_tree.predict(t_test)
print(accuracy_score(s_test, pred_tree))


# In[34]:


rfc = RandomForestClassifier(n_estimators = 50 , bootstrap=False, min_samples_leaf = 4 )
rfc.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])
#rfc.fit(t_train,s_train)
#round(rfc.score(t_test , s_test)*100 , 2)


# In[ ]:





# In[35]:


test_database = "../input/test.csv"
test_titanic = pd.read_csv(test_database)


# In[36]:


#test_titanic_new.isna().sum()
test_titanic.describe()


# In[37]:


test_titanic['Age'] = test_titanic[['Age' , 'Pclass']].apply(defineage , axis = 1)
test_titanic['Cabin'] = test_titanic[['Cabin']].apply(defcabin , axis = 1)


# In[38]:


test_titanic.describe()


# In[39]:


#test_titanic_new["Fare"] = test_titanic["Fare"].fillna((test_titanic["Fare"].median()),inplace= True)
#test_titanic_new = test_titanic.dropna(axis=0)
test_titanic['Fare'] = test_titanic['Fare'].fillna(test_titanic['Fare'].median())
test_titanic.describe()


# In[40]:


sex = pd.get_dummies(test_titanic['Sex'],drop_first = True)
embarked = pd.get_dummies(test_titanic['Embarked'], drop_first = True)
test_titanic.drop(['Sex','Embarked', 'Ticket', 'Name'],axis =1,inplace= True)


# In[41]:


test_titanic = pd.concat([test_titanic,sex,embarked] ,axis =1)
test_titanic['Age'] = test_titanic[['Age']].apply(definecatage , axis = 1)


# In[42]:


test_titanic.describe()


# In[43]:


answer = model.predict(test_titanic)


# In[44]:


answer = rfc.predict(test_titanic)


# In[45]:


#print(accuracy_score(answer, test_titanic_new['Survived']))


# In[46]:


answer_tree = generalized_tree.predict(test_titanic)
#print(accuracy_score(answer_tree, test_titanic_new['Survived']))


# In[47]:


output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],
                        'Survived': answer})


# In[48]:


output.to_csv('submission.csv', index=False)


# In[49]:


output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],
                        'Survived': answer_tree})
output.to_csv('submission_tree.csv', index=False)


# In[50]:


output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],
                        'Survived': answer})
output.to_csv('submission_randomforest.csv', index=False)


# In[ ]:




