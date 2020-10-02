#!/usr/bin/env python
# coding: utf-8

# <h1>Exploring Titanic the Dataset</h1>
# <autor>by Javier Villarroel</autor>

# <h2>Introduction</h2>
# 
# <p>This is my exploration in use the notebook in Kaggle, and of course in the Titanic Dataset</p>

# In[ ]:


#Libraries to import

import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)


# In[ ]:


# Import dataset 

titanic_train = pandas.read_csv('../input/train.csv')
titanic_test = pandas.read_csv('../input/test.csv')


# In[ ]:


# the train information
titanic_train.info()


# Cabin have 294 non-null values, Age 714  

# In[ ]:


titanic_train.head()


# In[ ]:


titanic_train.describe()


# SibSp and Parch look like don't have many value different of 0 

# In[ ]:


titanic_train['SibSp'].value_counts()


# In[ ]:


titanic_train['Parch'].value_counts()


# #Take a view of the data#

# In[ ]:


titanic_train['Survived'].value_counts().plot(kind='bar', title='Survival Counts')


# In[ ]:


survived = titanic_train['Survived'].value_counts()[0]/titanic_train.shape[0]
death = titanic_train['Survived'].value_counts()[1]/titanic_train.shape[0]
print("{0} {1:0.2f}".format("Survived % = ", survived))
print("{0} {1:0.2f}".format("Dead % = ", death))


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=titanic_train)
sns.plt.title('Survival by Gender')


# Interesting! females survived significantly. The ~75% of females survived, but only ~20% of men

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=titanic_train)
sns.plt.title('Survival by Passenger Class')


# People in first class have more probability to survived

# In[ ]:


titanic_train["Age"].hist(bins=20)
titanic_train[titanic_train['Survived']==1]["Age"].hist(bins=20)
sns.plt.xlabel('Age')
sns.plt.ylabel('Number of persons')
sns.plt.title('Distribution Survival by Age')


# Interesting! there were many kids
# 
# Remember, (891-714) 177 row haven't age values 

# In[ ]:


plt.scatter(titanic_train[titanic_train['Survived']==1]['PassengerId'],
            titanic_train[titanic_train['Survived']==1]['Fare'], c = 'b')
plt.scatter(titanic_train[titanic_train['Survived']==0]['PassengerId'],
            titanic_train[titanic_train['Survived']==0]['Fare'], c = 'r')

plt.xlabel('PassengerId')
plt.ylabel('Fare')
plt.xlim([0,900])
plt.ylim([0,550])
plt.legend(('Survived','Dead'),loc='upper left',fontsize=15,bbox_to_anchor=(1.05, 1))


# Looks like the highest Fare survived, and is very logic. More than 500

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==3)]['Age'],
            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==3)]['Fare'], c = 'r')
axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==2)]['Age'],
            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==2)]['Fare'], c = 'g')
axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==1)]['Age'],
            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==1)]['Fare'], c = 'b')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Fare')
axes[0].set_xlim([0,80])
axes[0].set_ylim([0,300])

axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==3)]['Age'],
            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==3)]['Fare'], c = 'r')
axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==2)]['Age'],
            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==2)]['Fare'], c = 'g')
axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==1)]['Age'],
            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==1)]['Fare'], c = 'b')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Fare')
axes[1].set_xlim([0,80])
axes[1].set_ylim([0,300])
axes[1].legend(('Pclass 3','Pclass 2', 'Pclass 1'),fontsize=15,bbox_to_anchor=(1.05, 1))


# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=titanic_train)
sns.plt.title('Survival by Embarked')


# In a previous study, I didn't find any relevant view in the charts of 'Cabin', 'SibSp' and 'Parch'

# #Feature Engineering#

# ##Dummy features##

# In[ ]:


#Creation of Dummy features for Sex, Embarker, Pclass

def dummy_features(df):
    new_embarked = pandas.get_dummies(df['Embarked'],prefix='Embarked')
    new_sex = pandas.get_dummies(df['Sex'])
    new_Pclass = pandas.get_dummies(df['Pclass'],prefix='Pclass')
    new_df = pandas.concat([new_embarked,new_sex,new_Pclass],axis=1)
    return new_df


# In[ ]:


train = pandas.concat([titanic_train,dummy_features(titanic_train)],axis=1)


# In[ ]:


train = train[train['Fare'] < 450]


# In[ ]:


#train.columns

#All the columns
#col = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Embarked_C',
#       'Embarked_Q', 'Embarked_S', 'female', 'male', 'Pclass_1', 'Pclass_2',
#       'Pclass_3']
#
#New columns

col = ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 
       'female', 'male', 
       'Pclass_1', 'Pclass_2', 'Pclass_3']


#  "col" is the columns  that I want to use for my models

# #Take a view on Stadistic

# In[ ]:


corr = train[train['Age']==train['Age']][col].corr()
corr.style.format("{:.2f}")


# looks like there is correlations (>|0.1|) between: 
# 
# - Survived and Fare, Embarked_C, Embarked_S, female, male, Pclass_1, Pclass_3
# - Age and  Fare, SibSp, Parch, Pclass_1, Pclass_3
#  - It is logic that if you are young, your traveling with familiars, so Age, SibSp and Parch are correlated
# - Fare and Sex, Embarked_C and S, Pclass_1 and 3
#  - It logic thar the price of the ticket are correlated with where people embarked, what class they buy, how many people trabvel together and the gender 100 year ago. So, this features could be reduced.
# 
# I thinks that PassengerId, SibSp and Parch aren't relevant, Embarked_Q and Pclass_2 neither. but Age doesn't look relevant but this is not very logic for me.

# #Predictions#

# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.metrics import accuracy_score


# ##first look##

# In[ ]:


cls = []
cls.append(LogisticRegression()) 
cls.append(SVC())
cls.append(RandomForestClassifier())
cls.append(KNeighborsClassifier())
cls.append(GaussianNB())
cls.append(tree.DecisionTreeClassifier())


# In[ ]:


col = ['Fare',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 
       'female', 'male', 
       'Pclass_1', 'Pclass_2', 'Pclass_3']


# In[ ]:


limit = int(0.8 * train.shape[0])
X_train = train[col].iloc[:limit,:]
X_test = train[col].iloc[limit:,:]
y_test = train['Survived'].iloc[limit:]
y_train = train['Survived'].iloc[:limit]


# In[ ]:


for cl in cls:
    cl.fit(X_train,y_train)
    print(cl)
    print("score: ", accuracy_score(y_test,cl.predict(X_test)))


# Decision Trees show the highest score 0.84

# I will try submit for example the Decision Trees like a test. 

# In[ ]:


test = pandas.concat([titanic_test,dummy_features(titanic_test)],axis=1)


# In[ ]:


#Refit the classifier with all the train set

cls[2].fit(train[col],train['Survived'])


# Before the prediction, we need to manage one NaN value in 'Fare'

# In[ ]:


test[test['Fare']!=test['Fare']]


# In[ ]:


#I looking for a person with the same characteristic in train data set
train[(train['Pclass']==3)&(train['Embarked']=='S')&(train['male']==1.0)&(train['Age']>60)]


# In[ ]:


test['Fare'] = test['Fare'].fillna(6.24)


# In[ ]:


prediction = cls[2].predict(test[col])


# I look in the test the person with more than 450 in Fare

# In[ ]:


test[test['Fare']>400]


# In[ ]:


#I check that she is alive
prediction[343]


# In[ ]:


submission = pandas.DataFrame({'PassengerId': test['PassengerId'],
                               'Survived': prediction})


# In[ ]:


submission.to_csv("submission_0.csv",index=False)


# In[ ]:




