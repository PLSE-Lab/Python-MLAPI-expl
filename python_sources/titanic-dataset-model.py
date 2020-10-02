#!/usr/bin/env python
# coding: utf-8

# # Titanic
# 

# #### The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

# ## Our challenge 

# We have to analyse what sorts of people were likely to survive. In particular, we have to apply the tools of machine learning to predict which passengers survived the tragedy.

# # Importing Libraries

# In[ ]:


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
import seaborn as sns


# In[ ]:


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# # Loading our dataset

# In[ ]:


train_data=pd.read_csv('../input/titanic/train.csv')
train_data.set_index('PassengerId',inplace=True)
test_data=pd.read_csv('../input/titanic/test.csv')
train_data.head()


# # Cleaning the dataset

# In[ ]:


#checking if there is missing value in training and testing dataset
print('Missing values in training dataset:\n')
print(train_data.isnull().sum())
print('\nMissing values in testing dataset:\n') 
print(test_data.isnull().sum())      


# ## Treating missing values

# In[ ]:


# for training data
imp=Imputer(missing_values='NaN',strategy='median',axis=1)
Age_train=imp.fit_transform(train_data.Age.values.reshape(1,-1))
Age_train=Age_train.astype('int64').T
train_data['Age']=Age_train
train_data['Embarked'].fillna('S',inplace=True)
train_data.drop('Cabin',axis=1,inplace=True)
train_data.isnull().sum()


# In[ ]:


# for testing data
Age_test=imp.fit_transform(test_data.Age.values.reshape(1,-1))
Age_test=Age_test.astype('int64').T
test_data['Age']=Age_test
test_data['Fare'].fillna(np.mean(test_data['Fare']),inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)
test_data.isnull().sum()


# In[ ]:


# converting sex column to categorical data
for sex in train_data.Sex:
    if sex=='female':
         train_data.replace(sex,0,inplace=True)
    else:
         train_data.replace(sex,1,inplace=True)
for sex in test_data.Sex:
    if sex=='female':
         test_data.replace(sex,0,inplace=True)
    else:
         test_data.replace(sex,1,inplace=True)    


# # Plotting Survived distributions

# In[ ]:


plt.figure(figsize=(10,7))
ax1=sns.countplot(x='Sex',hue='Survived',data=train_data,linewidth=2,edgecolor=(0,0,0),color='g')
plt.title("Survived/Not Survived distribution according to gender",fontsize=20)
labels=['female','male']
plt.xticks(sorted(train_data.Survived.unique()),labels,fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel('Sex',fontsize=20)
plt.ylabel('No. of passengers',fontsize=20)
leg=ax1.get_legend()
leg.set_title('Survived')
leg.texts[0].set_text('No')
leg.texts[1].set_text('Yes')


# In[ ]:


plt.figure(figsize=(10,7))
ax2=sns.countplot(x='Pclass',hue='Survived',data=train_data,edgecolor=(0,0,0),linewidth=2,color='r')
plt.title('Survived/Not Survived according to Class',fontsize=20)
plt.xlabel('Passenger\'s Class',fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('No.of Passengers',fontsize=20)
plt.yticks(fontsize=15)
leg=ax2.get_legend()
leg.texts[0].set_text('No')
leg.texts[1].set_text('Yes')


# ### From above graph we can see that survial is directly depend on the Class.
# 
# 1st class passenger survived percentage is ~63%
# 
# 2nd class passenger survived percentage is ~47%
# 
# 3rd class passenger survived percentage is only ~24%
# 

# # Feature Engineering

# In[ ]:


train_data.drop(['Name','Ticket'],axis=1,inplace=True)
test_data.drop(['Name','Ticket'],axis=1,inplace=True)
train_data.head()


# In[ ]:


# creating age group feature
def age_group(age):
    a=''
    if (age<=1):
        a='infant'
    elif (1<age<=10):
        a='child'
    elif (10<age<=17):
        a='teenager'
    elif (17<age<=30):
        a='young_adult'
    elif (30<age<=40):
        a='adult'    
    else:
        a='old'
    return a
train_data['Age_group']=train_data.Age.map(age_group)
test_data['Age_group']=test_data.Age.map(age_group)


# In[ ]:


# creating family group feature
train_data['family_size']=train_data['SibSp']+train_data['Parch']+1
test_data['family_size']=test_data['SibSp']+test_data['Parch']+1
def family_group(family_size):
    a=''
    if family_size<=1:
        a='alone'
    elif family_size<4:
        a='small'
    else:
        a='large'
    return a
train_data['family_group']=train_data.family_size.map(family_group)
test_data['family_group']=test_data.family_size.map(family_group)


# In[ ]:


# creating fare_per_person column
train_data['fare_per_person']=train_data['Fare']/train_data['family_size']
test_data['fare_per_person']=test_data['Fare']/test_data['family_size']
# creating fare_group feature
def fare_group(fare):
    a=''
    if fare<10:
        a='very_low'
    elif fare<20:
        a='low'
    elif fare<30:
        a='medium'
    elif fare<40:
        a='high'
    else:
        a='very high'
    return a    
train_data['fare_group']=train_data.fare_per_person.map(fare_group)      
test_data['fare_group']=test_data.fare_per_person.map(fare_group)        


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ## Creating dummy variables

# In[ ]:


# creating dummy variables for Embarked,age_group, family_group 
train_data=pd.get_dummies(train_data,columns=['Embarked','Age_group','family_group','fare_group'],drop_first=True)
train_data.head()
test_data=pd.get_dummies(test_data,columns=['Embarked','Age_group','family_group','fare_group'],drop_first=True)


# In[ ]:


# drop unnecessary columns
train_data.drop(['Age','SibSp','Parch','Fare','family_size','fare_per_person'],axis=1,inplace=True)
test_data.drop(['Age','SibSp','Parch','Fare','family_size','fare_per_person'],axis=1,inplace=True)


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# ### Creating Features and Labels

# In[ ]:


X_train=train_data.drop('Survived',1)
y_train=train_data['Survived']
X_test=test_data.drop('PassengerId',1)
X_train.shape


# # Calculating cross_val_score

# ### We are calculating cross_val_score for following classifiying models

# KNeighborsClassifier
# 
# GaussianNB
# 
# LogisticRegression
# 
# DecisionTreeClassifier
# 
# RandomForestClassifier
# 
# SVC
# 
# 

# ## KNeighborsClassifier

# In[ ]:


clf=KNeighborsClassifier(n_neighbors=3)
scoring='accuracy'
score_1=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_1=round(np.mean(score_1)*100,2)
score_1


# ## GaussianNB

# In[ ]:


clf=GaussianNB()
scoring='accuracy'
score_2=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_2=round(np.mean(score_2)*100,2)
score_2


# ## LogisticRegression

# In[ ]:


clf=LogisticRegression()
scoring='accuracy'
score_3=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_3=round(np.mean(score_3)*100,2)
score_3


# ## DecisionTreeClassifier

# In[ ]:


clf=DecisionTreeClassifier(max_depth=5)
scoring='accuracy'
score_4=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_4=round(np.mean(score_4)*100,2)
score_4


# ## RandomForestClassifier

# In[ ]:


clf=RandomForestClassifier(n_estimators=20)
scoring='accuracy'
score_5=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_5=round(np.mean(score_5)*100,2)
score_5


# ## SVC

# In[ ]:


clf=SVC(probability=True)
scoring='accuracy'
score_6=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=5,scoring=scoring)
score_6=round(np.mean(score_6)*100,2)
score_6


# # Comparing cross_val_score

# In[ ]:


plt.figure(figsize=(15,10))
y=[score_1,score_2,score_3,score_4,score_5,score_6]
x=['KNeighborsClassifier',
   'GaussianNB',
   'LogisticRegression',
   'DecisionTreeClassifier',
   'RandomForestClassifier',
   'SVC']
plt.title('Comparing accuracy of different models',fontsize=20)
splot=sns.barplot(x,y,edgecolor=(0,0,0),linewidth=2)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=15)
plt.xlabel('Models',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# # Creating Model

# In[ ]:


clf=RandomForestClassifier(n_estimators=20)
# training our model
clf.fit(X_train,y_train)
#predicting for test dataset
prediction=clf.predict(X_test)


# In[ ]:


# creating our submission dataframe
submission=pd.DataFrame({'PassengerId':sorted(test_data['PassengerId']),'Survived':prediction})
submission


# In[ ]:


# writing to csv file
submission.to_csv('submission.csv',index=False)

