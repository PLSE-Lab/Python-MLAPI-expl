#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn.svm import SVC

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


sub = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


train.isnull().sum()


# Let me first try to fix the null values.Age,Cabin and Embarked have null values 

# In[ ]:


train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)
plt.title('Survived')
plt.ylabel('')


# Only 38.4% people survived.Now let us se how each feature given in the dataset such as age,class,gender and such affect the survival rate

# In[ ]:


train.head()


# In our training dataset we have survived as the target class and various types of features.Let me explore them first.
# 
# Categorical Features:Sex,Embarked,Pclass(Can be grouped)
# 
# Categorical features also contain subset features called ordinal features where there can be hierarchial/relative ordering between group
# 
# Continuos Features:Age(Can vary between any two numbers and cannot be categorised)

# In[ ]:


train.columns


# In[ ]:


train.groupby(['Sex','Survived'])['Survived'].count()


# In[ ]:


sns.countplot('Sex',hue='Survived',data=train)
plt.title('Sex:Survived vs Dead')
plt.show()


# Even though male count wa shigh,Survival count of women s twice to that of survival count of men.So ,we can tell that gender plays some role in survival detection

# In[ ]:


train.groupby(['Pclass','Survived'])['Survived'].count()


# In[ ]:


sns.countplot('Pclass',hue='Survived',data=train)
plt.title('Sex:Survived vs Dead')
plt.show()


# From the above plot,we can tell that people of certain class had more survival rate and it can also be seen that people of higher class had more survival chances than that of lower class.It might be that passengers of higher class were given higher priority while rescue.
# Let us now check the survival rate of sex and Pclass together

# In[ ]:


sns.catplot(x="Sex", y="Survived", col="Pclass",data=train, saturation=.5,
            kind="bar", ci=None, aspect=.6)


# In[ ]:


pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# From the above cross tab,we can see that the survival rate of women in 3rd class is very high that out of 94 only 3 women could not survive and even the survival rate of women compared to men in 2 nd and 3rd classes is also high.Let me now work with age

# In[ ]:


train['Age'].max(),train['Age'].min(),train['Age'].mean()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# From the above plots ,we can tell that  the survival rate of children (below age 10 ) is high irrespective of the class.
# Survival rate of people aged 20-50 is high in pclass1 and that female sex aged 20-50 have high survival rate compared to men.Let me now explore remaining features.
# 

# In[ ]:


fig = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
fig.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
fig.add_legend()


# In[ ]:


sns.factorplot('Embarked','Survived',data=train)
plt.show()


# Observations from the above plots:
# 
# Female from class 1,2,3  who began their journey at Station C had less survival rate and male had very high survival rates.Whereas the trend is opposite when they embarked from stations Q and S.
# 
# The survival chances are almost 1 for women for Pclass1 and Pclass2 irrespective of the Pclass.
# Female  survival rate irrespective of station decreases when class increases
# 
# Station S looks to be very unlucky for Pclass3 Passenegers as the survival rate for both men and women is very low.
# 
# Station Q looks looks to be unlukiest for Men, and almost all were from Pclass 3.
# 
# Overall survival rate for Station C is highest around 0.55
# 
# 
# 

# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'],
        as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Parch", "Survived"]].groupby(['Parch'], 
            as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


pd.crosstab(train.SibSp,train.Pclass).style.background_gradient(cmap='summer_r')


# In[ ]:


pd.crosstab(train.Parch,train.Pclass).style.background_gradient(cmap='summer_r')


# The above tables shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. 
# As number of siblings,spouse ,parents and children  abroad is high,the survival rate decreases.This makes sense. That is, if I have a family on board, I will try to save them instead of saving myself first. Surprisingly the survival for families with 5-8 members is 0%.Let us analyse the reason for it.
# The reason is Pclass.As people with sibsp,parch greater than 3 are in class 3

# In[ ]:


train.isnull().sum()


# As  age  is an important feature ,we need to fill those null values.We need to replace those values with some value.But we cannot fill it with any random number as the age spread is more.so let us see family names and the number of siblings,parents abroad, we will try to fill those values.Let us first fill other missing values

# In[ ]:


sns.countplot('Embarked',data=train)
plt.title('Passengers embarked at each station')
plt.show()


# 
# As maximum people embarked at station S,Let us fill the missing Embarked values with S

# In[ ]:


train['Embarked'].fillna('S',inplace=True)


# In[ ]:


train['Embarked'].isnull().any()


# In[ ]:


train.head(5)


# In[ ]:


train['Salutation']=0
for i in train:
    train['Salutation']=train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[ ]:


train


# In[ ]:


train['Salutation'].unique()


# In[ ]:


pd.crosstab(train.Sex,train.Salutation)


# Mis spelled salutations such as mlle can be replaced by miss and mme can be replaced by mrs and the other ones accordingly and this will make it easy to categorize

# In[ ]:


train['Salutation'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


train.groupby('Salutation')['Age'].mean()


# Like we can fill missing values based on corresponding group avrage ages

# In[ ]:


train.loc[(train.Age.isnull())&(train.Salutation=='Mr'),'Age']=33
train.loc[(train.Age.isnull())&(train.Salutation=='Mrs'),'Age']=36
train.loc[(train.Age.isnull())&(train.Salutation=='Master'),'Age']=5
train.loc[(train.Age.isnull())&(train.Salutation=='Miss'),'Age']=22
train.loc[(train.Age.isnull())&(train.Salutation=='Other'),'Age']=46


# In[ ]:


train['Age'].isnull().sum()


# Passenger class is given,and that makes cabin not a useful feature.So we can safely eliminate this feature.PasengerId,Ticket number can also be eliminated as they are present to represent the uniqueness of an individual and we are targetting to 

# In[ ]:


sns.factorplot('Pclass','Survived',col='Salutation',data=train)
plt.show()


# From the above plots it is evident that  children had highest survival rate and then female and then male.So a feature based on age groups can be developed

# In[ ]:


plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


train['Age_Range']=0
train.loc[train['Age']<=10,'Age_Range']=0
train.loc[(train['Age']>10)&(train['Age']<=20),'Age_Range']=1
train.loc[(train['Age']>20)&(train['Age']<=40),'Age_Range']=2
train.loc[(train['Age']>40)&(train['Age']<=50),'Age_Range']=3
train.loc[(train['Age']>50)&(train['Age']<=65),'Age_Range']=4
train.loc[train['Age']>65,'Age_Range']=5


# In[ ]:


sns.factorplot('Age_Range','Survived',data=train,col='Pclass')
plt.show()


# From the above plots,It is evident that as age increases,survival rate decreases irrespective of the Pclass

# In[ ]:


train.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


train['Fare'].nunique()


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# In[ ]:


plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()


# People in same class also paid different fare,It might be because of preferred cabin and there is no sufficient cabin data,let us keep fare as it is and without any modifications.Distribution of fare  in class1 is large and goes on decreasing as the class increases.This continuous distribution can be converted in ranges and then they can be used as discrete values so that any algorithm can classify them easily

# In[ ]:


train['Fare_Range']=0
train.loc[train['Fare']<=7.91,'Fare_Range']=0
train.loc[(train['Fare']>7.91)&(train['Fare']<=14.454),'Fare_Range']=1
train.loc[(train['Fare']>14.454)&(train['Fare']<=31),'Fare_Range']=2
train.loc[(train['Fare']>31)&(train['Fare']<=513),'Fare_Range']=3


# In[ ]:


sns.heatmap(train.corr(),annot=True,cmap='summer_r',linewidths=0.2) 
plt.show()


# Positive Correlation means two features are directly affected and in Negative Correlation,Two features move in opposite ways.If two features are positively correlated,it means that they affect the target value in the same way .So keeping only one feature can also work and we can safely remove the other feature to reduce training time.

# In[ ]:


train['FamilySize']=train['SibSp']+train['Parch']+1
train['IsAlone'] = 1 
train['IsAlone'].loc[train['FamilySize'] > 1] = 0


# In[ ]:


train.columns


# In[ ]:


train.drop(['Name','Age','Fare'],axis=1,inplace=True)


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train['Salutation'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# In[ ]:


type(train)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# From the above correlation map we can see that there is high correlation between family size,sibsp,parch and isalone.Its because those features are derived from one another

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xg

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[ ]:


train_f,test_f=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])
train_X=train_f.drop(['Survived'],axis=1)
train_Y=train_f['Survived']
test_X=test_f.drop(['Survived'],axis=1)
test_Y=test_f['Survived']
X=train.drop(['Survived'],axis=1)
Y=train['Survived']


# In[ ]:


lrc = LogisticRegression()
lrc.fit(train_X,train_Y)
prediction_lrc=lrc.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction_lrc,test_Y))


# In[ ]:


svc=svm.SVC(kernel='rbf',C=1,gamma=0.1)
svc.fit(train_X,train_Y)
prediction_svc=svc.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction_svc,test_Y))


# In[ ]:


dtc=DecisionTreeClassifier()
dtc.fit(train_X,train_Y)
prediction_dtc=dtc.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction_dtc,test_Y))


# In[ ]:


knn=KNeighborsClassifier() 
knn.fit(train_X,train_Y)
prediction_knn=knn.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction_knn,test_Y))


# In[ ]:


rfc=RandomForestClassifier(n_estimators=80)
rfc.fit(train_X,train_Y)
prediction_rfc=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction_rfc,test_Y))


# In[ ]:


ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())


# In[ ]:


ada.fit(train_X,train_Y)


# In[ ]:


sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = ada.predict_proba(test)
sub.to_csv("sub.csv",index=False)


# In[ ]:


grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())


# In[ ]:


xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())


# In[ ]:


f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1])
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0])
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1])
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()


# In[ ]:




