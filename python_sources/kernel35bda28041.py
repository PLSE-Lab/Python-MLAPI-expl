#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:


gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train.head()


# In[ ]:


gender_submission.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cmap='magma')


# As there are too many null values in the cabin column we are going to drop this column

# In[ ]:


train.drop(['Cabin','Ticket'],axis=1,inplace=True)


# In[ ]:


train.info()


# **EDA**

# In[ ]:


# Lets check the correlation
train.corr()


# In[ ]:


# lets do some visualization
sns.countplot(train['Survived'])


# In[ ]:


sns.countplot(train['Sex'])


# In[ ]:


sns.countplot(train['Survived'],hue=train['Sex'])


# In[ ]:


sns.countplot(train['Pclass'])


# In[ ]:


sns.countplot(train['Survived'],hue=train['Pclass'])


# In[ ]:


sns.countplot(train['Parch'])


# In[ ]:


sns.countplot(train['Survived'],hue=train['Parch'])


# In[ ]:


sns.countplot(train['SibSp'])


# In[ ]:


sns.countplot(train['Survived'],hue=train['SibSp'])
plt.legend(loc='upper right')


# In[ ]:


sns.distplot(train['Fare'],bins=50)


# In[ ]:


sns.pairplot(train)


# # Lets fill those null values

# In[ ]:


def impute_age(cols):
    age=cols[0]
    pclass=cols[1]
    sibsp=cols[2]
    parch=cols[3]
    
    age_t=train[(train['Pclass']==pclass)&(train['SibSp']==sibsp)&(train['Parch']==parch)]['Age'].median()
    age_s=train[(train['Pclass']==pclass)]['Age'].median()
    if pd.isnull(age):
        if pd.isnull(age_t):
            age=age_s
        else:
            age=age_t
    else:
        age=age
    return age


# In[ ]:


train['Age']=train[['Age','Pclass','SibSp','Parch']].apply(impute_age,axis=1)


# In[ ]:


train.info()


# In[ ]:


train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)


# In[ ]:


train.info()


# # Lets check for the outliers

# In[ ]:


sns.boxplot(train['Age'])


# In[ ]:


sns.boxplot(train['Fare'])


# In[ ]:


from collections import Counter


# In[ ]:


Counter([1,1,1,3,3,2,3,2,5,2,3,2,2,3])


# In[ ]:


def detect_outliers(df,n,features):
    outlier_indices=[]
    
    for col in features:
        Q1=np.percentile(df[col],25)
        Q3=np.percentile(df[col],75)
        IQR=Q3-Q1
        outlier_step=IQR*1.5
        outlier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=[k for k,v in outlier_indices.items() if v>n]
    return multiple_outliers


# In[ ]:


outliers_to_drop=detect_outliers(train,1,['Age','SibSp','Parch','Fare'])


# In[ ]:


len(outliers_to_drop)


# In[ ]:


train.loc[outliers_to_drop]


# In[ ]:


train.drop(outliers_to_drop,inplace=True)
train.reset_index(drop=True,inplace=True)
train.info()


# In[ ]:


sns.distplot(train['Fare'],bins=40)


# In[ ]:


train['Fare']=train['Fare'].apply(lambda x:np.log(x) if x>0 else 0)


# In[ ]:


sns.distplot(train['Fare'],bins=40)


# # New Feature imputing

# In[ ]:


train['title']=train['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])


# In[ ]:


train['title'].unique()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot('title',hue='Survived',data=train)


# In[ ]:


train['title']=train['title'].replace(['Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','the Countess','Jonkheer'],'Rare')


# In[ ]:


train['title']=train['title'].map({'Mr':0,'Mrs':1,'Miss':1,'Master':2,'Rare':3})


# In[ ]:


train['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train['Sex']=pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


train.head()


# In[ ]:


train.drop('Name',axis=1,inplace=True)


# In[ ]:


train.head()


# # Assigning inputs and output

# In[ ]:


x=train.drop(['PassengerId','Survived'],axis=1)
y=train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lg=LogisticRegression()


# In[ ]:


lg.fit(x_train,y_train)


# In[ ]:


predictions=lg.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))


# # Cross Validation

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


# In[ ]:


mns=[]
stds=[]
clfs=[KNeighborsClassifier(),SVC(),DecisionTreeClassifier(),RandomForestClassifier(),ExtraTreesClassifier(),AdaBoostClassifier(),MultinomialNB()]
for i in clfs:
    cvs=cross_val_score(i,x,y,scoring='accuracy',cv=5,n_jobs=-1,verbose=1)
    mns.append(cvs.mean())
    stds.append(cvs.std())


# In[ ]:


for i in range(7):
    print(clfs[i].__class__.__name__,':',mns[i]*100)


# In[ ]:


rfc=RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(x_train,y_train) #training the model
features=x.columns
importances = rfc.feature_importances_ #taking the feature importance values into the a variable
indices = np.argsort(importances)
plt.figure(figsize=(10,10)) #plotting these in to a horizontal barplot
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# In[ ]:


predictions=rfc.predict(x_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


train['Survived'].value_counts()


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgbc=XGBClassifier()


# In[ ]:


xgbc.fit(x_train,y_train)


# In[ ]:


predictions=xgbc.predict(x_test)


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


test['title']=test['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])


# In[ ]:


test['title']=test['title'].replace(['Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','the Countess','Jonkheer'],'Rare')


# In[ ]:


test['title']=test['title'].map({'Mr':0,'Mrs':1,'Miss':1,'Master':2,'Rare':3})


# In[ ]:


test['Embarked']=pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


test['Sex']=pd.get_dummies(test['Sex'],drop_first=True)


# In[ ]:


test.drop(['Name','Cabin','PassengerId','Ticket'],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


test_predictions=xgbc.predict(test)


# In[ ]:


df= pd.read_csv("../input/titanic/test.csv")


# In[ ]:


Submission=df[['PassengerId','Fare']]


# In[ ]:


Submission['Survived']=test_predictions


# In[ ]:


Submission.drop('Fare',axis=1,inplace=True)
Submission.head()


# In[ ]:


Submission.to_csv('Submission_titanic.csv',index=False)


# In[ ]:




