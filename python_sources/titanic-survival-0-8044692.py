#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Analysis

# ### Classification model to predict the survival of Passenger

# ## Importing Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing Data

# #### Train_data

# In[ ]:


train_df=pd.read_csv('../input/titanic/train.csv')
train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.shape


# #### Test_data

# In[ ]:


test_df=pd.read_csv('../input/titanic/test.csv')
test_df.head()


# In[ ]:


test_df.describe()


# In[ ]:


test_df.shape


# ## Data Visualization

# In[ ]:


train_df.columns


# ### Pclass

# In[ ]:


sns.countplot(train_df.Pclass).set_title("Count of Pclass")


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=train_df,ci=None).set_title('Pclass v/s Survived')


# The above plot clearly shows that number of passengers in 3 Pclass were maximum but most passengers survived are from 1 Pclass 

# ### Sex

# In[ ]:


sns.countplot(train_df.Sex).set_title('Count of Sex')


# In[ ]:


sns.barplot(x='Sex',y='Survived',data=train_df,ci=None).set_title("Sex v/s Survived")


# The above plot shows that number of male were more that number of femails but femail passenger survived more then male passenger

# ### Age

# In[ ]:


sns.distplot(train_df.Age)


# In[ ]:


ax=sns.distplot(train_df['Age'][train_df['Survived']==1],hist=False,label='Survived')
sns.distplot(train_df['Age'][train_df['Survived']==0],hist=False,label='Death')


# The above Distplot shows that there are more passenger in age group 20 to 40 but the passenger in age group 0 to 10 (Childrens) survived the most 

# ### SibSp

# In[ ]:


sns.countplot(train_df.SibSp)


# In[ ]:


sns.barplot(x='SibSp',y='Survived',data=train_df,ci=None)


# the above graph represents that single and two siblings survived the most 

# ### Parch

# In[ ]:


sns.countplot(train_df.Parch)


# In[ ]:


sns.barplot(x='Parch',y='Survived',data=train_df,ci=None)


# graph represents that parent/children aboard in total number 3 survived the most

# ### Fare

# In[ ]:


sns.distplot(train_df.Fare)


# In[ ]:


ax=sns.distplot(train_df['Fare'][train_df['Survived']==1],label='Survived',hist=False)
sns.distplot(train_df['Fare'][train_df['Survived']==0],label='Death',hist=False,ax=ax)


# passenger paid the low fare survived the least

# ### Embarked

# In[ ]:


sns.countplot(train_df.Embarked)


# In[ ]:


sns.barplot(x='Embarked',y='Survived',data=train_df,ci=None)


# majority of passengers boarded from S Embarked but passengers boarded from Embarked C survived the most

# ## Feature Engineering

# ### Creating Features

# ### Train_data

# #### Name

# In[ ]:


train_df['Title']=train_df['Name'].str.extract('([A-Za-z]+)\.')
train_df.head()


# In[ ]:


train_df.Title.value_counts()


# In[ ]:


title_dict={
    'Mr':'Mr',
    'Miss':'Miss',
    'Mrs':'Mrs',
    'Master':'Master',
    'Dr':'Other',
    'Rev':'Other',
    'Major':'Other',
    'Mlle':'Other',
    'Col':'Other',
    'Jonkheer':'Other',
    'Lady':'Other',
    'Sir':'Other',
    'Mme':'Other',
    'Don':'Other',
    'Capt':'Other',
    'Ms':'Other',
    'Countess':'Other',
}


# In[ ]:


train_df.Title=train_df.Title.map(title_dict)
train_df.Title.value_counts()


# In[ ]:


sns.countplot(train_df.Title)


# In[ ]:


sns.barplot(x='Title',y='Survived',data=train_df,ci=None)


# #### Age

# In[ ]:


train_df.groupby(['Title']).mean()


# In[ ]:


train_df.Age.fillna(train_df.groupby('Title')['Age'].transform('mean'),inplace=True)


# In[ ]:


bins=np.linspace(train_df.Age.min(),train_df.Age.max(),6)
group_name=['Children','Adult','Maturity','Aging','OldAge']
train_df['Age_binning']=pd.cut(train_df.Age,bins,labels=group_name,include_lowest=True)
train_df.head()


# In[ ]:


sns.countplot(train_df.Age_binning)


# In[ ]:


sns.barplot(x='Age_binning',y='Survived',data=train_df,ci=None)


# #### Fare

# In[ ]:


bin_fare=np.linspace(train_df.Fare.min(),train_df.Fare.max(),4)
group_fare=['Low','Medium','High']
train_df['Fare_binning']=pd.cut(train_df.Fare,bin_fare,labels=group_fare,include_lowest=True)
train_df.head()


# In[ ]:


sns.countplot(train_df.Fare_binning)


# In[ ]:


sns.barplot(x='Fare_binning',y='Survived',data=train_df,ci=None)


# #### SibSp and Parch

# In[ ]:


train_df['Alone']=np.where((train_df["SibSp"]+train_df["Parch"])>0, 0, 1)


# ### Test_data

# #### Name

# In[ ]:


test_df['Title']=test_df['Name'].str.extract('([A-Za-z]+)\.')
test_df.head()


# In[ ]:


test_df.Title.value_counts()


# In[ ]:


title_dict_2={
    'Mr':'Mr',
    'Miss':'Miss',
    'Mrs':'Mrs',
    'Master':'Master',
    'Dr':'Other',
    'Rev':'Other',
    'Col':'Other',
    'Ms':'Other',
    'Dona':'Other'
}


# In[ ]:


test_df.Title=test_df.Title.map(title_dict_2)
test_df.Title.value_counts()


# #### Age

# In[ ]:


test_df.groupby(['Title']).mean()


# In[ ]:


test_df.Age.fillna(test_df.groupby('Title')['Age'].transform('mean'),inplace=True)


# In[ ]:


bins_test=np.linspace(test_df.Age.min(),test_df.Age.max(),6)
group_name_test=['Children','Adult','Maturity','Aging','OldAge']
test_df['Age_binning']=pd.cut(test_df.Age,bins_test,labels=group_name_test,include_lowest=True)
test_df.head()


# #### Fare

# In[ ]:


test_df.Fare.fillna(train_df.groupby('Fare_binning')['Fare'].transform('mean'),inplace=True)


# In[ ]:


bin_fare_test=np.linspace(test_df.Fare.min(),test_df.Fare.max(),4)
group_fare_test=['Low','Medium','High']
test_df['Fare_binning']=pd.cut(test_df.Fare,bin_fare_test,labels=group_fare_test,include_lowest=True)
test_df.head()


# #### SibSp and Parch

# In[ ]:


test_df['Alone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)


# ### Removing Null Values

# ### train_data

# In[ ]:


train_df.isna().sum()


# In[ ]:


print('Cabin Null Percentage: ',train_df.Cabin.isna().sum()/train_df.shape[0]*100)


# In[ ]:


train_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_df.Embarked.value_counts()


# In[ ]:


train_df.Embarked.fillna('S',inplace=True)


# ### test_data

# In[ ]:


test_df.isna().sum()


# In[ ]:


test_df.drop('Cabin',axis=1,inplace=True)


# ## Removing Features

# In[ ]:


train_df.head()


# In[ ]:


train_df.drop(['Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)


# In[ ]:


test_df.drop(['Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# ### Dummies Variable

# In[ ]:


dummies_train=pd.get_dummies(train_df[['Sex','Embarked','Title','Age_binning','Fare_binning']],drop_first=True)


# In[ ]:


dummies_test=pd.get_dummies(test_df[['Sex','Embarked','Title','Age_binning','Fare_binning']],drop_first=True)


# In[ ]:


train_df=pd.concat([train_df,dummies_train],axis=1)


# In[ ]:


test_df=pd.concat([test_df,dummies_test],axis=1)


# In[ ]:


train_df.drop(['Sex','Embarked','Title','Age_binning','Fare_binning'],axis=1,inplace=True)


# In[ ]:


test_df.drop(['Sex','Embarked','Title','Age_binning','Fare_binning'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# ## Feature Selection

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2


# #### Analysing relationship between Categorical (Features) and Categorical (Output) variables

# We will use Chi-Squared

# In[ ]:


x_chi=train_df.drop(['PassengerId','Survived'],axis=1)
y_chi=train_df.Survived


# In[ ]:


features=[]
for col in x_chi.columns:
    res=chi2(train_df[[col]],y_chi)
    if res[1]<0.05:
        print(col,end=": ")
        print(res)
        features.append(col)


# In[ ]:


print(features)


# ## Model Development

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[ ]:


X=train_df[features]
y=train_df.Survived


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)


# ### LogisticRegression

# In[ ]:


lm=LogisticRegression()


# In[ ]:


lm.fit(x_train,y_train)


# In[ ]:


yhat_lm=lm.predict(x_test)


# In[ ]:


f1_score_lm=f1_score(y_test,yhat_lm)
f1_score_lm


# In[ ]:


accuracy_score_lm=accuracy_score(y_test,yhat_lm)
accuracy_score_lm


# ### DecisionTreeClassifier

# In[ ]:


tree=DecisionTreeClassifier()


# In[ ]:


tree.fit(x_train,y_train)


# In[ ]:


yhat_tree=tree.predict(x_test)


# In[ ]:


f1_score_tree=f1_score(y_test,yhat_tree)
f1_score_tree


# In[ ]:


accuracy_score_tree=accuracy_score(y_test,yhat_tree)
accuracy_score_tree


# ### GaussianNB

# In[ ]:


naive=GaussianNB()


# In[ ]:


naive.fit(x_train,y_train)


# In[ ]:


yhat_naive=naive.predict(x_test)


# In[ ]:


f1_score_naive=f1_score(y_test,yhat_naive)
f1_score_naive


# In[ ]:


accuracy_score_naive=accuracy_score(y_test,yhat_naive)
accuracy_score_naive


# ### SVC

# In[ ]:


svc=SVC()


# In[ ]:


svc.fit(x_train,y_train)


# In[ ]:


yhat_svc=svc.predict(x_test)


# In[ ]:


f1_score_svc=f1_score(y_test,yhat_svc)
f1_score_svc


# In[ ]:


accuracy_score_svc=accuracy_score(y_test,yhat_svc)
accuracy_score_svc


# ### KNeighborsClassifier

# In[ ]:


neighbour=KNeighborsClassifier()


# In[ ]:


neighbour.fit(x_train,y_train)


# In[ ]:


yhat_neighbour=neighbour.predict(x_test)


# In[ ]:


f1_score_neighbour=f1_score(y_test,yhat_neighbour)
f1_score_neighbour


# In[ ]:


accuracy_score_neighbour=accuracy_score(y_test,yhat_neighbour)
accuracy_score_neighbour


# ### RandomForestClassifier

# In[ ]:


forest=RandomForestClassifier()


# In[ ]:


forest.fit(x_train,y_train)


# In[ ]:


yhat_forest=forest.predict(x_test)


# In[ ]:


f1_score_forest=f1_score(y_test,yhat_forest)
f1_score_forest


# In[ ]:


accuracy_score_forest=accuracy_score(y_test,yhat_forest)
accuracy_score_forest


# ## Conclusion

# ### F1_Score

# In[ ]:


models_name=['LogisticRegression','DecisionTreeClassifier','GaussianNB','SVC','KNeighborsClassifier','RandomForestClassifier']


# In[ ]:


f1_score_models=[f1_score_lm,f1_score_tree,f1_score_naive,f1_score_svc,f1_score_neighbour,f1_score_forest]


# In[ ]:


fig,ax=plt.subplots(figsize=(10,6))
ax.bar(models_name,f1_score_models)
ax.set_title("F1 Score of  Test Data",pad=20)
ax.set_xlabel("Models",labelpad=20)
ax.set_ylabel("F1_Score",labelpad=20)
plt.xticks(rotation=90)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))


# ### Accuracy_Score

# In[ ]:


accuracy_score_models=[accuracy_score_lm,accuracy_score_tree,accuracy_score_naive,accuracy_score_svc,accuracy_score_neighbour,accuracy_score_forest]


# In[ ]:


fig,ax=plt.subplots(figsize=(10,6))
ax.bar(models_name,accuracy_score_models)
ax.set_title("Accuracy of Models on Test Data",pad=20)
ax.set_xlabel("Models",labelpad=20)
ax.set_ylabel("Accuracy",labelpad=20)
plt.xticks(rotation=90)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))


# In[ ]:


forest.fit(train_df[features],train_df.Survived)


# ### test_data

# In[ ]:


yhat_test_df=forest.predict(test_df[features])


# In[ ]:


test_df['Survived']=pd.Series(yhat_test_df)


# In[ ]:


submission_df=test_df[['PassengerId','Survived']]
submission_df.head()


# In[ ]:


submission_df.to_csv('answer_forest.csv',index=False,header=True)

