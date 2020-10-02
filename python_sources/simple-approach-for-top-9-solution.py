#!/usr/bin/env python
# coding: utf-8

# **Hello! Kagglers welcome to your first machine learning competition on kaggle. In this notebook I have explained how one should approach any Machine Learning problem step by step,I too follow these steps while solving any machine learning problem.**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np 
import pandas as pd 
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline(False)
cf.set_config_file(sharing = 'public',theme = 'space')
from cufflinks import iplot


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test =  pd.read_csv('../input/titanic/test.csv')


# # Workflow
# 
# 1. Exploratory Data Anlysis
# 
#    1.Missing Values and ralation with target
#    
#    2.Numerical Variables(both continuous and discrete) distribution and their realtionship with target
#    
#    3.Outliers in Numerical Variables
#    
#    4.Categorical Variables distribution and their relationship with target.
#    
#    5.Cardinality of categorical variables.
# 2. Feature Engineering
# 3. Model Building
# 4. Hyperparameter Tuning

# ## Exploratory Data Analysis

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape,test.shape


# ## Missing values

# In[ ]:


#Columns with missing values and percetage of missing values in them
null_columns = [col for col in train.columns if train[col].isnull().sum()>1]
for col in null_columns:
    print(col,': {},count {}'.format(train[col].isnull().mean(),train[col].isnull().sum()))


# 1. Cabin column has most missing values(77%) - I might drop this column in feature engineering section or create some new meaningful feature 
# 2. Age column has around 20% missing values.
# 3. Embarked column has only 0.22% missing value-It will be filled by most frequent category in feature engineering section

# #### Check If there is any relationship betwwen null values and target variable 
# 
# For this i will replace all missing values in those above missing value containing columns and replace them by 1 and others by 0 and then by ploting i will try to interpret the relation.

# In[ ]:


for col in null_columns:
    df = train.copy()
    df[col] = np.where(df[col].isna(),1,0)
    df.groupby(col)['Survived'].value_counts().iplot(kind = 'bar',xTitle = '(Missing,Survived)',yTitle = 'count')


# 1. For missing values in Age 52 ot of 177 of the missing values instances Survived whereas 290 out of 714 of the non missing values instances Survived.
# 2. For Cabin column 206 out 687 missing values instances are Survived so there must be some relation ship between cabin and data and survival chance
# 3. For Embarked there are only 2 missing values and both are survived.

# ## Numerical variables.
# 
# There are two types of numerical variables in this dataset:
# 1. Continuous numerical
# 2. Discrete numerical

# In[ ]:


num_cols = [col for col in train.columns if train[col].dtypes!='O']#Features with numerical values
conti_cols = ['Age','Fare']#Features with continuous numerical values


# In[ ]:


#Let's find out the disribution of continuous varibles by plotting histograms.
for col in conti_cols:
    df = train.copy()
    df[col].iplot(kind = 'hist',linecolor = 'white',xTitle = col,yTitle = 'count')


# 1. Age column is preety much normal distributed with some right side skewness.
# 2. Fare column is highly skewed because most passengers on the ship were of third and second class passengers hence mostly data is distributed in low fare area.

# **Let's See how target variable is dependent on continuous numerical feature**

# In[ ]:


for col in conti_cols:
    df = train.copy()
    df.groupby('Survived')[col].mean().iplot(kind = 'bar',xTitle = 'Survived',yTitle = 'mean'+col)


# In[ ]:


for col in conti_cols:
    df = train.copy()
    fig = px.histogram(data_frame=df,color='Survived',x = col,barmode='group',template='plotly_dark')
    fig.show()


# ## Outliers in continuous variables

# In[ ]:


#Let's find out the outliers by ploting box plots for continuous variables
for col in conti_cols:
    df = train.copy()
    fig = px.box(y= col,data_frame=df,width=600,height=400,template = 'plotly_dark')
    fig.show()


# 1. Age column has some ouliers because of elderly people and kids on the ship.
# 3. Fare column has many outliers it is because rich people have given more money for there covinience and comfort.

# In[ ]:


disc_cols = [col for col in num_cols if col not in conti_cols+['PassengerId']]#Features with discrete numerical values
disc_cols


# **Let's see the realtionship between the discrete numerical fetures and dependent variables**

# In[ ]:


for col in disc_cols:
    df = train.copy()
    df.groupby('Survived')[col].value_counts().iplot(kind = 'bar',xTitle = '(Survived,'+col+')',yTitle = 'count')


# 1. 342 out of 891 People survived, so our target variable is preety much balanced.
# 2. Very large no. of passengers(372) from class 3 are not survived, whereas more passangers are survived in class 2 and class 3 than died.

# ## Categorical Variables

# In[ ]:


cat_col = [col for col in train.columns if train[col].dtypes == 'O']
cat_col


# In[ ]:


train[cat_col].head()


# ## Cardinality of categorical variables

# In[ ]:


for col in cat_col:
    print(col,"cardinality is {}".format(train[col].nunique()))


# 1. Name,Cabin,and Ticket columns have very high cardinality i will find a some way to reduce their high cardinality for that more analysis is needed

# #### Relationship between Sex and Embarked categorical feature and target variable

# In[ ]:


for col in ['Sex','Embarked']:
    df = train.copy()
    fig = px.histogram(x= col,data_frame=df,color = 'Survived',height = 400,width = 600,barmode='group',template = 'plotly_dark')
    fig.show()


# In[ ]:


fig = px.bar(x ='Cabin',data_frame = train,barmode='group',template='plotly_dark')
fig.show()


# In[ ]:


df = train.copy()
df['deck'] = train.Cabin.str[0]


# In[ ]:


df.deck.unique()


# In[ ]:


train['missingcabin'] = np.where(train.Cabin.isna(),1,0)


# In[ ]:


train.missingcabin.value_counts()


# In[ ]:


train['deck'] = df.deck.fillna('Z')


# In[ ]:


train.deck.value_counts()


# In[ ]:


df = test.copy()
df['deck'] = df.Cabin.str[0]


# In[ ]:


df.deck.unique()


# In[ ]:


test['missingcabin'] = np.where(test.Cabin.isna(),1,0)
test['deck'] = df.deck.fillna('Z')
test.deck.value_counts()


# In[ ]:


train['title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


test['title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


test['title'].value_counts()


# In[ ]:


train['title'].value_counts()


# In[ ]:


rep = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master','Dr':'Miss','Rev':'Mr','Col':'Mr','Mile':'Miss','Major':'Mr','Ms':'Rare','Countess':'Mrs','Lady':'Mrs','Jonkheer':'Mrs','Mme':'Rare','Don':'Rare','Capt':'Rare','Sir':'Rare'}

train['title'] = train.title.map(rep)


# In[ ]:


train.title.value_counts()


# In[ ]:


rep = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master','Col':'Mr','Rev':'Mr','Dr':'Miss','Dona':'Rare','Ms':'Rare'}
test['title'] = test.title.map(rep)
test.title.value_counts()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.drop(['PassengerId','Name'],axis = 1,inplace = True)
test.drop(['PassengerId','Name'],axis = 1,inplace = True)


# In[ ]:


train.shape,test.shape


# In[ ]:


train['famsize'] = train['SibSp']+train['Parch']+1
test['famsize'] = test['SibSp']+test['Parch']+1


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.drop(['Cabin','Ticket'],axis = 1,inplace = True)
test.drop(['Cabin','Ticket'],axis = 1,inplace = True)


# In[ ]:


x = train.groupby(['Pclass','title']).mean()['Age']# class and title wise age for impotation of missing values
print(x)


# In[ ]:


x.iplot(kind = 'bar',yTitle='Mean-Age',xTitle='(class,Title)',linecolor = 'white')


# In[ ]:


def apply_age(title,Pclass):
    if(title=='Master' and Pclass==1):
        age=5
    elif (title=='Miss' and Pclass==1):
        age=31
    elif (title=='Mr' and Pclass==1):
        age=42
    elif (title=='Mrs' and Pclass==1):
        age=40
    elif (title=='Rare' and Pclass==1):
        age=46
    elif (title=='Master' and Pclass==2):
        age=2
    elif (title=='Mr' and Pclass==2):
        age=33
    elif (title=='Mrs' and Pclass==2):
        age=33
    elif (title=='Miss' and Pclass==2):
        age=23
    elif (title=='Rare' and Pclass==2):
        age=28
    elif (title=='Master' and Pclass==3):
        age=5
    elif (title=='Mr' and Pclass==3):
        age=28
    elif (title=='Miss' and Pclass==3):
        age=16
    elif (title=='Mrs' and Pclass==3):
        age=33
    else:
        age=30 # mean age considered from describe()
    return age


# In[ ]:


y = test.groupby(['Pclass','title']).mean()['Age']
print(y)


# In[ ]:


y.iplot(kind = 'bar',xTitle = '(Class,Title)',yTitle = "mean-age")


# In[ ]:


train['Agemissing'] = np.where(train.Age.isna(),1,0)


# In[ ]:


age_null = train[train.Age.isna()]
age_null['Age'] = age_null.apply(lambda row : apply_age(row['title'],row['Pclass']), axis = 1) 
train['Age'].fillna(value=age_null['Age'],inplace=True)


# In[ ]:


def apply_age_test(title,Pclass):
    if(title=='Master' and Pclass==1):
        age=8
    elif (title=='Miss' and Pclass==1):
        age=31
    elif (title=='Mr' and Pclass==1):
        age=42
    elif (title=='Mrs' and Pclass==1):
        age=43
    elif (title=='Rare' and Pclass==1):
        age=40
    elif (title=='Master' and Pclass==2):
        age=4
    elif (title=='Mr' and Pclass==2):
        age=33
    elif (title=='Mrs' and Pclass==2):
        age=33
    elif (title=='Miss' and Pclass==2):
        age=17
    elif (title=='Rare' and Pclass==2):
        age=28
    elif (title=='Master' and Pclass==3):
        age=5
    elif (title=='Mr' and Pclass==3):
        age=28
    elif (title=='Miss' and Pclass==3):
        age=16
    elif (title=='Mrs' and Pclass==3):
        age=33
    else:
        age=30 # mean age considered from describe()
    return age


# In[ ]:


test['Agemissing'] = np.where(test.Age.isna(),1,0)


# In[ ]:


age_null_test = test[test.Age.isna()]
age_null_test['Age'] = age_null_test.apply(lambda row : apply_age(row['title'],row['Pclass']), axis = 1) 
test['Age'].fillna(value=age_null_test['Age'],inplace=True)


# In[ ]:


test.Age.isna().sum()


# In[ ]:


cat_train = train.copy()
cat_test = test.copy()


# In[ ]:


train = pd.get_dummies(train,drop_first=True)
test = pd.get_dummies(test,drop_first=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


col = [col for col in train.columns if col not in test.columns]
print(col)


# In[ ]:


train.drop('deck_T',axis = 1,inplace = True)


# In[ ]:


train.shape


# In[ ]:


y = train['Survived']
train.drop('Survived',axis = 1,inplace = True)


# In[ ]:


pclass = {1:3,2:2,3:1}
train['Pclass'] = train.Pclass.map(pclass)


# In[ ]:


train.Pclass.value_counts()


# In[ ]:


test['Pclass'] = test.Pclass.map(pclass)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


test.Fare.fillna(test.Fare.median(),inplace=True)


# In[ ]:


test.info()


# In[ ]:


test.Agemissing.value_counts()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier,XGBRFClassifier
from catboost import CatBoostClassifier


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score,f1_score,roc_auc_score


# In[ ]:


rf = RandomForestClassifier(n_estimators=150,max_depth=4,random_state=42)


# In[ ]:


rf.fit(train,y)


# In[ ]:


print(classification_report(y,rf.predict(train)))


# In[ ]:


roc_auc_score(y,rf.predict(train))


# In[ ]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


submission['Survived'] = rf.predict(test)


# In[ ]:


submission.to_csv('Submission1',index=False)


# In[ ]:


lgb = LGBMClassifier(n_estimators=200,max_depth=5,learning_rate=0.01,random_state=42)
lgb.fit(train,y)
print(classification_report(y,lgb.predict(train)),roc_auc_score(y,lgb.predict(train)))


# In[ ]:


submission['Survived'] = lgb.predict(test)
submission.to_csv('submission2',index = False)


# #### Till this version of the notebook of I have done Exploratory Data Analysis,Feature Engineeringand created a baseline model, maybe in the next version i will do some hyperparameter tuning to improve score.
# 
# **If you liked this notebook please upvote**
