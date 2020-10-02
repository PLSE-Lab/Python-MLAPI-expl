#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# ## 1. Checking Data
# * 1.1 Categorical vs Numerical features
# 
# * 1.2 Data Description
# 
# * 1.3 About Missing Values
# 
# * 1.4 Checking Response(Target) Variable
# 
# ## 2. Exploratory Data Analysis
# * 2.1 Pclass
# 
# * 2.2 Sex
# 
# * 2.3 Embarked
# 
# * 2.4 Ticket
# 
# * 2.5 Cabin
# 
# * 2.6 Age
# 
# * 2.7 SibSp and Parch
# 
# * 2.8 Fare
# 
# * 2.9 Name
# 
# ## 3. Filling Missing Values
# * 3.1 Embarked
# 
# * 3.2 Age
# 
# ## 4. Feature Engineering
# * 4.1 Age feature to categorical
# 
# * 4.2 Fare feature to categorical
# 
# * 4.3 FamilySize feature to categorical
# 
# * 4.4 Sex feature
# 
# * 4.5 One - hot encoding for categorical features
# 
# * 4.6 Dropping unnecessary columns
# 
# ## 5. Model Selection

# In[ ]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


#loading data
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
data=[train, test]


# ## 1. Checking Data

# In[ ]:


train.head()


# ### 1.1  Categorical vs Numerical features
# 
# We can divide features into two groups; categorical and numerical features.
# 
# * Categorical: Pclass, Name, Sex, Ticket, Cabin, Embarked
#    (Pclass is Ordinal)
# * Numerical: Age, Fare, SibSp, Parch
# 
# ### 1.2  Data Description

# In[ ]:


train.info()


# As seen above, there are total 891 observations, and 12 columns. Some values are null values, so we need to deal with null values. 

# In[ ]:


train.describe(include='all')


# ### 1.3   About Missing Values

# In[ ]:


import missingno as msno

msno.bar(train)


# In[ ]:


msno.bar(test)


# #### Training Data
# * There are two features which have quite a lot of missing values; Age and Cabin columns.
# * Especially, it seems that it is hard to fill Cabin column's missing values, since around 80% of values are null. Thus, we will drop Cabin column later.
# * Furthermore, 2 values of Embarked column were missing. Except above columns, non had missing values.
# 
# #### Test Data
# * In test data, age and cabin columns had a lot of null values.
# * One missing value in Fare column was detected.

# In[ ]:


#Set sns style
plt.style.use('seaborn')
sns.set(font_scale=1.5)


# ### 1.4  Checking Response(Target) Variable
# 
# * Checking target variable is important. In this problem, we need to predict whether the passenger survived or not. 
# * Target Variable is 'Survived' in this problem.
# * If target variable has skewed distribution, it can cause class imbalance problem.

# In[ ]:


sns.countplot('Survived', data=train)


# In[ ]:


print(train['Survived'].value_counts(normalize=True))


# * Around 38% of passengers in the training dataset survived. It seems that there will be no big influence from the class imbalanced problem, since the distribution is quite balanced.

# ## 2. Exploratory Data Analysis

# ### 2.1  Pclass

# In[ ]:


fig, ax=plt.subplots(1,2,figsize=(20,5))
(train[['Survived', 'Pclass']].groupby(['Pclass']).mean()).plot.bar(ax=ax[0], color='orange')
ax[0].set_title('Mean Survival Rate of Passengers by Pclass')
ax[0].set_ylabel('Mean Survival Rate')
sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Dead/Survived count by Pclass')


# * Mean Survival Rate of Passengers by Pclass differed from class 1 to class 3
# * Passengers with higher class survived a lot, while passengers with lower class survived less.
# * Pclass variable plays an significant role on predicting the target variable.

# ### 2.2  Sex

# In[ ]:


train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color='orange')
plt.title('Female / Male Survival Rate')
plt.ylabel('Survival Rate')

print(train[['Sex','Survived']].groupby(['Sex']).mean())


# * There was a big difference between survival rate of female and male.
# * Female survival rate was a lot higher than male survival rate.
# * Sex is an important feature for the target variable.

# In[ ]:


g=sns.pointplot('Pclass','Survived',hue='Sex',data=train)
g.legend(bbox_to_anchor=(0.95, 1), ncol=1)


# * There was no exception. In all classes, female survival rate was much more higher than male survival rate.

# ### 2.3  Embarked

# In[ ]:


train[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(color='orange')


# * Passengers from Cherbourg(C) port had higher survival rate than passengers from other port.
# 
# Let's get deeper!

# In[ ]:


fig, ax=plt.subplots(1,2,figsize=(20,7))
sns.countplot('Embarked', hue='Sex',data=train, ax=ax[0])
ax[0].set_title('Sex, Embarked Together')
sns.countplot('Embarked',hue='Pclass',data=train, ax=ax[1])
ax[1].set_title('Pclass, Embarked Together')


# From above two plots, we can say
# * For C, Q, gender ratio was about 1, while S had more male passengers.
# * Low survival rate of S port may be related with ratio of male passengers.
# 
# 
# * 3rd class was the most prevalent for passengers from port S and Q.
# * Passengers from port C were mostly in class 1, 3.
# * Low survival rate of port S and high survival rate of port C may be related with class distribution in each port. 

# ### 2.4  Ticket

# In[ ]:


train['Ticket'].value_counts()


# It is hard to find specific patterns in ticket variable. Thus, I will drop this column later.

# ### 2.5  Cabin

# In[ ]:


train['Cabin'].isnull().sum()/len(train['Cabin'])


# * We already know that this variable has about 77% null values.
# * It is hard to derive useful information.
# * Thus, I will exclude this varialbe from my model.

# ### 2.6  Age

# In[ ]:


train.loc[train['Survived']==0,'Age'].plot.hist(bins=20, alpha=0.5)
train.loc[train['Survived']==1,'Age'].plot.hist(bins=20, alpha=0.5)
plt.legend(['Dead','Survived'])
plt.title('Distribution of Age by Survival')
plt.xlabel('Age')


# We can find some interesting facts related to age.
# * Infants, and children had high survival rate.
# * Most passnegers were 15~35 years old.
# * Large number of passengers whose age is over 20 did not survive.
# * It would be better to divide age values into several intervals.

# ### 2.7  SibSp and Parch
# 
# * For SibSp and Parch, both variables are related to the number of family members. It would be better to combine two columns into one column.
# * Our new column name is FamilySize, and it represents the number of family members.
# * It can be derived by SibSp + Parch + 1. The reason we add 1 is to include passenger themselves.

# In[ ]:


for dataset in data:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1


# In[ ]:


sns.barplot('FamilySize','Survived',data=train)


# * Survival rate differed a lot by FamilySize.
# * Single family and family with more than 5 members had low survival rate.
# * Family with 2~4 members had higher survival rate.
# * Family with 5~7 members had lower survival rate.

# ### 2.8  Fare

# In[ ]:


train['Fare'].plot.hist(bins=20)


# * Fare variable is right-skewed. Skewness can lead to overweight high valued ourliers, causing bad performance. To fix this skewness, I will transform this values with log function.
# 
# * Before transformation, there is one missing value in the test data. We will fill this with the median value of the test data.

# In[ ]:


test['Fare']=test['Fare'].fillna(test['Fare'].median())


# In[ ]:


for dataset in data:
    dataset['Fare']=dataset['Fare'].map(lambda x: np.log(x) if x > 0 else 0)


# In[ ]:


train['Fare'].plot.hist(bins=20)


# After transformation, Fare column became less skewed.

# In[ ]:


train.loc[train['Survived']==0,'Fare'].plot.hist(bins=20, alpha=0.5)
train.loc[train['Survived']==1,'Fare'].plot.hist(bins=20, alpha=0.5)
plt.legend(['Dead','Survived'])
plt.title('Distribution of Fare by Survival')
plt.xlabel('Fare')


# * Survival rate of passengers with cheap ticket was lower than passengers with expensive ticket.
# * Almost passengers with Fare smaller than 2 died. 
# * Meanwhile, most passengers with Fare bigger than 4 survived.
# * It looks like survival rates differ from the intervals. 

# In[ ]:


pd.concat([train['Survived'], pd.cut(train['Fare'], 4)], axis=1).groupby(['Fare']).mean().plot.bar(color='orange', rot=45)
plt.title('Survival rate by Fare intervals')


# * We verified that survival rate among the intervals differed a lot. 
# * It would be better to divide Fare values into several intervals. 

# In[ ]:


sns.barplot('Pclass', 'Fare', data=train)


# * We could also verify that higher class tends to have expensive fare. 

# ### 2.9 Name

# In[ ]:


train['Name'].head()


# * It is easy to catch that Name values include passengers' title.
# * For example, Mr., and Mrs. appeared above. 
# * Title is significant information and it is even related to passengers' age.
# * I will extract those titles from the original Name column. To extract title, we can utilize the fact that comma is followed by title.

# In[ ]:


train['Title']=[each[1].split('.')[0].strip() for each in train['Name'].str.split(',')]
test['Title']=[each[1].split('.')[0].strip() for each in test['Name'].str.split(',')]


train['Title'].value_counts()


# * Mlle is french word of Miss. Mme is french word of Mrs. 
# * Considering above facts, we will divide Title values into Mr, Miss, Mrs, Master, and Rare(which means etc value).

# In[ ]:


for dataset in data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train['Title'].value_counts()


# In[ ]:


sns.barplot('Title','Survived',data=train)


# * Above barplot corresponds with the analysis that female and children passengers were likely to survive more.
# * It is shown that Mr(represents male, adult) title passengers survived less than other titles.

# ## 3. Filling Missing Values

# ### 3.1 Embarked

# In[ ]:


print(train['Embarked'].isnull().sum())
train['Embarked'].value_counts()


# * There are only 2 missing values in Embarked feature(training data). 
# * We can simply replace missing values with the most frequent value of Embarked (S).

# In[ ]:


train['Embarked']=train['Embarked'].fillna('S')


# ### 3.2  Age

# In[ ]:


print(train['Age'].isnull().sum())


# There are 177 missing values in Age column. Since it is not a small number, we cannot fill them with just mean value, or median value. Here, I would like to replace missing age values using Title feature. Title feature is related to Age, definitely. Mrs usually implies older women, while Miss implies younger women. Also, Mr usually implies older men, while Master implies younger men. 
# 
# 
# We can fill missing age values with mean age of the corresponding Title value. 

# In[ ]:


Title_list=list(train['Title'].unique())


# In[ ]:


for each in Title_list:
    train.loc[(train['Age'].isnull())&(train['Title']==each), 'Age'] = round(train[['Age', 'Title']].groupby(['Title']).mean().loc[each,'Age'])
    test.loc[(test['Age'].isnull())&(test['Title']==each), 'Age'] = round(test[['Age', 'Title']].groupby(['Title']).mean().loc[each,'Age'])


# ## 4. Feature Engineering

# Until now, we did exploratory data analysis and found some significant correlation with features and response variable. In this feature engineering section, we modify, combine, drop feature variables to maximize our prediction model accuracy. 

# In[ ]:


train.describe(include = 'all')


# ### 4.1 Age feature to categorical

# During EDA section, we found some discrete patterns with age levels. For example, passengers younger than 16 years old survived a lot, while passengers older than 16 years old died a lot. Dividing continuous age feature into several discrete levels will be helpful for our model accuracy. I will divide age into 5 levels.
# 
# We can use pandas cut method to implement this transformation.

# In[ ]:


train['Age']=pd.cut(train['Age'],5, labels=[0,1,2,3,4])
test['Age']=pd.cut(test['Age'], 5, labels=[0,1,2,3,4])


# ### 4.2 Fare feature to categorical

# On EDA section, we divided fare values into 4 intervals and found survival rates differ from the intervals. Dividing into several intervals and making it categorical will help our model performance. 
# 
# We can use pandas cut method. 

# In[ ]:


train['Fare']=pd.cut(train['Fare'],4, labels=[0,1,2,3])
test['Fare']=pd.cut(test['Fare'], 4, labels=[0,1,2,3])


# ### 4.3 FamilySize feature to categorical

# Survival rates of 2~4 familysize passengers were similar. Survival rates of 5~7 familysize passengers were also similar. None of passengers with more than 8 family members survived. 
# 
# Thus, let's divide familysize into 4 categories; Alone, 2~4, 5~7, more than 8.

# In[ ]:


train['FamilySize']=train['FamilySize'].map(lambda x: 0 if x == 1 else (1 if x<=4 else (2 if x<=7 else 3)))
test['FamilySize']=test['FamilySize'].map(lambda x: 0 if x == 1 else (1 if x<=4 else (2 if x<=7 else 3)))


# ### 4.4 Sex feature

# Currently, type of sex feature is string. We need to convert these string values into numerical values so that we can use this feature in machine learning method. 
# 
# I will map (male,female) into numerical value (1,0).

# In[ ]:


train['Sex']=train['Sex'].map({'male': 1, 'female':0})
test['Sex']=test['Sex'].map({'male': 1, 'female':0})


# ### 4.5 One - hot encoding for categorical features

# In titanic dataset, there are two kinds of categorical variables. One is ordinal categorical variables, such as Age, Fare, Pclass, FamilySize (Age, Fare features were categorized above). 
# 
# These ordinal cateogorical features can be ordered with specific rules. We can handel those with 2 methods. 
# 
# One is label encoding, which transforms values into simple numerical values. Order of each level is preserved, but when similar levels have much different survival rates, this method will not help that much. 
# 
# The other is one - hot encoding, which creates dummy variables. When similar levels have much different survival rates, this method will help our model accuracy. However, order of each level will be no longer meaningful. 
# 
# We need to check every categorical features whether similar levels have similar survival rates or not. For Age, FamilySize, Fare, Pclass, they had quite different survival rate even though each levels are similar. 
# 
# Therefore, I will use one - hot encoding for those features. 
# 
# 
# The other type of categorical variable is non ordinal categorical variables. In this case, I will use one - hot encoding for these variables. Embarked, Title features are non ordinal categorical variables. 
# 
# We can make dummy variables using pandas get_dummies method.

# In[ ]:


train=pd.concat([train, pd.get_dummies(train['Age'], prefix='Age')], axis=1)
test=pd.concat([test, pd.get_dummies(test['Age'], prefix='Age')], axis=1)

train=pd.concat([train, pd.get_dummies(train['FamilySize'],prefix='FamilySIze')], axis=1)
test=pd.concat([test, pd.get_dummies(test['FamilySize'],prefix='FamilySIze')], axis=1)

train=pd.concat([train, pd.get_dummies(train[['Embarked']])], axis=1)
test=pd.concat([test, pd.get_dummies(test[['Embarked']])], axis=1)

train=pd.concat([train, pd.get_dummies(train[['Title']])], axis=1)
test=pd.concat([test, pd.get_dummies(test[['Title']])], axis=1)

train=pd.concat([train, pd.get_dummies(train['Pclass'], prefix='Pclass')], axis=1)
test=pd.concat([test, pd.get_dummies(test['Pclass'], prefix='Pclass')], axis=1)

train=pd.concat([train, pd.get_dummies(train['Fare'], prefix='Fare')], axis=1)
test=pd.concat([test, pd.get_dummies(test['Fare'], prefix='Fare')], axis=1)


# ### 4.6 Dropping unnecessary columns

# In[ ]:


train.columns


# In[ ]:


drop_columns=['PassengerId','Name','Age','SibSp','Parch','Ticket','Cabin', 'Embarked','FamilySize', 'Title','Fare','Pclass']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)


# ## 5. Model Selection
# 
# For checking model accuracy, we will use 5-fold cross-validation. We can estimate test accuracy using cross-validation checking. 
# 
# Here are the models we will use for this problem. For each model, hyperparameter tunning will be done to find the best model. 
# 
# * Logistic Regression
# * Support Vector Machine
# * SGD Classifier
# * Random Forest
# * Gradient Boosting
# * Adaboost
# * XGboost

# In[ ]:


train_X=train[list(train.columns.drop('Survived'))]
train_Y=train['Survived']


# In[ ]:


train_X.dtypes


# In[ ]:


# Logistic Regression
lr=LogisticRegression()
print(cross_val_score(lr, train_X, train_Y, cv=5).mean())


# In[ ]:


# Support Vector Machine
svc=SVC()
svc_param={'kernel': ['linear', 'poly','rbf'], 
           'C': [1,10,20,50,100,200,500,1000], 
          'class_weight':[None, 'balanced']}
svc_grid=GridSearchCV(svc, svc_param, n_jobs=4, cv=5)
svc_grid.fit(train_X, train_Y)
print(svc_grid.best_params_)
print(svc_grid.best_score_)


# In[ ]:


# SGD Classifier
sgd=SGDClassifier()
print(cross_val_score(sgd, train_X, train_Y, cv=5).mean())


# In[ ]:


# Random Forest
rf=RandomForestClassifier()
rf_param={'n_estimators':[10, 50, 100, 500, 1000], 
         'min_samples_split': [2,5,10],
          'class_weight':[None, 'balanced']
         }
rf_grid=GridSearchCV(rf, rf_param, n_jobs=4, cv=5)
rf_grid.fit(train_X, train_Y)
print(rf_grid.best_params_)
print(rf_grid.best_score_)


# In[ ]:


# Gradient Boosting
gb=GradientBoostingClassifier()
gb_param={'n_estimators':[100, 500, 1000],
          'learning_rate':[0.01, 0.1, 0.2],
          'max_depth':[3,6,9],
          'min_samples_split': [2,5],
          'max_leaf_nodes':[8,16,32]
         }
gb_grid=GridSearchCV(gb, gb_param, n_jobs=4, cv=5)
gb_grid.fit(train_X, train_Y)
print(gb_grid.best_params_)
print(gb_grid.best_score_)


# In[ ]:


# Xgboost classifier
xgbst=xgb.XGBClassifier()
xgbst_param={'n_estimators':[100, 500, 1000],
          'learning_rate':[0.01, 0.1, 0.2],
          'max_depth':[3,6,9]
         }
xgbst_grid=GridSearchCV(xgbst, xgbst_param, n_jobs=4, cv=5)
xgbst_grid.fit(train_X, train_Y)
print(xgbst_grid.best_params_)
print(xgbst_grid.best_score_)


# In[ ]:


# Adaboost classifier
ada=AdaBoostClassifier()
ada_param={'n_estimators':[50, 100, 500, 1000],
          'learning_rate':[0.1, 0.5, 1]
         }
ada_grid=GridSearchCV(ada, ada_param, n_jobs=4, cv=5)
ada_grid.fit(train_X, train_Y)
print(ada_grid.best_params_)
print(ada_grid.best_score_)


# Judging from the cross validation score, support vector machine classifier with parameter {'C': 100, 'class_weight': None, 'kernel': 'rbf'} was the best. I will choose this model to predict test dataset. 

# In[ ]:


submission=pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


predict=svc_grid.predict(test)
submission['Survived']=predict
submission.to_csv('final_submission.csv', index=False)


# In[ ]:




