#!/usr/bin/env python
# coding: utf-8

# # Titanic is Sinking, Let's see who will Survive
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## Let Do EDA (Exploratory Data Analysis)

# In[ ]:


# import all the needed package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)


# In[ ]:


# train dataframe head
train.head()


# In[ ]:


# test dataframe head
test.head()


# In[ ]:


# let check info for dataframe
def check_info(train, test):
  print(train.info())
  print('_'*40)
  print(test.info())
  print('_'*40)


# In[ ]:


check_info(train,test)


# In[ ]:


# Check Data Description to have idea of the data
def describe_info(train, test):
  print(train.describe())
  print('_'*40)
  print(test.describe())
  print('_'*40)


# In[ ]:


describe_info(train, test)


# ### What we have done till now
# 
# 
# 
# 1.   Imported the Train and Test file in Respectative DataFrame
# 2.   Make the data index by Passerenger ID in both train and test
# 3.   Check the Data Infomation using info tag which tell us the that there are float, integer and text data in the set.
# 4.  There are missing data in the column which we need to fix in **Missing Data ** Section down below.
# 5.  We will have to explore the data more basis on Survived tab so we do not have dataset imbalanced issue. 
# 

# ## Missing Data Opertation
# 
# 
# 
# 1.   As Age is missing we would use mean to fill the missing value in train and test dataset
# 2.   Cabin data missing darstically, it would be good that we drop the column in both train and test.
# 3.   One Record for fare is missing in test dataset so will fill the mode.
# 
# 

# In[ ]:


TrainAgeMean = round(train['Age'].mean())
TestAgeMean = round(test['Age'].mean())
print('Train Age Mean :-', TrainAgeMean)
print('Test Age Mean :-', TestAgeMean)


# In[ ]:


train['Age'] = train['Age'].fillna(value=TrainAgeMean)
test['Age'] = test['Age'].fillna(value=TestAgeMean)


# In[ ]:


TestFareMode = test['Fare'].mode()[0]
print(TestFareMode)


# In[ ]:


test['Fare'] = test['Fare'].fillna(value=TestFareMode)


# In[ ]:


train.drop(columns=['Cabin'], axis=1, inplace=True)
test.drop(columns=['Cabin'], axis=1, inplace=True)


# In[ ]:


check_info(train, test)


# #### What we have done so far?
# 
# 
# 1.   Filled the missing value for Age in Train and Test Dataset using Mean Vale
# 2.   Remove Cabin from both the dataset as most of the data is missing
# 3.   A single record for fare was missing which was filled using Mode Value
# 
# 

# ## Let's Create some Graph

# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")


# In[ ]:


corr = train.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)


# In[ ]:


train['Family'] = train['Parch'] + train['SibSp']
test['Family'] = test['Parch'] + test['SibSp']


# In[ ]:


corr = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)


# In[ ]:


# Value Count Chart
ax = train['Survived'].value_counts().plot.bar()
ax.set_xlabel("Not Survived or Survived")
ax.set_ylabel("Number of People Survied or Not Survived")


# In[ ]:


# Number of People who survived the even according to Sex
train[train['Survived'] == 1].Sex.value_counts().plot.bar()


# In[ ]:


# Number of People who not survived the even according to Sex
train[train['Survived'] == 0].Sex.value_counts().plot.bar()


# ### What can we conclude from the above data?
# 
# From the above data we can conclude that the most people who died on the ship were male. Most of Female were saved might be due to the fact shown in movie about saving women and children on boat first. To Confirm the story our next action would be to see how many children who are below the age of 18 are survived the sinking.

# In[ ]:


train[(train['Age'] <= 18)].Sex.value_counts()


# In[ ]:


train[(train['Age'] <= 18) & (train['Sex'] == 'male')].Survived.value_counts().plot.bar()


# In[ ]:


train[(train['Age'] <= 18) & (train['Sex'] == 'female')].Survived.value_counts().plot.bar()


# By looking at the data above there are 139 Children on Titanic, out of which 71 were male and 68 were female. Most of Femal Child survived the sinking of the ship where male Child did not.

# Now that we know that maximum Female were alived, let check from where did they embarked.

# In[ ]:


train[train['Survived'] == 1]['Embarked'].value_counts().plot.bar()


# In[ ]:


train[train['Survived'] == 0]['Embarked'].value_counts().plot.bar()


# Looking at the data above we can say that the S Class had maximum people died and and survied Followed by C and Q

# In[ ]:


# Lets check the distubution of Fare
data = train['Fare']
sns.distplot(data)


# In[ ]:


sns.boxplot(data)


# From the above Boxplot and Distplot we can see that the Disturbation of Fare is right Skwed, and we sme outier which where are are above USD 200 to USD 500

# In[ ]:


check_info(train,test)


# In[ ]:


train.drop(columns=['Name'], axis=1, inplace=True)
test.drop(columns=['Name'], axis=1, inplace=True)


# In[ ]:


train.drop(columns=['Ticket'], axis=1, inplace=True)
test.drop(columns=['Ticket'], axis=1, inplace=True)


# In[ ]:


categorical_feature_mask = train.dtypes==object
categorical_cols = train.columns[categorical_feature_mask].tolist()
categorical_cols


# In[ ]:


test_categorical_feature_mask = test.dtypes==object
test_categorical_cols = test.columns[test_categorical_feature_mask].tolist()
test_categorical_cols


# In[ ]:


# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()


# In[ ]:


train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))


# In[ ]:


test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))


# In[ ]:


corr = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)


# In[ ]:


# Import all libary for Sklearn Model Evalulation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# In[ ]:


# Let's Create our Machine Learning Model to check how it is working but before that we will create a function which will print all the ROC, AUC and Confusion Matix so we will not have to performance these step mannualy.
def run_model(model, name, Xtrain, Xtest, ytrain, ytest):
  print(name + 'Model Details')
  model.fit(Xtrain, ytrain)
  ypred = model.predict(Xtest)
  print("F1 score is: {}".format(f1_score(ytest, ypred)))
  print("AUC Score is: {}".format(roc_auc_score(ytest, ypred)))


# In[ ]:


from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression


# In[ ]:


gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
xgb = XGBClassifier()
ada = AdaBoostClassifier()
clf = tree.DecisionTreeClassifier()
log = LogisticRegression()


# In[ ]:


model_list = [gbc, rfc, xgb, ada, clf]


# In[ ]:


model_name = ['Gradient Boosting', 'Random Forest', 'XGBoost', 'Ada Boost', 'Decision Tree']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train.drop(columns=['Survived'], axis=1)
y = train.Survived


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2)


# In[ ]:


gbc.fit(Xtrain, ytrain)
gbcpred = gbc.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, gbcpred)))
print("F1 score is: {}".format(f1_score(ytest, gbcpred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, gbcpred)))


# In[ ]:


rfc.fit(Xtrain, ytrain)
rfcpred = rfc.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, rfcpred)))
print("F1 score is: {}".format(f1_score(ytest, rfcpred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, rfcpred)))


# In[ ]:


xgb.fit(Xtrain, ytrain)
xgbpred = xgb.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, xgbpred)))
print("F1 score is: {}".format(f1_score(ytest, xgbpred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, xgbpred)))


# In[ ]:


ada.fit(Xtrain, ytrain)
adapred = ada.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, adapred)))
print("F1 score is: {}".format(f1_score(ytest, adapred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, adapred)))


# In[ ]:


clf.fit(Xtrain, ytrain)
clfpred = clf.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, clfpred)))
print("F1 score is: {}".format(f1_score(ytest, clfpred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, clfpred)))


# In[ ]:


log.fit(Xtrain, ytrain)
logpred = log.predict(Xtest)
print("Accuracy score is: {}".format(accuracy_score(ytest, logpred)))
print("F1 score is: {}".format(f1_score(ytest, logpred)))
print("AUC Score is: {}".format(roc_auc_score(ytest, logpred)))


# In[ ]:


test['Survived'] = xgb.predict(test)


# In[ ]:


test.head()


# In[ ]:


test.to_csv('survived.csv')

