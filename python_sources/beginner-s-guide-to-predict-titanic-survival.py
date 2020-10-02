#!/usr/bin/env python
# coding: utf-8

# ## About the problem

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# **Key Take Aways**
# 
# 1) Exploratory Data Analysis
# 
# 2) Feature Engineering
# 
# 3) Advanced Machine Learning Techinques
# 
# The important of all is that you will get familiar with how the data science competition works. I hope this kernal will help people to prepare themselves for the data science competitions and people can use this kernal to brush up their skills.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno  #visualize the missing values from the data
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loading the train and test datasets
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train_df.head()


# **Finding the missing nature of the data**

# In[ ]:


train_df.isnull().sum()


# In[ ]:


isna_train = train_df.isnull().sum().sort_values(ascending=False)
isna_test = test_df.isnull().sum().sort_values(ascending=False)
plt.subplot(2,1,1)
plt_1=isna_train.plot(kind='bar')
plt.ylabel('Train Data')
plt.subplot(2,1,2)
isna_test.plot(kind='bar')
plt.ylabel('Test Data')
plt.xlabel('Number of features which are NaNs')


# **Observation:**
# 
# 1)From this figure it is clear that missing nature of the data is same in both train and test data 
# 
# 2)We have to figure out a common way to fill the missing data in both train and test. 
# 
# 3)For that let's go for exploratory Data Analysis.

# ## Exploratory Data Analysis

# **Understanding the Survival Nature of Titanic**

# In[ ]:


plt.pie(train_df.Survived.groupby(train_df.Sex).sum(), explode=(0,0.1), labels=[0,1], colors=['green', 'red'],
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[ ]:


#Distribution of Survival Age-wise
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


#Survival of people on gender
grid = sns.FacetGrid(train_df, col='Survived', row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[ ]:


#Survival of people on gender
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# ## Feature Engineering

# **Imputing the missing values:**
# 
# As name is nominal column and it has title's for each name (eg.**Mr.** Donald) and it is one of the best key to impute age.

# In[ ]:


Title_train=[]
for name in train_df.Name:
    Title_train.append(name.split('.')[0].split(',')[1])
    
Title_test=[]
for name in test_df.Name:
    Title_test.append(name.split('.')[0].split(',')[1])


# **Using the titles obtained we'll fill the age**

# In[ ]:


train_df['Title']=Title_train
test_df['Title']=Title_test

Title=list(set(Title_train))
for title in Title:
    train_df.loc[train_df["Title"]==title,'Age']=train_df.loc[train_df["Title"]==title,'Age'].fillna(train_df.loc[train_df["Title"]==title,'Age'].median())
Title=list(set(Title_test))
for title in Title:
    test_df.loc[test_df["Title"]==title,'Age']=test_df.loc[test_df["Title"]==title,'Age'].fillna(test_df.loc[test_df["Title"]==title,'Age'].median())


# **Explanation:**
# 
# 1) Subgrouping age based on title.
# 
# 2) Filling the missing values based on median so that distribution of data remains the same after filling Na's

# **Imputing Fare**
# 
# There is a single value missing from fare in test data let's impute that value with mean

# In[ ]:


test_df['Fare']=test_df.groupby('Title')['Fare'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


train_df.head()


# **Imputing the Embarked**

# In[ ]:


pd.crosstab([train_df.Embarked,train_df.Pclass],[train_df.Sex,train_df.Survived],margins=True).style.background_gradient(cmap='winter_r')


# Since age and sex seem's to be a good parameter we'll try to localize the Embarked based on those values and impute them

# In[ ]:


#Embarked in trian has two missing values we'll drop those rows
train_df['Embarked']=train_df.groupby(['Age','Sex'])['Embarked'].transform(lambda x: x.fillna(x.mode()))
test_df['Embarked']=train_df.groupby(['Age','Sex'])['Embarked'].transform(lambda x: x.fillna(x.mode()))


# **Creating a new feature based on Age**
# 
# 1) Since age is from 0-100 we can form a new column based on age
# 
#         Age Group    Class
#         0-18      -  Minor 0
#         18-40     -  Adult 1
#         40-60     -  Middle age 2
#         60-100    -  Old 3
#         

# In[ ]:


def AgeGroup(Age):
    Age_group=[]
    for age in Age:
        if (age)  < 18:
            Age_group.append(0)
        elif (age) >= 18 and (age) < 40:
            Age_group.append(1)
        elif (age) >= 40 and (age) < 60:
            Age_group.append(2)
        elif (age) >= 60 and (age) < 100:
            Age_group.append(3)
        else:
            Age_group.append(2)
    return Age_group

train_df["AgeGroup"]=AgeGroup(train_df['Age'])
test_df["AgeGroup"]=AgeGroup(test_df['Age'])


# **Creating a feature based on Sibsp and parch**
# 
# sibsp	- of siblings / spouses aboard the Titanic	
# parch	- of parents / children aboard the Titanic	
# 
# Since these two features represent Family we create new Column Family

# In[ ]:


train_df['FamilySize']=train_df['SibSp']+train_df['Parch']+1
test_df['FamilySize']=test_df['SibSp']+train_df['Parch']+1


# In[ ]:


import pylab 
import scipy.stats as stats

stats.probplot(train_df.Fare, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


train_df['Fare'].describe()


# **Fare Category**
# 
# Since Fare is From 0 - 512 we'll form a new variable Fare Category based on the quartile's it is distributed
#  
#        FareCategory  - Quartile
#             0        -  0-25%
#             1        -  25-50%
#             2        -  50-75%
#             3        -  75-100%

# In[ ]:


quartile_1=np.percentile(train_df.Fare, 25)
quartile_2=np.percentile(train_df.Fare, 50)
quartile_3=np.percentile(train_df.Fare, 75)


# In[ ]:


def Far_Cat(Fare):
    FarCat=[]
    for i in Fare:
        if i >=0 and i< quartile_1:
            FarCat.append(0)
        if i >=quartile_1 and i< quartile_2:
            FarCat.append(1)
        if i >=quartile_2 and i< quartile_3:
            FarCat.append(2)
        if i >=quartile_3:
            FarCat.append(3)
    return FarCat
train_df['FarCat']=Far_Cat(train_df.Fare)
test_df['FarCat']=Far_Cat(test_df.Fare)


# In[ ]:


train_df.head()


# In[ ]:


test_df['FamilySize']=test_df['FamilySize'].fillna(0)
test_df['Age']=test_df['Age'].fillna(0)


# **Dropping Columns**
# 
# **Name**        -     Unique so not needed
# 
# **Age**         -     Since we have AgeGroup,we'll delete this.
# 
# **Ticket**      -     Unique so not needed
# 
# **Fare**        -      Since we have FareCat,we'll delete this 
# 
# **Cabin**       -       Many Nan so imputing might lead to bais
# 
# **PassengerId** - Cannot be categorised

# In[ ]:


test_df.head()


# In[ ]:


pid=test_df.PassengerId
train_df=train_df.drop(['Ticket','Name','Age','Fare','Cabin','PassengerId'],axis=1)
test_df=test_df.drop(['Ticket','Name','Age','Fare','Cabin','PassengerId'],axis=1)


# **Label encoding categorical columns**

# In[ ]:


#Finding the columns whether they are categorical or numerical
cols = train_df.columns
num_cols = train_df._get_numeric_data().columns
print("Numerical Columns",num_cols)
cat_cols=list(set(cols) - set(num_cols))
print("Categorical Columns:",cat_cols)

from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    train_df[i]=LabelEncoder().fit_transform(train_df[i].astype(str)) 
    test_df[i]=LabelEncoder().fit_transform(test_df[i].astype(str)) 


# **Feature Selection**

# In[ ]:


fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_df.corr(),ax=ax,annot= False,linewidth= 0.02,linecolor='black',fmt='.2f',cmap = 'Blues_r')
plt.show()


# **Distribution with respect Survived**

# In[ ]:


for i in range(0, len(train_df.columns), 5):
    sns.pairplot(data=train_df,
                x_vars=train_df.columns[i:i+5],
                y_vars=['Survived'])


# ## Modelling

# In[ ]:


X=train_df.loc[:,train_df.columns!='Survived']
Y=train_df.Survived


# **Logistic regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty="l1", C=1).fit(X, Y)
prediction = lr.predict(test_df)
pred_lr = pd.DataFrame()
pred_lr['PassengerId']=pid
pred_lr['Survived'] = prediction
pred_lr.to_csv("../working/submission_lr.csv", index = False)


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission_lr.csv')


# **SVM Classifier**

# In[ ]:


from sklearn import svm
svc = svm.SVC(
    C=5,
    kernel="rbf",
    degree=3,
    gamma="auto",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    random_state=None,
)
model = svc.fit(X, Y)
prediction = model.predict(test_df)
pred_svc = pd.DataFrame()
pred_svc['PassengerId']=pid
pred_svc['Survived'] = prediction
pred_svc.to_csv("../working/submission_svc.csv", index = False)


# **Knn**
# 

# In[ ]:


from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model = model.fit(X,Y)
prediction = model.predict(test_df)
pred_knn = pd.DataFrame()
pred_knn['PassengerId']=pid
pred_knn['Survived'] = prediction
pred_knn.to_csv("../working/submission_svc.csv", index = False)


# **Decision Tree**

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight=None,
    presort=False,
)

model = clf.fit(X,Y)
prediction = model.predict(test_df)
pred_dt = pd.DataFrame()
pred_dt['PassengerId']=pid
pred_dt['Survived'] = prediction
pred_dt.to_csv("../working/submission_svc.csv", index = False)


# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(
    n_estimators=800,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="auto",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
)
rf.fit(X, Y)


# In[ ]:


prediction = rf.predict(test_df)
pred_rf = pd.DataFrame()
pred_rf['PassengerId']=pid
pred_rf['Survived'] = prediction
pred_rf.to_csv("../working/submission_rf.csv", index = False)


# **Gradient Boosting Classifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    loss="deviance",
    learning_rate=0.1,
    n_estimators=200,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    init=None,
    random_state=None,
    max_features=None,
)
model = clf.fit(X, Y)


# In[ ]:


prediction = rf.predict(test_df)
pred_GB = pd.DataFrame()
pred_GB['PassengerId']=pid
pred_GB['Survived'] = prediction
pred_GB.to_csv("../working/submission_gb.csv", index = False)


# **XGB Classifier**

# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X,Y)


# In[ ]:


prediction = model_xgb.predict(test_df)
pred_xgb = pd.DataFrame()
pred_xgb['PassengerId']=pid
pred_xgb['Survived'] = prediction
pred_xgb.to_csv("../working/submission_xgb.csv", index = False)


# ## Ensembling

# In[ ]:


from statistics import mode
final_pred = np.array([])
for i in range(0,len(test_df)):
    final_pred = np.append(final_pred, mode([pred_lr['Survived'][i],pred_dt['Survived'][i],pred_knn['Survived'][i],pred_svc['Survived'][i],pred_rf['Survived'][i], pred_GB['Survived'][i], pred_xgb['Survived'][i]]))


# In[ ]:


prediction = model_xgb.predict(test_df)
pred_ensemble = pd.DataFrame()
pred_ensemble['PassengerId']=pid
pred_ensemble['Survived'] = prediction
pred_ensemble.to_csv("../working/submission_ensemble.csv", index = False)


# Thanks **Manav Sehgal** for your kernel

# **Don't Forget to upvote :)**
