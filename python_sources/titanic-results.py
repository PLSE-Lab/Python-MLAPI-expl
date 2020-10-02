#!/usr/bin/env python
# coding: utf-8

# * #  TITANIC RESULTS ...

# # 1 Introduction
# ## 1.1 Load and check data
# # 2 Data Visualization
# ## 2.1 Numerical values
# ### 2.1.1 Age
# ### 2.1.2 SibSp
# ### 2.1.3 Parch
# ### 2.1.4 Fare
# ## 2.2 Categorical values
# ### 2.2.1 Survived
# ### 2.2.2 Sex
# ### 2.2.3 Pclass
# ### 2.2.4 Embarked
# ### 2.2.5 Cabin
# # 3. Filling Missing Values
# ## 3.1 Age
# ## 3.2 Embarked
# ## 3.3 Test "Fare"
# ## 3.4 Cabin
# ## 3.5 Ticket
# # 4.Variable Transformation
# ## 4.1 Sex
# ## 4.2 Name
# ## 4.3 Fare
# ## 4.4 AgeGroup
# # 5.Feature Engineering
# ## 5.1 Embarked & unvan
# ## 5.2 Pclass 
# #  6.Modeling, Evaluation and Model Tuning
# ## 6.1 Spliting the train data
# ## 6.2 Gradient Boosting Classifier
# #  7.Deployment
# 

# # 1 Introduction

# This is my first kernel at Kaggle. I choosed the Titanic competition which is a good way to introduce feature engineering and ensemble modeling. Firstly, I will display some feature analyses then will focus on the feature engineering. Last part concerns modeling and predicting the survival on the Titanic using an voting procedure.
# 
# This script follows three main parts:
# 
# 1.Feature analysis
# 2.Feature engineering
# 3.Modeling
# 

# ## 1.1 Importing Librarires 

# In[ ]:



import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')
# to display all columns:
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


# Read train and test data with pd.read_csv():
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# ## 1.2 Load and check data

# In[ ]:


# copy data in order to avoid any change in the original:
train = train_data.copy()
test = test_data.copy()


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.ndim


# In[ ]:


test.ndim


# In[ ]:


train.describe().T


# In[ ]:


test.describe().T


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


test.dtypes


# In[ ]:


train.dtypes


# In[ ]:


train.info()


# # 2. Feature analysis

# ## 2.1 Numerical values
# 

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


#  ### 2.1.1 Age

# In[ ]:


train["Age"].describe()


# In[ ]:


train["Age"].value_counts()


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# #### It seems that very young passengers have more chance to survive. 
# 
# 

# ### 2.1.2 SibSp

# In[ ]:


train["SibSp"].value_counts()


# In[ ]:


train["SibSp"].value_counts().plot.barh();


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train);


# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# #### It seems that passengers having a lot of siblings/spouses have less chance to survive
# 
# #### Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive

# ### 2.1.3 Parch

# In[ ]:


train["Parch"].value_counts()


# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train)
plt.show()


# #### Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).
# 
# #### Be carefull there is an important standard deviation in the survival of passengers with 3 parents/children
# 
# 

# ## 2.1.4 Fare

# In[ ]:


train["Fare"].value_counts()


# In[ ]:


sns.boxplot(x = train['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
print(lower_limit)

upper_limit = Q3 + 1.5*IQR
upper_limit


#  # 2.2 Categorical values

# ## 2.2.1 Survived

# In[ ]:


train["Survived"].value_counts()


# In[ ]:


train["Survived"].value_counts().plot.barh();


# ## 2.2.2 Sex

# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train["Sex"].value_counts()


# In[ ]:


train["Sex"].value_counts().plot.barh();


# In[ ]:


sns.catplot(x = "Sex", y = "Age", hue= "Survived",data = train);


# #### It is clearly obvious that Male have less chance to survive than Female.
# 
# #### So Sex, might play an important role in the prediction of the survival.
# 
# #### For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first".

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train)


# ## 2.2.3 Pclass

# In[ ]:


train["Pclass"].value_counts()


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train);


# In[ ]:


print("Pclass Percantage = 1  survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Pclass Percantage = 2  survived :", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Pclass Percantage = 3 survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# #### The passenger survival is not the same in the 3 classes. 
# #### First class passengers have more chance to survive than second class and third class passengers.

# ## 2.2.4 Embarked

# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# #### It seems that passenger coming from Cherbourg (C) have more chance to survive.
# 
# #### My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
# 
# #### Let's see the Pclass distribution vs Embarked

# In[ ]:


g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# #### Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.
# 
# #### At this point, i can't explain why first class has an higher survival rate. My hypothesis is that first class passengers were prioritised during the evacuation due to their influence.

# #### In boxplot, there are too many outlier data; we can not change all. Just repress the highest value -512

# ## 2.2.5 Cabin

# In[ ]:


train["Cabin"].value_counts()


# # 3. Filling Missing Values
# 

# ### 3.1 Age

# In[ ]:


train.isnull().sum()


# In[ ]:


g = sns.factorplot(y="Age",x="Sex",data=train,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=train,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=train,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=train,kind="box")


# In[ ]:


## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = train["Age"].median()
    age_pred = train["Age"][((train['SibSp'] == train.iloc[i]["SibSp"]) & (train['Parch'] == train.iloc[i]["Parch"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        train['Age'].iloc[i] = age_pred
    else :
        train['Age'].iloc[i] = age_med


# In[ ]:


index_NaN_age = list(test["Age"][test["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = test["Age"].median()
    age_pred = test["Age"][((test['SibSp'] == test.iloc[i]["SibSp"]) & (test['Parch'] == test.iloc[i]["Parch"]) & (test['Pclass'] == test.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        test['Age'].iloc[i] = age_pred
    else :
        test['Age'].iloc[i] = age_med


# In[ ]:


train.isnull().sum()


# ### 3.2 Embarked

# ##### Fill Embarked nan values of dataset set with 'S' most frequent value

# In[ ]:



train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### 3.3 Test "Fare"

# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"] = test["Fare"].fillna(12)


# In[ ]:


test["Fare"].isnull().sum()


#  ### 3.4 Cabin

# In[ ]:


train["Yeni_cabin"] = (train["Cabin"].notnull().astype('int'))
test["Yeni_Cabin"] = (test["Cabin"].notnull().astype('int'))
print("Percentage of Yeni_cabin = 1 who survived:", train["Survived"][train["Yeni_cabin"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Yeni_cabin = 0 who survived:", train["Survived"][train["Yeni_cabin"] == 0].value_counts(normalize = True)[1]*100)

sns.barplot(x="Yeni_cabin", y="Survived", data=train)

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# #### But we can see that passengers with a cabin have generally more chance to survive than passengers without cabin.
# 
# 

# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### 3.5 Ticket

# In[ ]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# In[ ]:


train.head()


# # 4.Variable Transformation

# ## 4.1 Sex

# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
train["Sex"] = train["Sex"].map({"male": 0, "female":1})
test["Sex"] = test["Sex"].map({"male": 0, "female":1})


# In[ ]:


train.head()


# ## 4.2 Name

# In[ ]:


train["unvan"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["unvan"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['unvan'] = train['unvan'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['unvan'] = train['unvan'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['unvan'] = train['unvan'].replace('Mlle', 'Miss')
train['unvan'] = train['unvan'].replace('Ms', 'Miss')
train['unvan'] = train['unvan'].replace('Mme', 'Mrs')


# In[ ]:


test['unvan'] = test['unvan'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['unvan'] = test['unvan'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['unvan'] = test['unvan'].replace('Mlle', 'Miss')
test['unvan'] = test['unvan'].replace('Ms', 'Miss')
test['unvan'] = test['unvan'].replace('Mme', 'Mrs')


# In[ ]:


train[['unvan', 'Survived']].groupby(['unvan'], as_index=False).mean()


# In[ ]:


# Map each of the unvan groups to a numerical value

unvan_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

train['unvan'] = train['unvan'].map(unvan_mapping)


# In[ ]:


test['unvan'] = test['unvan'].map(unvan_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


train.head()


# ## 4.3 Fare

# In[ ]:


# Map Fare values into groups of numerical values:
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[ ]:


# Drop Fare values:
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


train.head()


# ## 4.4 AgeGroup

# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[ ]:


# Map each Age value to a numerical value:
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[ ]:


#dropping the Age feature for now, might change:
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


# 5.Feature Engineering


# ## 5.1 Embarked & unvan
# 

# In[ ]:


### Embarked & Title
# Convert Title and Embarked into dummy variables:

train = pd.get_dummies(train, columns = ["unvan"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()


# In[ ]:


test = pd.get_dummies(test, columns = ["unvan"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()


# ## 5.2 Pclass 
# 

# In[ ]:



# Create categorical values for Pclass:
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
train.head()


# In[ ]:


test.head()


# # 6.Modeling, Evaluation and Model Tuning

# ## 6.1 Spliting the train data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
train


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# ## 6.2 Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


xgb_params = {
        'n_estimators': [200, 500],
        'subsample': [0.6, 1.0],
        'max_depth': [2,5,8],
        'learning_rate': [0.1,0.01,0.02],
        "min_samples_split": [2,5,10]}


# In[ ]:


xgb = GradientBoostingClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)


# In[ ]:


xgb_cv_model.fit(x_train, y_train)


# In[ ]:


xgb_cv_model.best_params_


# In[ ]:


xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 
                    max_depth = xgb_cv_model.best_params_["max_depth"],
                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],
                    n_estimators = xgb_cv_model.best_params_["n_estimators"],
                    subsample = xgb_cv_model.best_params_["subsample"])


# In[ ]:


xgb_tuned =  xgb.fit(x_train,y_train)


# In[ ]:


y_pred = xgb_tuned.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# ## 6.3 Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression() 
logreg.fit(x_train, y_train) 
y_pred = logreg.predict(x_test) 
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(acc_logreg)


# In[ ]:





# ## 6.4 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[ ]:


get_ipython().system('pip install lightgbm')


# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


lgb_model = LGBMRegressor().fit(x_train, y_train)


# In[ ]:


lgb_model


# In[ ]:


y_pred = lgb_model.predict(x_test)


# In[ ]:


lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}


# In[ ]:


lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(x_train, y_train)


# In[ ]:


lgbm_cv_model.best_params_


# In[ ]:


lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 6, 
                          n_estimators = 20).fit(x_train, y_train)


# In[ ]:


y_pred = lgbm_tuned.predict(x_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test, y_pred))


# # 7.Deployment

# In[ ]:


test


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()


# In[ ]:




