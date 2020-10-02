#!/usr/bin/env python
# coding: utf-8

# WHO HAS SURVIVED IN TITANIC?
# 
# 1. Objective: 
# 
# The overall objective of this study is to understand better who has survived in Titanic accident. In order to do that a machine learning model will be developed by using the existing data covering several features of the people who were at Titanic during the accident. The model will predict which passengers survived during the accident.
# 
# 2. Variables and Their Types:
# 
# Survival: Survival -> 0 = No, 1 = Yes
# 
# Pclass: Ticket class -> 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# Sex: Sex
# 
# Age: Age in years
# 
# SibSp: # of siblings / spouses aboard the Titanic
# 
# Parch: # of parents / children aboard the Titanic
# 
# Ticket: Ticket number
# 
# Fare: Passenger fare
# 
# Cabin: Cabin number
# 
# Embarked: Port of Embarkation -> C = Cherbourg, Q = Queenstown, S = Southampton
# 
# 3. Variable Notes:
# 
# Pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower 
# 
# Age: Age is frictional if less than 1. If the age is estimated, it is in the form of xx.5
# 
# SibSp: The dataset defines family relations in this way.
# 
# Sibling = brother, sister, stepbrother, stepsister 
# 
# Spouse = husband, wife
# 
# Parch: The dataset defines family relations in this way.
# 
# Parent = mother, father 
# 
# Child = daughter, son, stepdaughter, stepson 
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 4. Importing Libraries:

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split, GridSearchCV


# 5. Loading and Making a Copy of the Data:

# In[ ]:


df_train = pd.read_csv("/kaggle/input/titanic/train.csv") 
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train = df_train.copy() 
test = df_test.copy()


# 6. Exploring Data Features:
# 
# What are the number of observations per variables?
# 
# What are the types of variables?

# In[ ]:


train.info()


# How does the initial observations look like?

# In[ ]:


train.head()


# What are the general characteristics of the numerical variables?

# In[ ]:


train.describe().T


# What is the survival rate in general?

# In[ ]:


train["Survived"].value_counts()


# In[ ]:


Survival_Rate = 342/891*100


# In[ ]:


print(str(Survival_Rate)+'%')


# How is the value breakdown of the categorical variables?

# In[ ]:


train["Sex"].value_counts()


# In[ ]:


train["Pclass"].value_counts()


# In[ ]:


train["SibSp"].value_counts()


# In[ ]:


train["Parch"].value_counts()


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train["Cabin"].value_counts()


# 7. Data Visualization:

# In[ ]:


sns.catplot(x="Survived", kind="count", data=train);


# In[ ]:


sns.catplot(x="Sex", hue="Survived", kind="count", data=train);


# In[ ]:


sns.barplot(x = "Sex", y = "Survived", data = train);


# In[ ]:


sns.catplot(x="Pclass", hue="Survived", kind="count", data=train);


# In[ ]:


sns.barplot(x = "Pclass", y = "Survived", data = train);


# In[ ]:


sns.catplot(x="SibSp", hue="Survived", kind="count", data=train);


# In[ ]:


sns.barplot(x = "SibSp", y = "Survived", data = train);


# In[ ]:


sns.catplot(x="Parch", hue="Survived", kind="count", data=train);


# In[ ]:


sns.barplot(x = "Parch", y = "Survived", data = train);


# In[ ]:


plt.hist(train["Age"])
plt.show()


# In[ ]:


sns.boxplot(x="Survived", y="Age", data=train);


# In[ ]:


sns.boxplot(x="Survived", y="Fare", data=train);


# 8. Data Preparation
# 
# 8.1. Deleting Unnecessary Variables
# 
# We are deleting the variable of Ticket and Name, since they are irrelavant for data analysis.
# We will also delete the variable of Cabin, since the total number of observations for this variable is 204, although total number of observations are 891. This is too low and it may not help us to understand who is survived in the Titanic.

# In[ ]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(['Cabin'], axis = 1) 
test = test.drop(['Cabin'], axis = 1)


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# 8.2. Outlier Treatment
# 
# It seems that the standard deviation for the variable Fare is too much. 
# 
# According to the boxplot it is clear that there is a number of outliers for this variable. 
# 
# In this regard, we will replace them with an acceptable maximum figure.

# In[ ]:


sns.boxplot(y="Fare", data=train);


# In[ ]:


Q1 = train['Fare'].quantile(0.25) 
Q3 = train['Fare'].quantile(0.75) 
IQR = Q3 - Q1


# In[ ]:


lower_limit = Q1 - 1.5*IQR 
lower_limit


# In[ ]:


upper_limit = Q3 + 1.5*IQR
upper_limit


# In[ ]:


train['Fare'] > (upper_limit)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


test.sort_values("Fare", ascending=False)


# In[ ]:


count = 0
for i in train["Fare"] : 
    if i > 65.6344 :
        count = count + 1


# In[ ]:


print("The numbers of observations greater than upper limit of 65.6344 : " + str(count))


# In[ ]:


count = 0
for i in train["Fare"] : 
    if i > 200 :
        count = count + 1


# In[ ]:


print("The numbers of observations greater than 200 : " + str(count))


# The number of observations higher than 75% of the median value for the variable of Fare is 116, 
# 
# It is too much for replacing with the upper limit of 65.6344.
# 
# However, the number of outliers higher than 200 (taking into account the boxplot) is just 20.
# 
# So, the Fare values of these observations will be replaced by 200 to make the dataframe less skewed.

# In[ ]:


for i in train["Fare"] : 
    if i > 200 :
        train["Fare"].replace(i, 200, inplace=True)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


for i in test["Fare"] : 
    if i > 200 :
        test["Fare"].replace(i, 200, inplace=True)


# In[ ]:


test.sort_values("Fare", ascending=False).head()


# In[ ]:


sns.boxplot(y="Fare", data=train);


# In[ ]:


train.describe().T


# 8.3. Missing Value Treatment

# When we look at the number of available observations per variables, we see that there are some missing values.
# 
# While the variable Age lacks 177 values, the variable Embarked has just 2 missing values. 
# 
# For the test data set, Age lacks 86 values and there is one missing value for Fare variable.
# 
# We will fill the missing values with relavant figures to have a complete data set. 

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# For the variable of Age, we will use the mean value for filling the missing values:

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train.isnull().sum()


# For the variable of Fare in the test set, we will use the mean value for filling the missing values:

# In[ ]:


test.isnull().sum()


# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].mean())


# In[ ]:


test.isnull().sum()


# For the variable of Embarked, we will use the most frequent value for filling the missing values:

# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


train["Embarked"].isnull().sum()


# 8.4. Variable Transformation

# In this section, we will transform categorical variables into numerical variables to make machine learning workable.

# The categorical variable of Sex is turned into 1 and 0 under a new variable of Gender. Sex is dropped.
# 
# The categorical variable of Embarked is turned into 0,1 and 2 under a new variable of Embarked_new. Embarked is dropped.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit_transform(train["Sex"])
train["Gender"] = lbe.fit_transform(train["Sex"])


# In[ ]:


train.drop(["Sex"], inplace = True, axis =1)


# In[ ]:


train.tail()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit_transform(train["Embarked"])
train["Embarked_new"] = lbe.fit_transform(train["Embarked"])


# In[ ]:


train.drop(["Embarked"], inplace = True, axis =1)


# In[ ]:


train.head()


# In[ ]:


lbe.fit_transform(test["Sex"])
test["Gender"] = lbe.fit_transform(test["Sex"])
test.drop(["Sex"], inplace = True, axis =1)
lbe.fit_transform(test["Embarked"])
test["Embarked_new"] = lbe.fit_transform(test["Embarked"])
test.drop(["Embarked"], inplace = True, axis =1)
test.head()


# 8.5. Feature Engineering
# 
# A new variable under the name of Family Size will be created by combining the variables of SibSp and Parch. SibSp and Parch will be deleted from the data set.

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


train.drop(["SibSp"], inplace = True, axis = 1)


# In[ ]:


test.drop(["SibSp"], inplace = True, axis = 1)


# In[ ]:


train.drop(["Parch"], inplace = True, axis = 1)


# In[ ]:


test.drop(["Parch"], inplace = True, axis = 1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# 8.6. Creating Dummy Variable
# 
# In this section, dummy variables will be created for Pclass, Gender, Embarked_new and Family Size.

# In[ ]:


train = pd.get_dummies(train, columns = ["Gender"], prefix ="Gen") 
train = pd.get_dummies(train, columns = ["Embarked_new"], prefix="Em")
train = pd.get_dummies(train, columns = ["Pclass"], prefix="Pclass")
train = pd.get_dummies(train, columns = ["FamilySize"], prefix="Famsize")

test = pd.get_dummies(test, columns = ["Gender"], prefix ="Gen") 
test = pd.get_dummies(test, columns = ["Embarked_new"], prefix="Em")
test = pd.get_dummies(test, columns = ["Pclass"], prefix="Pclass")
test = pd.get_dummies(test, columns = ["FamilySize"], prefix="Famsize")


# In[ ]:


train.head()


# In[ ]:


test.head()


# 9. Modeling, Evaluation and Model Tuning
# 
# 9.1. Splitting the Train Data

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
predictors = train.drop(['Survived', 'PassengerId'], axis=1) 
target = train["Survived"] 
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 42)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# 9.1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression() 
logreg.fit(x_train, y_train) 
y_pred = logreg.predict(x_test) 
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(acc_logreg)


# 9.2. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier() 
randomforest.fit(x_train, y_train) 
y_pred = randomforest.predict(x_test) 
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(acc_randomforest)


# In[ ]:





# 9.3. Gradient Boosting Classifier

# In[ ]:





# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier() 
gbk.fit(x_train, y_train) 
y_pred = gbk.predict(x_test) 
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(acc_gbk)


# In[ ]:


xgb_params = { 'n_estimators': [200, 500], 'subsample': [0.6, 1.0], 'max_depth': [2,5,8], 'learning_rate': [0.1,0.01,0.02], "min_samples_split": [2,5,10]}
xgb = GradientBoostingClassifier()
xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
xgb_cv_model.best_params_


# In[ ]:


xgb = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, min_samples_split = 10, n_estimators = 200, subsample = 0.6)
xgb_tuned = xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test) 
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(acc_gbk)


# 10. Deployment
# 
# Since the accuracy rate achieved by the Random Forest Classifier is the highest, we decided to deploy this model for predicting the Survival values. 
# 
# We will set ids as PassengerId and predict the survival
# We will set the output as a dataframe and convert to csv file named submission.csv 

# In[ ]:


ids = test['PassengerId'] 
predictions = randomforest.predict(test.drop('PassengerId', axis=1))


# In[ ]:


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions }) 
output.to_csv('submission.csv', index=False)
output.head()


# In[ ]:


output.describe().T


# In[ ]:


output["Survived"].value_counts()


# In[ ]:


Survival_Rate = 152/418*100


# In[ ]:


print(str(Survival_Rate)+'%')


# In[ ]:




