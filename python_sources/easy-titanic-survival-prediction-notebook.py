#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


# to import train and test of titanic dataset from kaggle
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


#  to copy titanic datasets ( as train and test) into another variable.
dtr = train_data.copy()
dts = test_data.copy()


# In[ ]:


dtr.info()


# In[ ]:


dts.info()


# test ve train datasetini islemleri kisaltmak adina birlestirlim. en sonunda tekrar ayiracagiz

# In[ ]:


# To combine the dtr and dts data into one variable. 
# (We do this so as not to repeat the same operations on both notebooks (dtr and dts) on both datasets.)
df = pd.concat([dtr,dts], ignore_index=True)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


# some of the operations on dataset that has numerical varibales. it is meaningless for categorical variables.
df.describe().T


# To find out how many different types of observation units are in categorical variables (**.value_counts()** function is for that.)

# In[ ]:


df["Pclass"].value_counts()


# In[ ]:


df["Sex"].value_counts()


# In[ ]:


df["Embarked"].value_counts()


# In[ ]:


df["SibSp"].value_counts()


# In[ ]:


df["Parch"].value_counts()


# In[ ]:


df["Ticket"].value_counts()


# Data Visulation

# In[ ]:


sns.barplot(x= "Pclass", y = "Survived", data = df);


# In[ ]:


sns.barplot(x= "Sex", y = "Survived", data = df);


# In[ ]:


sns.barplot(x= "Embarked", y= "Survived", data =df);


# In[ ]:


sns.barplot(x= "SibSp", y = "Survived", data= df);


# In[ ]:


sns.barplot(x= "Parch", y = "Survived", data= df);


# Data preparation

# In[ ]:


df.head()


# In[ ]:


# To assign categorically 1 to passengers with any cabin number and 0 to those without a cabin number.
df["Cabin"] = df["Cabin"].notnull().astype("int")


# In[ ]:


df["Cabin"].value_counts()


# In[ ]:


sns.barplot(x= "Cabin", y = "Survived", data= df);


# In[ ]:


df["Embarked"].value_counts()


# In[ ]:


df["Embarked"] = df["Embarked"].fillna("S")


# In[ ]:


df.info()


# In[ ]:


df["Age"] = df["Age"].fillna(df["Age"].mean())


# In[ ]:


df.isnull().sum()


# In[ ]:


df[df["Fare"].isnull()]


# In[ ]:



df[["Pclass", "Fare"]].groupby("Pclass").mean()


# In[ ]:


df["Fare"][1043] = 13


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.describe().T


# In[ ]:


sns.boxplot(x=df["Fare"])


# In[ ]:


Q1 = df["Fare"].quantile(0.25)
Q1


# In[ ]:


Q3 =  df["Fare"].quantile(0.75)
Q3


# In[ ]:


IQR = Q3-Q1


# In[ ]:


low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR
high_limit


# In[ ]:


df[df["Fare"] > high_limit]


# In[ ]:


df["Fare"].sort_values(ascending=False).head()


# In[ ]:


df["Fare"] = df["Fare"].replace(512.3292, 263)


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


embarked_mapping = {"S":1, "C":2, "Q": 3} 
df["Embarked"] = df["Embarked"].map(embarked_mapping)

#2. method

# for i in range(0, len(df["Embarked"])):
#     if df["Embarked"][i] == "S":
#         df["Embarked"][i] = 1
#     elif df["Embarked"][i] == "C":
#         df["Embarked"][i] = 2
#     elif df["Embarked"][i] == "Q":
#         df["Embarked"][i] = 3


# In[ ]:


df.head(20)


# In[ ]:


df.drop(["Ticket"],axis =1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


for i in range(0, len(df["Sex"])):
    if df["Sex"][i] == "male":
        df["Sex"][i] = 1
    elif df["Sex"][i] == "female":
        df["Sex"][i] = 0
# 2. method
# # from sklearn import preprocessing
# # lbe = preprocessing.LabelEncoder()
# # df.Sex = lbe.fit_transform(df.Sex)
   


# In[ ]:


df.head()


# In[ ]:


df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


df.head()


# In[ ]:


df["Title"].value_counts()


# In[ ]:


df.Title = df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
df.Title = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
df.Title = df['Title'].replace('Mlle', 'Miss')
df.Title = df['Title'].replace('Ms', 'Miss')
df.Title = df['Title'].replace('Mme', 'Mrs')


# In[ ]:


df.Title.value_counts()


# In[ ]:


df[["Title", "Survived"]].groupby(["Title"], as_index = False ).mean()


# In[ ]:


Title_mapping = {"Mr":1, "Miss":2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5} 
df["Title"] = df["Title"].map(Title_mapping)


# In[ ]:


df.head()


# In[ ]:


df.drop("Name", axis =1, inplace= True)


# In[ ]:


df.head()


# In[ ]:



#  to make an Agegroup
# df["AgeGroup"] = 0
# for i in range(0, len(df["Age"])):
#     if df["Age"][i] <= 5:
#         df["AgeGroup"][i] = 1
#     elif df["Age"][i] <= 12:
#         df["AgeGroup"][i] = 2
#     elif df["Age"][i] <= 18:
#         df["AgeGroup"][i] = 3
#     elif df["Age"][i] <= 24:
#         df["AgeGroup"][i] = 4
#     elif df["Age"][i] <= 35:
#         df["AgeGroup"][i] = 5
#     elif df["Age"][i] <= 60:
#         df["AgeGroup"][i] = 6
#     elif df["Age"][i] > 60:
#         df["AgeGroup"][i] = 7
# 2. Method
# bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
# mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
# df['AgeGroup'] = pd.cut(df["Age"], bins, labels = mylabels)

# age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
# df['AgeGroup'] = df['AgeGroup'].map(age_mapping)

# df.drop("Age", axis = 1, inplace = True)


# In[ ]:


#  to make Fares in Group

# df["FareBand"]= pd.qcut(df["Fare"], 5 , [1,2,3,4,5])
# df.FareBand.value_counts()
# df.drop("Fare", axis= 1, inplace = True)


# In[ ]:


df["FamilySize"] = df["SibSp"] + df["Parch"]+1


# In[ ]:


df.head()


# In[ ]:


df["Single"] = df["FamilySize"].map(lambda x: 1 if x ==   1 else 0)
df["SmaFam"] = df["FamilySize"].map(lambda x: 1 if x ==   2 else 0)
df["MedFam"] = df["FamilySize"].map(lambda x: 1 if 3<=x<= 4 else 0)
df["LarFam"] = df["FamilySize"].map(lambda x: 1 if x >    4 else 0)


# In[ ]:


df.head(5)


# In[ ]:


df = pd.get_dummies( df, columns = ["Title"], prefix = "Tit")
df = pd.get_dummies( df, columns = ["Embarked"], prefix = "Em")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# Applying "One Hot Encoding" method in "Pclass" Variable
# df["Pclass"] = df["Pclass"].astype("category")
# df = pd.get_dummies(df, columns = ["Pclass"], prefix = "Pc")
# df.head()


# In[ ]:


df.drop(["Parch","SibSp"], axis =1, inplace =True)


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


dtr = df[0:891]


# In[ ]:


dtr.info()


# In[ ]:


dtr.head()


# In[ ]:


dtr["Survived"] = dtr["Survived"].astype("int")


# In[ ]:


dtr.head()


# In[ ]:


dts = df[891:]
dts.index = dts.index -891


# In[ ]:


dts.head()


# In[ ]:


dts.info()


# In[ ]:


dts.drop("Survived", axis =1, inplace =True)
dts.head(5)


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = dtr.drop(["Survived","PassengerId"], axis = 1)
target = dtr["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 42)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[ ]:


dts.head(5)


# In[ ]:


# Random  Forest Classifier Maschine Learning Model

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[ ]:


#set ids as PassengerId and predict survival 
ids = dts['PassengerId']
predictions = logreg.predict(dts.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv',index=False)


# In[ ]:


output.head()


# In[ ]:


# GradientBoosting Classifier Maschine Learning Model

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
y_pred = xgb_tuned.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)

