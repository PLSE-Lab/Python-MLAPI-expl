#!/usr/bin/env python
# coding: utf-8

# # Data Understanding

# ## Librarires

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


# ## Loading Data

# In[ ]:


pwd


# In[ ]:


# Read train and test data with pd.read_csv():
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


# copy data in order to avoid any change in the original:
train=train_data.copy()
test=test_data.copy()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# # Analysis and Visualization of Numeric and Categorical Variables

# ## Basic summary statistics about the numerical data

# In[ ]:


train.describe().T


# ## Classes of some categorical variables

# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


train["Sex"].value_counts()


# In[ ]:


train["SibSp"].value_counts()


# In[ ]:


train["Parch"].value_counts()


# In[ ]:


train["Ticket"].value_counts()


# In[ ]:


train["Cabin"].value_counts()


# In[ ]:


train["Embarked"].value_counts()


# ## Visualization

# In general, barplot is used for categorical variables while histogram, density and boxplot are used for numerical data.

# In[ ]:


sns.barplot(x="Pclass",y="Survived", data=train) ;


# In[ ]:


sns.barplot(x="SibSp", y="Survived" , data= train);


# In[ ]:


sns.barplot(x= "Parch" , y="Survived", data=train);


# In[ ]:


sns.barplot(x="Sex",y="Survived" , data= train);


# In[ ]:


train.info()


# In[ ]:


Age_visualization=train["Age"].dropna()


# In[ ]:


sns.distplot(Age_visualization, kde = False);


# In[ ]:


sns.kdeplot(train["Fare"], shade = True);


# In[ ]:


(sns
 .FacetGrid(train,
              hue = "Survived",
              height = 7,
              xlim = (0, 500))
 .map(sns.kdeplot, "Fare", shade= True)
 .add_legend()
);


# In[ ]:


(sns
 .FacetGrid(train,
              hue = "Survived",
              height = 5,
              xlim = (0, 90))
 .map(sns.kdeplot, "Age", shade= True)
 .add_legend()
);


# # Data Preparations

# ## Deleting Unnecessary Variables

# In[ ]:


train.head()


# ## Ticket

# In[ ]:


train= train.drop("Ticket", axis=1)
test=test.drop("Ticket", axis=1)
train.head()


# ## Outlier Treatment

# In[ ]:


train.describe().T


# In[ ]:


sns.boxplot(x=train["Fare"]);


# In[ ]:


Q1= train["Fare"].quantile(0.25)
Q3= train["Fare"].quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
upper_limit


# In[ ]:


test.isnull().sum()


# In[ ]:


train.sort_values("Fare", ascending=False).head(20)


# In[ ]:


train_Fare=train["Fare"]


# In[ ]:


test_Fare=test["Fare"]


# In[ ]:


upper_fare=263


# In[ ]:


aykiri_train = (train_Fare>upper_fare)


# In[ ]:


aykiri_test = (test_Fare> upper_fare)


# In[ ]:


train_Fare[aykiri_train] = upper_fare


# In[ ]:


train["Fare"]=train_Fare


# In[ ]:


test_Fare[aykiri_test] = upper_fare


# In[ ]:


test[test["PassengerId"]==1044]


# In[ ]:


train.sort_values("Fare", ascending=False).head(20)


# In[ ]:


test.sort_values("Fare", ascending=False).head()


# # Missing Value Treatment

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## Age 

# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.head()


# In[ ]:


train['Title'] = train['Title'].replace([ 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[ ]:


test['Title'] = test['Title'].replace([ 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


# In[ ]:


train[["Title","Age"]].groupby("Title").mean()


# In[ ]:


for i in train["Title"]:
    if i=="Master":
        train["Age"]=train["Age"].fillna(4)
    elif i=="Miss":
        train["Age"]=train["Age"].fillna(22) 
    elif i=="Mr":
        train["Age"]=train["Age"].fillna(32)
    elif i=="Mrs":
        train["Age"]= train["Age"].fillna(36)
    elif i=="Rare":
        train["Age"]= train["Age"].fillna(46)
    else:
        train["Age"]=train["Age"].fillna(41)


# In[ ]:


train.isnull().sum()


# In[ ]:


test[["Title","Age"]].groupby("Title").mean()


# In[ ]:


for i in train["Title"]:
    if i=="Master":
        test["Age"]=test["Age"].fillna(7)
    elif i=="Miss":
        test["Age"]=test["Age"].fillna(22) 
    elif i=="Mr":
        test["Age"]=test["Age"].fillna(32)
    elif i=="Mrs":
        test["Age"]= test["Age"].fillna(38)
    elif i=="Rare":
        test["Age"]= test["Age"].fillna(44)
    else:
        test["Age"]=test["Age"].fillna(41)


# In[ ]:


test.isnull().sum()


# ## Fare 

# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"] = test["Fare"].fillna(12)


# In[ ]:


test.isnull().sum()


# In[ ]:


train.isnull().sum()


# ## Cabin

# In[ ]:



train["N_cabin"] = (train["Cabin"].notnull().astype('int'))
test["N_Cabin"] = (test["Cabin"].notnull().astype('int'))

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train.head()


# ## Embarked

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train["Embarked"]=train["Embarked"].fillna("S")


# # Variable Transformation

# ## Embarked

# In[ ]:



from sklearn import preprocessing

lbe=preprocessing.LabelEncoder()
train["Embarked"]=lbe.fit_transform(train["Embarked"])
test["Embarked"]=lbe.fit_transform(test["Embarked"])


# In[ ]:


train.head()


# ## Sex

# In[ ]:


Sex_mapping={"male":0,"female":1}
train["Sex"]=train["Sex"].map(Sex_mapping)
test["Sex"]=test["Sex"].map(Sex_mapping)


# In[ ]:


train.head()


# ## Name and Title

# In[ ]:


train[["Title","Survived"]].groupby(["Title"], as_index=False).mean().sort_values("Survived")


# In[ ]:





# In[ ]:


Title_mapping={"Mr":1,"Rare":2,"Master":3,"Miss":4,"Mrs":5,"Royal":2}
train["Title"]=train["Title"].map(Title_mapping)
test["Title"]=test["Title"].map(Title_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_name=train["Name"]
for i in train['Name']:
    train['Name']= train['Name'].replace(i,len(i))
    
      


# In[ ]:


train["Name"]


# In[ ]:


for i in test['Name']:
    test['Name']= test['Name'].replace(i,len(i))


# In[ ]:


test["Name"].describe()


# In[ ]:


bins = [0,25,40, np.inf]
mylabels = ['s_name', 'm_name', 'l_name',]
train["Name_len"] = pd.cut(train["Name"], bins, labels = mylabels)
test["Name_len"] = pd.cut(test["Name"], bins, labels = mylabels)


# In[ ]:


train["Name_len"].value_counts()


# In[ ]:


train[["Name_len","Survived"]].groupby("Name_len").mean()


# In[ ]:


Name_mapping = {'s_name': 1, 'm_name': 2 , 'l_name': 3}
train['Name_len'] = train['Name_len'].map(Name_mapping)
test['Name_len'] = test['Name_len'].map(Name_mapping)


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# ## AgeGroup

# In[ ]:


sns.distplot(train["Age"], kde = False);


# In[ ]:


sns.distplot(Age_visualization, kde = False);


# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[ ]:


train[["AgeGroup","Survived"]].groupby("AgeGroup").mean()


# In[ ]:


# Map each Age value to a numerical value:
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult':5 , 'Adult': 6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8])
test['FareBand'] = pd.qcut(test['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8])


# In[ ]:


train.head()


# # Feature Engineering

# ## Family Size

# In[ ]:


train.head()


# In[ ]:


train["FamilySize"] =train["SibSp"]+train["Parch"]+1
train["FamilySize"].mean()


# In[ ]:


test["FamilySize"] =test["SibSp"]+test["Parch"]+1
test["FamilySize"].mean()


# In[ ]:


sns.distplot(train["FamilySize"], kde = False);


# In[ ]:


train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# ## Embarked & Title &   Pclass

# In[ ]:


train.head()


# In[ ]:


train = pd.get_dummies(train, columns = ["Title"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


test = pd.get_dummies(test, columns = ["Title"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[ ]:


train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")


# In[ ]:


test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# In[ ]:


train.head()


# In[ ]:





# # Modeling, Evaluation and Model Tuning

# ## Spliting the train data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# ## Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


gbk


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


# ## Deployment

# In[ ]:


test


# In[ ]:


ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:


output


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




