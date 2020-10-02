#!/usr/bin/env python
# coding: utf-8

# # Titanic Surival Prediction

# # Develop  Machine Learning Model to predict Titanic Survival

# ### About Dataset

# **Categorical:**
# ***
# 
# **Nominal**
# - **Cabin**
# - **Embarked** (Port of Embarkation)
#             -C(Cherbourg)
#             -Q(Queenstown) 
#             -S(Southampton)
#         
# -(Nominal variable with only two categories)
# - **Sex**
#     - Female
#     - Male
#     
# **Ordinal**
# - **Pclass** (socio-economic status of passenger) 
#             - 1(Upper)
#             - 2(Middle) 
#             - 3(Lower)
# 
# **Numeric:**
# ***
# **Discrete**
# - **Passenger ID**(Unique identifing # for each passenger)
# - **SibSp**
# - **Parch**
# - **Survived** (Outcome)
#     - 0 : Not survived
#     - 1 : survived
#     
# **Continous**
# - **Age**
# - **Fare**
# 
# **String**
# ***
# - **Ticket** (Ticket number for passenger.)
# - **Name**(  Name of the passenger.) 

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# ### Read csv Files

# In[ ]:


pwd


# In[ ]:


train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_df=train.copy()
test_df=test.copy()


# # Explore Dataset

# In[ ]:


train_df.describe().T


# In[ ]:


test_df.describe().T


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# ## Categorical Variable and Visualisaton

# In[ ]:


train_df['Sex'].value_counts()


# In[ ]:


train_df.groupby('Sex')['Survived'].mean()


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train_df)


# In[ ]:


train_df['Pclass'].value_counts()


# In[ ]:


train_df.groupby('Pclass')['Survived'].mean()


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train_df)


# In[ ]:


fig = plt.figure(figsize=(12,6),)

ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 
               color='red',
               shade=True,
               label='not survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived', 
              )
plt.title('Survived vs Non-Survived')
plt.ylabel("Frequency of Survived Passenger")
plt.xlabel("Class of Passenger")
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);


# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


train_df.groupby('Embarked')['Survived'].mean()


# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=train_df);


# In[ ]:


train_df['Survived'].value_counts().plot.barh().set_title('Frequency of Survived');


# In[ ]:


train_df['Ticket'].value_counts()


# In[ ]:


train_df['SibSp'].value_counts()


# In[ ]:


train_df.groupby('SibSp')['Survived'].mean()


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train_df);


# In[ ]:


train_df['Parch'].value_counts()


# In[ ]:


train_df.groupby('Parch')['Survived'].mean()


# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train_df);


# # Treatment of Missing Values

# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


100*train_df.isnull().sum()/len(train_df)


# Cabin is dropped because it has 78.23 missing value 

# In[ ]:


100*test_df.isnull().sum()/len(test_df)


# In[ ]:


train_df = train_df.drop(columns="Cabin")


# In[ ]:


test_df = test_df.drop(columns="Cabin")


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df['Title'] = train_df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


test_df['Title'] = test_df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


test_df['Title'].value_counts()


# In[ ]:


[train_df.groupby('Title')['Age'].median(),train_df.groupby('Title')['Age'].std(),train_df.groupby('Title')['Age'].mean()]


# Fill Missing Value will be filled median of each Title Category

# In[ ]:


train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'),inplace=True)


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'),inplace=True)


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


train_df[train_df.Embarked.isnull()]


# In[ ]:


train_df.groupby(['Embarked','Title'])['Title'].count()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna('S')


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df[test.Fare.isnull()]


# In[ ]:


test_df[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test_df["Fare"] = test_df["Fare"].fillna(12.46)


# In[ ]:


test_df.isnull().sum()


# ### Outliers

# In[ ]:


train_df.describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T


# fare seems to have outlier. max value is 512.3292 and 99 percentile is 249.00622

# In[ ]:


sns.boxplot(x = train_df['Fare']);


# In[ ]:


Q1=train_df['Fare'].quantile(0.25)
Q3=train_df['Fare'].quantile(0.75)
IQR=Q3-Q1
print(IQR)


# In[ ]:


lower_limit=Q1 - 1.5 * IQR
upper_limit=Q3 + 1.5 * IQR
print('lower limit: '+str(lower_limit))
print('upper limit: '+str(upper_limit))


# In[ ]:


train_df['Fare'][train_df['Fare']>upper_limit].count()


# Number of frequncy is too much. this upper limit cannot be accepted  

# In[ ]:


train_df['Fare'][train_df['Fare']>upper_limit].sort_values(ascending=False).head()


# 275 is acceptable for upper limit

# In[ ]:


train_df['Fare']=train_df['Fare'].replace(512.3292,275)
test_df['Fare']=test_df['Fare'].replace(512.3292,275)


# ### **Transformation of Variables**

# In[ ]:


embarked_dict={'S':1,'C':2,'Q':3}


# In[ ]:


train_df['Embarked']=train_df['Embarked'].map(embarked_dict)
test_df['Embarked']=test_df['Embarked'].map(embarked_dict)


# In[ ]:


train_df.head()


# In[ ]:


train_df['Sex']=train_df['Sex'].map(lambda x:0 if x=='female' else 1).astype(int)
test_df['Sex']=test_df['Sex'].map(lambda x:0 if x=='female' else 1).astype(int)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df['Title'].unique()


# In[ ]:


test_df['Title'].unique()


# In[ ]:


# There is no Royal Category at test_df,
title_dict={'Mr':1,'Mrs':2,'Miss':3,'Master':4,'Rare':5,'Royal':5}


# In[ ]:


train_df['Title']=train_df['Title'].map(title_dict)
test_df['Title']=test_df['Title'].map(title_dict)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ### **Drop Fields**

# In[ ]:


train_df=train_df.drop(['Name','Ticket','Fare'],axis=1)


# In[ ]:


train_df.head()


# In[ ]:


test_df=test_df.drop(['Name','Ticket','Fare'],axis=1)


# In[ ]:


test_df.head()


# ### **Converting to Dummy Variables**
# 

# In[ ]:


# new field family size
train_df['FamilySize']=train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize']=test_df['SibSp'] + test_df['Parch'] + 1


# In[ ]:


train_df.head()


# In[ ]:


train_df['is_Single']=train_df['FamilySize'].map(lambda x: 1 if x < 2 else 0)
test_df['is_Single']=test_df['FamilySize'].map(lambda x: 1 if x < 2 else 0)


# Dummy Fields - Emabarked Title Pclass

# In[ ]:


train_df=pd.get_dummies(train_df,columns=['Title'],drop_first=False)
test_df=pd.get_dummies(test_df,columns=['Title'],drop_first=False)


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Embarked"], prefix="Em")
test_df = pd.get_dummies(test_df, columns = ["Embarked"], prefix="Em")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df['Pclass'] = train_df['Pclass'].astype('category',copy=False)
train_df=pd.get_dummies(train_df,columns=['Pclass'],drop_first=False)
train_df.head()


# In[ ]:


test_df['Pclass'] = test_df['Pclass'].astype('category',copy=False)
test_df=pd.get_dummies(test_df,columns=['Pclass'],drop_first=False)
test_df.head()


# In[ ]:


train_df.head()


# ## **Modelling**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
Y = train_df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 13)


# ### **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log_model = round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_log_model)+str('%'))
print(confusion_matrix(y_pred, y_test))


# ### **Support Vector Machines**

# In[ ]:


from sklearn.svm import SVC
svm_model = SVC(kernel = "linear").fit(x_train, y_train)
y_pred = svm_model.predict(x_test)


# In[ ]:


acc_svm_model = round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_svm_model)+str('%'))
print(confusion_matrix(y_pred, y_test))


# #### Model Tunning

# In[ ]:


svc_params = {"C": np.arange(1,5)}

svc = SVC(kernel = "linear")

svc_cv = GridSearchCV(svc,svc_params, 
                            cv = 10, 
                            n_jobs = -1, 
                            verbose = 2 )

svc_cv.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(svc_cv.best_params_))


# In[ ]:


svc_tuned = SVC(kernel = "linear", C = 1).fit(x_train, y_train)
y_pred = svc_tuned.predict(x_test)


# In[ ]:


acc_svc_tuned = round(accuracy_score(y_pred, y_test), 4)*100
print(str(acc_svc_tuned)+str('%'))
print(confusion_matrix(y_pred, y_test))


# ### **CART**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
cart = DecisionTreeClassifier()
cart_model = cart.fit(x_train, y_train)
y_pred = cart_model.predict(x_test)


# In[ ]:


cart_model = round(accuracy_score(y_test, y_pred),4)*100
print(str(cart_model)+str('%'))
print(confusion_matrix(y_pred, y_test))


# #### Model Tunning

# In[ ]:


cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }


# In[ ]:


cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(cart_cv_model.best_params_))


# In[ ]:


cart = tree.DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)
cart_tuned = cart.fit(x_train, y_train)
y_pred = cart_tuned.predict(x_test)


# In[ ]:


cart_acc = round(accuracy_score(y_pred, y_test), 4)*100
print(str(cart_acc)+str('%'))
print(confusion_matrix(y_pred, y_test))


# ### **Gradient Boost Classifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X, Y)
y_pred = gb.predict(x_test)
acc_gradient = round(accuracy_score(y_pred, y_test), 4)*100
print(str(acc_gradient)+str('%'))
print(confusion_matrix(y_pred, y_test))


# #### Model Tunning

# In[ ]:


gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [100,500,1000],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}


# In[ ]:


gbm = GradientBoostingClassifier()

gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)


# In[ ]:


gbm_cv.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(gbm_cv.best_params_))


# In[ ]:


gbm = GradientBoostingClassifier(learning_rate = 0.001, 
                                 max_depth = 3,
                                min_samples_split = 2,
                                n_estimators = 1000)


# In[ ]:


gbm_tuned =  gbm.fit(x_train,y_train)


# In[ ]:


y_pred = gbm_tuned.predict(x_test)
gbm_acc = round(accuracy_score(y_pred, y_test), 4)*100
print(str(gbm_acc)+str('%'))
print(confusion_matrix(y_pred, y_test))


# ### **Random Forest Classifier**

# In[ ]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
ypred=rfc.predict(x_test)
acc_rfc = round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_rfc)+str('%'))
print(confusion_matrix(y_test,y_pred))


# #### Model Tunning

# In[ ]:


rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}


# In[ ]:


rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 


# In[ ]:


rf_cv_model.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(rf_cv_model.best_params_))


# In[ ]:


rf_tuned = RandomForestClassifier(max_depth = 5, 
                                  max_features = 2, 
                                  min_samples_split = 10,
                                  n_estimators = 1000)

rf_tuned.fit(x_train, y_train)


# In[ ]:


y_pred = rf_tuned.predict(x_test)
acc_rfc_tuned = round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_rfc_tuned)+str('%'))
print(confusion_matrix(y_test,y_pred))


# ### **LightGBM Classifier**

# In[ ]:


from lightgbm import LGBMClassifier
lgbm = LGBMClassifier().fit(x_train, y_train)
y_pred = lgbm.predict(x_test)
acc_lgbm=round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_lgbm)+str('%'))
print(confusion_matrix(y_test,y_pred))


# #### Model Tunning

# In[ ]:


lgbm = LGBMClassifier()
lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}


# In[ ]:


lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(lgbm_cv_model.best_params_))


# In[ ]:


lgbm = LGBMClassifier(learning_rate = 0.02, 
                       max_depth = 3,
                       subsample = 0.6,
                       n_estimators = 100,
                       min_child_samples = 10)


# In[ ]:


lgbm_tuned = lgbm.fit(x_train,y_train)


# In[ ]:


y_pred = lgbm_tuned.predict(x_test)
acc_lgbm_tuned=round(accuracy_score(y_pred, y_test) , 4)*100
print(str(acc_lgbm_tuned)+str('%'))
print(confusion_matrix(y_test,y_pred))


# ### **XGBOOST**

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb_model = XGBClassifier().fit(x_train, y_train)


# In[ ]:


y_pred = xgb_model.predict(x_test)
acc_xgb=round(accuracy_score(y_test, y_pred),4)*100
print(str(acc_xgb)+str('%'))
print(confusion_matrix(y_test,y_pred))


# #### Model Tunning

# In[ ]:


xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_samples_split": [2,5,10]}


# In[ ]:


xgb = XGBClassifier()
xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)


# In[ ]:


xgb_cv_model.fit(x_train, y_train)


# In[ ]:


print("Best Parameters: " + str(xgb_cv_model.best_params_))


# In[ ]:


xgb = XGBClassifier(learning_rate = 0.02, 
                    max_depth = 3,
                    min_samples_split = 2,
                    n_estimators = 100,
                    subsample = 0.6)


# In[ ]:


xgb_tuned =  xgb.fit(x_train,y_train)


# In[ ]:


y_pred = xgb_tuned.predict(x_test)
acc_xgb=round(accuracy_score(y_test, y_pred),4)*100
print(str(acc_xgb)+str('%'))
print(confusion_matrix(y_test,y_pred))


# ### Submission

# In[ ]:


#set ids as PassengerId and predict survival 
ids = test_df['PassengerId']
ypred = lgbm_tuned.predict(test_df.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': ypred })
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head(7)

