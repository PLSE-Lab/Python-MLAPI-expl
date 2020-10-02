#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read dataset 
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_train.head()


# In[ ]:


# import packages 
import warnings
import pandas as  pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np    # linear algebra
import matplotlib.pyplot as plt # visulization
import seaborn as sns # visulization
import missingno as msno # handling missing value
from sklearn.preprocessing import StandardScaler,MinMaxScaler # dataset range (0 to 1)
from sklearn.model_selection import train_test_split # splitting the test dataset and train dataset
from sklearn.metrics import confusion_matrix,accuracy_score # model perfomance 
from sklearn.linear_model import LogisticRegression  # clasiifier
from sklearn.neighbors import KNeighborsClassifier # # clasiifier
from sklearn.tree import DecisionTreeClassifier # # clasiifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier # clasiifier
from sklearn.svm import SVC # clasiifierpackages  
import xgboost as xgb # clasiifierpackages 
from sklearn import preprocessing


# In[ ]:


#  show  top 5 row of train data set
titanic_train.head(5)


# In[ ]:


#  show  top 5 row of test data set
titanic_test.head(5)


# In[ ]:


# numbers of columns in datasets
titanic_train.columns


# In[ ]:


# dataset shape (row ,columns)
print(titanic_train.shape)
print(titanic_test.shape)


# In[ ]:


# data types in datasets
titanic_train.dtypes


# 
# 
# # Which features are categorical?
# These values classify the samples into sets of similar samples.Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.

# # Which features are numerical?
# These values change from sample to sample.Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# Continous: Age, Fare. Discrete: SibSp, Parch

# In[ ]:


# Get some statistical information
titanic_train.hist(bins=20,figsize=(20,10),color="#F1948B")


# In[ ]:


# What is the distribution of numerical feature values across the samples?

titanic_train.describe()


# In[ ]:


titanic_train.describe().T


# In[ ]:


# What is the distribution of categorical features?
titanic_train.describe(include="object")


# In[ ]:


# Survived by pclass
titanic_train[["Pclass","Survived"]].groupby("Pclass",as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


titanic_train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Survived by Sex
titanic_train[["Survived","Sex"]].groupby("Sex").mean().sort_values(by="Survived",ascending=False)


# In[ ]:


titanic_train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# visualizing age by survived
age_survived=sns.FacetGrid(titanic_train,col="Survived").map(plt.hist,"Age",bins=20)


# In[ ]:


pclass_survived=sns.FacetGrid(titanic_train,col="Survived").map(plt.hist,"Age",bins=20)


# In[ ]:


titanic_train[titanic_train.Survived==1].Sex.value_counts()


# In[ ]:


#Deleting columns which are of no use

# Train Set

titanic_train=titanic_train.drop(["PassengerId","Ticket"],axis=1)

# Submission
submission=pd.DataFrame(columns=["PassengerId", "Survived"])
submission["PassengerId"]=titanic_test["PassengerId"]

# test dataset
titanic_test = titanic_test.drop(["PassengerId", "Ticket"], axis=1)


# In[ ]:


# Check for missing values
titanic_train.isnull().sum()


# In[ ]:


# bar plot missing value

fig, axis=plt.subplots(1,2 ,figsize=(20,5),sharey=True)

msno.bar(titanic_train,ax=axis[0],color='#2FA353')

msno.bar(titanic_test,ax=axis[1], color='#2F7DA3')


# In[ ]:


#Filling Age missing values of training & test data set with Median

titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)
titanic_test["Age"].fillna(titanic_test["Age"].median(), inplace=True)


# In[ ]:


# As we saw earlier also in the graph
titanic_train["Embarked"].value_counts()


# In[ ]:


# Fill NAN of Embarked in training set with 'S'
titanic_train["Embarked"].fillna("S", inplace = True)


# In[ ]:


# Fill Missing Values for Cabin in training set with 0
titanic_train["Cabin"] = titanic_train["Cabin"].apply(lambda x: str(x)[0])
titanic_train.groupby(["Cabin", "Pclass"])["Pclass"].count()


# In[ ]:


titanic_train["Cabin"] = titanic_train["Cabin"].replace("n", 0)
titanic_train["Cabin"] = titanic_train["Cabin"].replace(["A", "B", "C", "D", "E", "T"], 1)
titanic_train["Cabin"] = titanic_train["Cabin"].replace("F", 2)
titanic_train["Cabin"] = titanic_train["Cabin"].replace("G", 3)


# In[ ]:


titanic_test["Cabin"] = titanic_test["Cabin"].apply(lambda x: str(x)[0])
titanic_test.groupby(["Cabin", "Pclass"])["Pclass"].count()


# In[ ]:


titanic_test["Cabin"] = titanic_test["Cabin"].replace("n", 0)
titanic_test["Cabin"] = titanic_test["Cabin"].replace(["A", "B", "C", "D", "E"], 1)
titanic_test["Cabin"] = titanic_test["Cabin"].replace("F", 2)
titanic_test["Cabin"] = titanic_test["Cabin"].replace("G", 3)


# In[ ]:


#Creating new variable Family Size & Alone
# Train Set
titanic_train["Family"] = titanic_train["SibSp"]+titanic_train["Parch"]

#Test Set
titanic_test["Family"] = titanic_test["SibSp"]+titanic_test["Parch"]


# In[ ]:


# 1 If alone & 0 if it has family members
titanic_train["Alone"] = titanic_train["Family"].apply(lambda x:1 if x==0 else 0)
titanic_test["Alone"] = titanic_test["Family"].apply(lambda x:1 if x==0 else 0)


# In[ ]:


titanic_test.head(3)


# In[ ]:


# Considering the other features, filling the NAN value of Fare accordingly
m_fare = titanic_test[(titanic_test["Pclass"] == 3) & (titanic_test["Embarked"] == "S") & (titanic_test["Alone"] == 1)]["Fare"].mean()
m_fare


# In[ ]:


titanic_test["Fare"] = titanic_test["Fare"].fillna(m_fare)
titanic_train["Fare"] = titanic_train["Fare"].fillna(m_fare)


# In[ ]:


def title(name):
    for string in name.split():
        if "." in string:
            return string[:-1]

titanic_train["Title"] = titanic_train["Name"].apply(lambda x: title(x))
titanic_test["Title"] = titanic_test["Name"].apply(lambda x: title(x))

print(titanic_train["Title"].value_counts())
print(titanic_test["Title"].value_counts())


# we can replace a few titles: like there is `Ms which can be replaced with Miss, as they refer to the same gender group. And few like Major, Capt, etc replace them with others

# In[ ]:


for titanic in [titanic_train, titanic_test]:
    titanic["Title"] = titanic["Title"].replace(["Dr", "Rev", "Major", "Col", "Capt", "Lady", "Jonkheer", "Sir", "Don", "Countess", "Dona"], "Others")
    titanic["Title"] = titanic["Title"].replace("Mlle", "Miss")
    titanic["Title"] = titanic["Title"].replace("Ms", "Miss")
    titanic["Title"] = titanic["Title"].replace("Mme", "Mr")


# In[ ]:


# Remove few more columns

titanic_train = titanic_train.drop(["Name", "SibSp", "Parch"], axis=1)
titanic_test = titanic_test.drop(["Name", "SibSp", "Parch"], axis=1)


# In[ ]:


# Print the unique values of the categorical columns
print(titanic_train['Sex'].unique())
print(titanic_train['Embarked'].unique())
print(titanic_train['Title'].unique())


# In[ ]:


titanic_train.isnull().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()
var_mod = ['Sex','Embarked','Title']
for i in var_mod:
    titanic_train[i] = label_encode.fit_transform(titanic_train[i])
    titanic_test[i] = label_encode.fit_transform(titanic_test[i])


# In[ ]:


titanic_train = pd.get_dummies(titanic_train, columns =['Sex','Embarked','Cabin', 'Pclass', 'Title'])
titanic_test = pd.get_dummies(titanic_test, columns =['Sex','Embarked', 'Cabin', 'Pclass', 'Title'])


# In[ ]:


# Split the titanic_train data set into features ``x`` & label ``y``
x = titanic_train.iloc[:,1:22].values
y = titanic_train.iloc[:,0].values

titanic_train.columns


# In[ ]:


# Splitting the data set into 80% Training & 20% Testing
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.2, random_state = 42)
train_x.shape, test_x.shape, train_y.shape, test_y.shape


# In[ ]:


feature_scale = StandardScaler()
train_x = feature_scale.fit_transform(train_x)
test_x = feature_scale.transform(test_x)


# In[ ]:


# from sklearn import utils
# lab_enc = preprocessing.LabelEncoder()
# train_y = lab_enc.fit_transform(train_y)
# print(train_y)
# print(utils.multiclass.type_of_target(train_y))
# print(utils.multiclass.type_of_target(train_y.astype('int')))
# print(utils.multiclass.type_of_target(train_y))


# In[ ]:


# Scaling titanic_test data set as well
scale_test_data = feature_scale.fit_transform(titanic_test)


# In[ ]:


scale_test_data.shape


#  # Creating a Function with multiple models

# In[ ]:


def models(train_x, train_y):
    
    #Logistic Regression
    log_reg = LogisticRegression(random_state = 42)
    log_reg.fit(train_x,train_y)
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(train_x, train_y)
    
    #SVC Linear
    svc_lin = SVC(kernel = 'linear', random_state=42)
    svc_lin.fit(train_x, train_y)
    
    #SVC RBF
    svc_rbf = SVC(kernel = 'rbf', random_state=42)
    svc_rbf.fit(train_x, train_y)
    
    #Decision Tree Classifier
    dec_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dec_tree.fit(train_x, train_y)
    
    #Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=10, random_state=42)
    rf.fit(train_x, train_y)
    
    model=DecisionTreeClassifier(criterion="entropy",max_depth=1)
    AdaBoost=AdaBoostClassifier(base_estimator=model,n_estimators=100,learning_rate=1)
    bostmodel=AdaBoost.fit(train_x,train_y)
    
    xgb_cls =xgb.XGBClassifier(learning_rate =0.1,n_estimators=1000, max_depth=4,min_child_weight=6,gamma=0, subsample=0.8,
                            colsample_bytree=0.8,reg_alpha=0.005,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,
                            seed=27)
    xgb_cls.fit(train_x,train_y)
    #Printing accuracy for every model
    print('[0] Logistic Regression training accuracy: ', log_reg.score(train_x, train_y))
    print('[1] KNN training accuracy: ', knn.score(train_x, train_y))
    print('[2] SVC_Linear training accuracy: ', svc_lin.score(train_x, train_y))
    print('[3] SVC_RBF training accuracy: ', svc_rbf.score(train_x, train_y))
    print('[4] Decision Tree training accuracy: ', dec_tree.score(train_x, train_y))
    print('[5] Random Forest training accuracy: ', rf.score(train_x, train_y))
    
    print('[6] AdaBoostClassifie training accuracy: ', AdaBoost.score(train_x, train_y))
    print('[7] XGBoostClassifie training accuracy: ', xgb_cls.score(train_x, train_y))
        
    return log_reg ,knn, svc_lin, svc_rbf, dec_tree, rf, xgb_cls,AdaBoost


# In[ ]:


# Get and Train all the models
model = models(train_x, train_y)


# In[ ]:


# Printing the prediction of Random Forest
pred = model[4].predict(test_x)
print(pred)

print()

# Printing the actual values
print(test_y)


# In[ ]:


# Creating confusion matrix and see the accuracy for all the models for test data

for i in range( len(model) ):
    cm  = confusion_matrix(test_y, model[i].predict(test_x))
    
    # Extract the confusion matrix parameters
    TN, FP, FN, TP = confusion_matrix(test_y, model[i].predict(test_x)).ravel()
    
    test_score = (TP+TN) / (TP+TN+FP+FN)
    
    print(cm)
    print('Model[{}] Testing Accuracy ="{}"'.format(i, test_score))
    print()


# In[ ]:


dt= model[7]
pred_rand_for = dt.predict(scale_test_data)
submission["Survived"] = pred_rand_for


# In[ ]:


submission.head(6)


# In[ ]:




