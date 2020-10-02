#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction 

# ### This study is going to predict which passengers would survive from the Titanic disaster by means of machine learning modelling techniques. It is going present a full machine learning work flow, use 10 Machine Learning algorithms, tune their parameters and ensemble the best n (e.g. 6) of them using their accuracy scores for the validation set. 

# <font color = 'blue'>
#  CONTENTS:  
#     
#    1. [Introduction](#1)
#        * 1.1 [Summary Information about the variables and their types in the data](#1.1)
#    2. [Exploratory Data Analysis](#2)
#        * 2.1 [Importing Libraries](#2.1)
#        * 2.2 [Loading Data](#2.2)
#        * 2.3 [Basic summary statistics about the data](#2.3)       
#    3. [Data Preparation](#3)
#        * 3.1 [Dropping Ticket number and Embarked Variables](#3.1)  
#        * 3.2 [Extraction of Title and Nicknamed variables from Name variable](#3.2)
#            * 3.2.1 [Clustering Title variable](#3.2.1) 
#        * 3.3 [Create Cabin_dummy variable](#3.3)
#        * 3.4 [Outlier Treatment](#3.4)
#        * 3.5 [Missing Value Treatment](#3.5)
#            * 3.5.1 [For Age](#3.5.1)     
#            * 3.5.2 [For Fare](#3.5.2)   
#        * 3.6 [Categorical Variables' Encoding](#3.6)
#            * 3.6.1 [Label encoding of sex variable to a dummy variable (0-1)](#3.6.1)
#            * 3.6.2 [One hot encoding of Title and Pclass](#3.6.2)         
#    4. [Visualizations](#4)
#        * 4.1 [Correlation matrix as heatmap](#4.1)
#        * 4.2 [SibSp versus Survived](#4.2)
#        * 4.3 [Parch versus Survived](#4.3)
#        * 4.4 [Age versus Survived](#4.4)
#    5. [More Feature Engineering and Final Correlation Matrix](#5)
#        * 5.1 [Correlation matrix](#5.1)
#        * 5.2 [Generating small_family, dropping family size and sex](#5.2)
#        * 5.3 [Final correlation matrix as heatmap](#5.3)
#    6. [Modeling, Model Evaluation and Model Tuning](#6)
#        * 6.1 [Splitting the train data](#6.1) 
#        * 6.2 [Validation Set Test Accuracy for the default models](#6.2) 
#        * 6.3 [Cross validation accuracy and std of the default models for all the train data](#6.3)    
#        * 6.4 [Model tuning using crossvalidation](#6.4)   
#        * 6.5 [Ensembling](#6.5) 
#    7. [Submission](#7)
#  

# ## 1.1 Summary Information about the variables and their types in the data <a id = '1.1'></a><br>
# 
# 
# Survival: Survival -> 0 = No, 1 = Yes
# 
# Pclass: Passennger ticket class -> 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower)
# 
# Name: Name of the passenger including title and (if written in quotes) nickname
# 
# Sex: Male or Female
# 
# Age: Age in years
# 
# SibSp: # of Siblings (brother,sister,stepbrother,stepsister) and Spouses (husband or wife) aboard the ship
# 
# Parch: # of Parents and Children  aboard the ship
# 
# Ticket: Ticket code
# 
# Fare: Passenger fare paid
# 
# Cabin: Cabin code
# 
# Embarked: Port of Embark for the passenger -> C = Cherbourg, Q = Queenstown, S = Southampton

# # 2. Exploratory Data Analysis <a id = '2'></a><br> 

# ### 2.1 Importing Libraries <a id = '2.1'></a><br>

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd
import re

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

#timer
import time
from contextlib import contextmanager

# Importing modelling libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} done in {:.0f}s".format(title, time.time() - t0))


# ## 2.2 Loading Data <a id = '2.2'></a><br>

# In[ ]:


# Read train and test data with pd.read_csv():
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## 2.3 Basic summary statistics about the data <a id = '2.3'></a><br>

# ##### Descriptive statistics excluding PassengerId which does not carry any meaningful information for Survival.

# In[ ]:


train.iloc[:,1:len(train)].describe([0.01,0.1,0.25,0.5,0.75,0.99]).T


# In[ ]:


test.iloc[:,1:len(test)].describe([0.01,0.1,0.25,0.5,0.75,0.99]).T


# In[ ]:


print('There seem to be obvious outlier observations for Fare variable.')


# In[ ]:


for var in train:
    if var != 'Survived':
        if len(list(train[var].unique())) <= 10:
                print(pd.DataFrame({'Mean_Survived': train.groupby(var)['Survived'].mean()}), end = "\n\n\n")


# In[ ]:


print('Number of missing values and their percentage for Train and Test sampleS respectively', end = "\n\n")
for df in [train,test]:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    print(pd.concat([total,percent], axis=1, keys=['Total','Percent']), end = "\n\n")


# In[ ]:


print('There are missing observations for Age, Fare,Embarked and Cabin variables.')


# # 3. Data Preparation <a id = '3'></a><br> 

# ## 3.1 Dropping Ticket number and Embarked Variables <a id = '3.1'></a><br>

# We can drop Ticket feature since it is unlikely to have useful information. In addition, although the embark places shows diffent correlations of the passengers is unlikely to be related to the future survival of the passengers.

# In[ ]:


train.drop(['Ticket','Embarked'], axis = 1,inplace=True)   
test.drop(['Ticket','Embarked'], axis = 1,inplace=True)


# ## 3.2 Extraction of Title and Nicknamed variables from Name variable <a id = '3.2'></a><br>

# In the names, there are title and nickname information which may be useful in our analysis. Title gives more information about socioeconomics status, and the people with nicknames may be more lively.

# In[ ]:


for d in [train,test]:
    d['Nicknamed']=d['Name'].apply(lambda x: 1 if '''"''' in x else 0)
    d["Title"] = d["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


train[['Title', 'Age']].groupby(['Title'], as_index=False).mean() 


# In[ ]:


test[['Title', 'Age']].groupby(['Title'], as_index=False).mean()


# In[ ]:


train.groupby("Sex")['Title'].value_counts()


# ### 3.2.1 Clustering Title variable <a id = '3.2.1'></a><br>
# 
# Lady,Madame,Mademoiselle,Don,Dona,Countess and Sir are used to show nobility hence they are groupped in the noble category. Jonkheer and Reverand are at one of the lowest nobility categories, thus they are classified in ordinary 'Mr' Category. 
# (Sources:https://en.wikipedia.org/wiki/Imperial,_royal_and_noble_ranks, https://en.wikipedia.org/wiki/Forms_of_address_in_the_United_Kingdom ,https://en.wikipedia.org/wiki/Don_(honorific))
# 
# 
# Colonels and Majors are among the highest rank army officials so they are also classified in the noble category. As being in a lower rank 'Captain' is classified in ordinary 'Mr' Category. (Source: https://www.va.gov/vetsinworkplace/docs/em_rank.html). 
# 
# 'Dr' titles are classified according to their gender (Mr or Mrs). 

# In[ ]:


for d in [train,test]:
    d['Title'] = d['Title'].replace(['Lady','Mme','Mlle','Don','Col', 'Major', 'Dona','Countess', 'Sir'], 'Noble')
    d['Title'] = d['Title'].replace('Ms', 'Mrs')
    d['Title'] = d['Title'].replace(['Capt','Jonkheer','Rev'], 'Mr')
    d.loc[d['Sex']=='female', 'Title']=d.loc[d['Sex']=='female', 'Title'].replace('Dr', 'Mrs')
    d.loc[d['Sex']=='male', 'Title']=d.loc[d['Sex']=='male', 'Title'].replace('Dr', 'Mr')
    d.drop(['Name'], axis = 1,inplace=True)


# In[ ]:


train[["Title","PassengerId"]].groupby("Title").count()


# In[ ]:


test[["Title","PassengerId"]].groupby("Title").count()


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


train.head()


# In[ ]:


print('New variables are created using Name variable: Title and Nicknamed.')
print('Name variable is dropped')


# ## 3.3 Create Cabin_dummy variable <a id = '3.3'></a><br>

# Cabin_dummy variablecan give imformation about whether someone has a Cabin data or not:

# In[ ]:


for d in [train,test]:
    d["Cabin_dummy"] = d["Cabin"].notnull().astype('int')
    d.drop(['Cabin'], axis = 1, inplace=True)


# In[ ]:


print('Cabin_dummy variable is created using Cabin variable')
print('Cabin variable is dropped')


# ## 3.4 Outlier Treatment <a id = '3.4'></a><br>

# In[ ]:


##Create all sample including test and train data
full_data=pd.concat([train, test], ignore_index=True)


# In[ ]:


sns.boxplot(x = full_data['Fare']);


# In[ ]:


#Defining the upper limit as 99% of all data for winsoring its above
upper_limit = full_data['Fare'].quantile(0.99)
print('Outlier treatment starts...')
print('Repress the Fare variable at maximum to %99 value:','%.2f'% upper_limit )


# In[ ]:


for d in [train,test,full_data]:
    d.loc[d['Fare'] > upper_limit,'Fare'] = upper_limit


# In[ ]:


sns.boxplot(x = full_data['Fare']);


# ## 3.5 Missing Value Treatment <a id = '3.5'></a><br>

# In[ ]:


print('Missing value treatment starts...')


# ### 3.5.1 For Age <a id = '3.5.1'></a><br>

# In[ ]:


train["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)
print('Set the median age of each title for the missing Age values')


# ### 3.5.2 For Fare <a id = '3.5.2'></a><br>

# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test["Fare"].fillna(full_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
print('Set the median Fare of each passenger class for the missing Fare values.')


# In[ ]:


test["Fare"].isnull().sum()


# ## 3.6 Categorical Variables' Encoding <a id = '3.6'></a><br>

# ### 3.6.1 Label encoding of sex variable to a dummy variable (0-1) <a id = '3.6.1'></a><br>

# In[ ]:





# In[ ]:


for d in [train,test]:
    d["Sex"]=d["Sex"].map(lambda x: 0 if x=='female' else 1)


# ### 3.6.2 One hot encoding of Title and Pclass <a id = '3.6.2'></a><br>

# In[ ]:


train, test= [ pd.get_dummies(data, columns = ['Title','Pclass']) for data in [train, test]]


# In[ ]:


train.head()


# In[ ]:


test.head()


# # 4. Visualizations <a id = '3'></a><br> 

# In this section we are going to illustrate the relationship between variables by using visualization tools.

# 
# ## 4.1 Correlation matrix <a id = '4.1'></a><br>

# In[ ]:


# Let's visualize the correlations between numerical features of the train set.
fig, ax = plt.subplots(figsize=(12,6)) 
sns.heatmap(train.iloc[:,1:len(train)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()


# We can see from the heatmaps that "Survived" variable has the highest positive correlations with "Mrs" and "Miss" titles,"Cabin_dummy","PClass" and "Fare" variables. The passengers with nicknames have 17% positive correlation survival which descriptively supports our hypothesis of them for being a useful predictor. Highest negative correlation occurs with 'Mr' title.
# **We can say,**
# * The passengers are more likely to survive if they paid more, if their cabin information is known, and/or they are in first class.
# * Being woman has a very dominant effect on survival.
# * As there is 90% correlation between Sex and 'Mr', it is better to drop Sex variable which is already represented by titles.
# * The correlation is positive for Age and Parch, however it is negative for SibSp. On the other hand, it is only -0.04 for  SibSp and 0.08 for the former ones. It could be better to visualize these variables in detail.

# ## 4.2 SibSp and Survived <a id = '4.2'></a><br>       

# In[ ]:


g= sns.factorplot(x = "SibSp", y = "Survived", data = train, kind = "bar", size = 6)
g.set_ylabels("Survival Probability")
plt.show()


# Having one sibling or spouse has the highest correlation with survival, incrasing the number more than 2 survival probability decreases dractically.  

# ## 4.3 Parch and Survived <a id = '4.3'></a><br>       

# In[ ]:


g= sns.factorplot(x = "Parch", y = "Survived", data = train, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# In[ ]:


train.groupby("Survived")['Parch'].value_counts()


# This variable seems not to have a clear
#   

# ## 4.4 Age versus Survived <a id = '4.4'></a><br>   

# In[ ]:


g= sns.FacetGrid(train, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# The graph on the left hand side shows the distribution of the died passengers while the graph on the right hand side demonstrates the distribution of the survived passengers. 
# * For very old and very young passengers, there are high survival rates with respect to death rates which shows they had been saved priviligously. 
# * Most of the passengers in the Titanic were between the ages of 15-35.
# * Most of the **died passengers** in the Titanic were between the ages of **15-35.**
# * Most of the **survived passengers** in the Titanic were between the ages of **20-35.**

# # 5. More Feature Engineering and Final Correlation Matrix <a id = '5'></a><br>  

# ## 5.1 Family size <a id = '5.1'></a><br>

# Variables "SibSp" and "Parch" give the information about passengers' family, hence we can add them up to reach the family size of the passenger including the passenger himself/herself.

# In[ ]:


# FamilySize: Siblings+Spouse+Parent+Children+1(passenger)
for d in [train,test]:
    d["Familysize"] = d["SibSp"] + d["Parch"] + 1


# In[ ]:


# The relationship between survival rate and FamilySize is. 
g = sns.factorplot(x = "Familysize", y = "Survived", data = train, kind = "bar");
g.set_ylabels("Survival probability");


# ## 5.2 Generating small_family, dropping family size and sex <a id = '5.2'></a><br>

# It is very clear that, the survival is higher if the family size between 1 and 5. Hence, I can crate a new feature of small_family which is 1 if family size between 1 and 5 and 0 else (for single and big families As I mentioned in the previous section, sex variabile has a very high correlation with 'Mr' title and likewise Familysize with its source variables SibSp and Parch. So we drop Familysize and Sex variables.

# In[ ]:


for d in [train,test]:
    d["small_family"] = [1 if 1<i < 5 else 0 for i in d["Familysize"] ]
    d.drop(["Familysize"], axis = 1, inplace=True)
    d.drop(["Sex"], axis = 1, inplace=True)


# In[ ]:


train.groupby("small_family")['Survived'].value_counts()


# Amount of survived small families are considerbaly higher with respect to large families and singles. Amount of deaths are more than two times higher in large families and singles with respect to small families.

# In[ ]:


#  The relationship between survival rate and Familysize is. 
g = sns.factorplot(x = "small_family", y = "Survived", data = train, kind = "bar");
g.set_ylabels("Survival probability");


# Survival probability of small families is almost more than two times higher than big families.

# ## 5.3 Final correlation matrix as heatmap <a id = '5.3'></a><br>

# In[ ]:


fig, ax = plt.subplots(figsize=(12,6)) 
sns.heatmap(train.iloc[:,1:len(train)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()


# <a id = '6'></a><br> 
# # 6. Modeling, Evaluation and Model Tuning  

# ## 6.1 Splitting the train data <a id = '6.1'></a><br>
The given train set is splitted again into inner train and validation sets to test the accuracy of training with the untrained 20% of the sample.
# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


x_val.shape


# ## 6.2 Validation Set Accuracy for the default models <a id = '6.2'></a><br>

# In[ ]:


r=1309
models = [LogisticRegression(random_state=r),GaussianNB(), KNeighborsClassifier(),
          SVC(random_state=r,probability=True),DecisionTreeClassifier(random_state=r),
          RandomForestClassifier(random_state=r), GradientBoostingClassifier(random_state=r),
          XGBClassifier(random_state=r), MLPClassifier(random_state=r),
          CatBoostClassifier(random_state=r,verbose = False)]
names = ["LogisticRegression","GaussianNB","KNN","SVC",
             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]


# In[ ]:



print('Default model validation accuracies for the train data:', end = "\n\n")
for name, model in zip(names, models):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val) 
    print(name,':',"%.3f" % accuracy_score(y_pred, y_val))


# ## 6.3 Cross validation accuracy and std of the default models for all the train data <a id = '6.3'></a><br>

# In[ ]:


results = []
print('10 fold Cross validation accuracy and std of the default models for the train data:', end = "\n\n")
for name, model in zip(names, models):
    kfold = KFold(n_splits=10, random_state=1001)
    cv_results = cross_val_score(model, predictors, target, cv = kfold, scoring = "accuracy")
    results.append(cv_results)
    print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))


# ## 6.4 Model tuning using crossvalidation <a id = '6.4'></a><br>

# In[ ]:


# Possible hyper parameters
names = ["LogisticRegression","GaussianNB","KNN","SVC",
             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]
logreg_params= {"C":np.logspace(-1, 1, 10),
                    "penalty": ["l1","l2"], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], "max_iter":[1000]}

NB_params = {'var_smoothing': np.logspace(0,-9, num=100)}
knn_params= {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
svc_params= {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1, 5, 10 ,50 ,100],
                 "C": [1,10,50,100,200,300,1000]}
dtree_params = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}
rf_params = {"max_features": ["log2","Auto","None"],
                "min_samples_split":[2,3,5],
                "min_samples_leaf":[1,3,5],
                "bootstrap":[True,False],
                "n_estimators":[50,100,150],
                "criterion":["gini","entropy"]}
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [100,500,100],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [100,500,100],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}

xgb_params ={
        'n_estimators': [50, 100, 200],
        'subsample': [ 0.6, 0.8, 1.0],
        'max_depth': [1,2,3,4],
        'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5],
        "min_samples_split": [1,2,4,6]}

mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100),
                                     (3,5), 
                                     (5, 3)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}
catb_params =  {'depth':[2, 3, 4],
              'loss_function': ['Logloss', 'CrossEntropy'],
              'l2_leaf_reg':np.arange(2,31)}
classifier_params = [logreg_params,NB_params,knn_params,svc_params,dtree_params,rf_params,
                     gbm_params, xgb_params,mlpc_params,catb_params]               
                  


# In[ ]:


# Tuning by Cross Validation  
cv_result = {}
best_estimators = {}
for name, model,classifier_param in zip(names, models,classifier_params):
    with timer(">Model tuning"):
        clf = GridSearchCV(model, param_grid=classifier_param, cv =10, scoring = "accuracy", n_jobs = -1,verbose = False)
        clf.fit(x_train,y_train)
        cv_result[name]=clf.best_score_
        best_estimators[name]=clf.best_estimator_
        print(name,'cross validation accuracy : %.3f'%cv_result[name])


# In[ ]:


accuracies={}
print('Validation accuracies of the tuned models for the train data:', end = "\n\n")
for name, model_tuned in zip(best_estimators.keys(),best_estimators.values()):
    y_pred =  model_tuned.fit(x_train,y_train).predict(x_val)
    accuracy=accuracy_score(y_pred, y_val)
    print(name,':', "%.3f" %accuracy)
    accuracies[name]=accuracy


# ####  Extracting first n (e.g. 6) models

# In[ ]:


n=6
accu=sorted(accuracies, reverse=True, key= lambda k:accuracies[k])[:n]
firstn=[[k,v] for k,v in best_estimators.items() if k in accu]


# In[ ]:


# Ensembling First n Score

votingC = VotingClassifier(estimators = firstn, voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_val),y_val))


# # 7. Submission  <a id = '7'></a><br>

# In[ ]:


ids = test['PassengerId']
x_test=test.drop('PassengerId', axis=1)
predictions = votingC.predict(x_test)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission_first{}.csv'.format(n), index=False)


# In[ ]:




