# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:52:37 2018

@author: Prashant
"""
#This is KNN - classifier model created by learning techniques from the kernel 
#https://www.kaggle.com/nicapotato/titanic-voting-pipeline-stack-and-guide#NickBrooks

#Will be updating the kernel with new code with time to come

import csv

import pandas as pd

import numpy as np

import random as rnd

import os

import re

#Visualization Import
import seaborn as sns

import matplotlib.pyplot as plt

import scikitplot as skplt

# Supervised Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import feature_selection
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier # <3

# Unsupervised Models
from sklearn.decomposition import PCA

# Evalaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

# Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Esemble Voting
#from mlxtend.classifier import EnsembleVoteClassifier
#from sklearn import metrics
#from sklearn.metrics import classification_report, accuracy_score

# Stacking
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Warnings
import warnings
warnings.filterwarnings('ignore')

import time
import datetime
import platform
start = time.time()
print('Version      :', platform.python_version())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())

print("\nCurrent date and time using isoformat:")
print(datetime.datetime.now().isoformat())

#get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (16, 8)

save = True

# Master Parameters:
n_splits = 5 # Cross Validation Splits
n_iter = 80 # Randomized Search Iterations
scoring = 'accuracy' # Model Selection during Cross-Validation
rstate = 27 # Random State used 
testset_size = 0.30

# Trees Parameters
n_tree_range = st.randint(600, 1200)

# XGboost boosting rounds
num_rounds = 1000

"""
Loading and Preprocessing
"""
#PATH = "G:\\extra things\\Knowledge\\Python_Practice\\"

train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')
train_df.describe()
test_df.describe()

"""
Pre-Processing
combine train/test data to simulatneously apply transformations
"""

Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived',axis=1).copy()

df = pd.concat([train_df,test_df])

df.describe()
df.info()

traindex = train_df.index
testdex = test_df.index

#removes the object from the memeory
del train_df
del test_df    

"""
To understand Feature Engineering
"""
"""
#Feature Engineering

full_data = [train_df , test_df]
full_data
train_df.info()
#1.PClass
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index = 'False').mean()
#2.Sex
train_df[['Sex','Survived']].groupby(['Sex'],as_index = 'False').mean()
#3.SibSp and Parch -Sibling/Spouse, Parent/Children

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index = 'False').mean()

#4.Categorize people as per their lonliness

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1
train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index='False').mean()
"""
"""
Feature Engineering
"""
df.info()
df.describe()
#Family Size

df['FamilySize'] = df ['SibSp']+df['Parch']+1

#Name Length

df['NameLength'] = df['Name'].apply(len)

#IsAlone?
df['IsAlone'] = 0
df.loc[df['FamilySize']==1,'IsAlone'] = 1
df
#Title 

df['Title'] = 0
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.',expand=True)
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
df[['Title','Age']].groupby(['Title'],as_index= 'False').mean()

#Age

df.loc[(df.Age.isnull()) & (df.Title=='Mr'),'Age']=df.Age[df.Title=='Mr'].mean()
df.loc[(df.Age.isnull()) & (df.Title=='Mrs'),'Age']=df.Age[df.Title=='Mrs'].mean()
df.loc[(df.Age.isnull()) & (df.Title=='Miss'),'Age']=df.Age[df.Title=='Miss'].mean()
df.loc[(df.Age.isnull()) & (df.Title=='Master'),'Age']=df.Age[df.Title=='Master'].mean()
df.loc[(df.Age.isnull()) & (df.Title=='Other'),'Age']=df.Age[df.Title=='Other'].mean()

df =df.drop('Name',axis=1)


#Categorical Variable-Emabraked (2NA values)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

#Continuous Variable: Fare
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

#Assigning Binary to string (Sex)variable.

df['Sex'] =df['Sex'].map({'female' : 1, 'male' :0}).astype(int)

#Title
df['Title'] = df['Title'].map({'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Other':4})
df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])
df['Title'] = df['Title'].astype(int)

#Embarked
df['Embarked'] = df['Embarked'].map({'Q':0,'S':1,'C':2}).astype(int)

#We can get rid of Ticket and Cabin variable

df = df.drop(['Ticket','Cabin'],axis=1)

df.head()
"""
After doing feature Engineering we can visualize to see the state of variable, which is neccessary for good output of prediction through machine learning.(Clue:Lookout for Bell Curve for variables, containing min,max,25%,50%,75%,count)
Helps in finding the bias of variable in getting good predictive ability of the models
"""
"""
using traindex to get the state of variable
"""
#Histogram
pd.concat([df.loc[traindex,:],Survived],axis=1).hist()
plt.show()
#Correlation- we see closer to zero corelation for FamilySize
sns.heatmap(pd.concat([df.loc[traindex,:],Survived],axis=1).corr(),annot=True,fmt = ".2f")

#Scaling between -1 and 1, good practice for continuous variables
from sklearn import preprocessing
for col in ['Fare','Age','NameLength']:
    transf =df[col].reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)

#After preprocessing,split the data into train/test data_again
train_df = df.loc[traindex,:]
train_df['Survived']=Survived

test_df = df.loc[testdex,:]

train_df.info()

test_df.info()


#Decide on the dependent and independent variable

X = train_df.drop(['Survived'],axis=1)
y=train_df['Survived']

print ("X,y Test Shape: ",X.shape,y.shape,test_df.shape)

#Storage for models and results

results = pd.DataFrame(columns=['Model','Para','Test_Score','CV Mean','CV Std_Dev'])
ensemble_methods ={}

#Imbalanced DEpendent variable
print("Dependent Variable Distribution")
print(y.value_counts(normalize = True)*100)
print("0 = Died \n1 = Survived")

#Dimensionality Reductions: Principal Components

print("Feature Count (With One Hot Encoding):",X.shape[1])

levels = [2,4,6,8,10,12]
for x in levels:
    pca = PCA(n_components = x)
    fit = pca.fit(train_df)
    print(("{} Components \n Explained Variance: {}\n").format(x,fit.explained_variance_ratio_))
    
#Stratified Train/Test Split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testset_size,stratify=y,random_state=rstate)

X_train.shape,y_train.shape,X_test.shape,y_test.shape

cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rstate)

"""
Helper Functions
Compute,print and save models' Evaluation
"""

def save(model,modelname):
    global results
    
    #Once best_model is found,establish more evaluation metrics
    model.best_estimator_.fit(X_train,y_train)
    scores = cross_val_score(model.best_estimator_,X_train,y_train,cv =         5,scoring=scoring,verbose =0)
    CV_scores = scores.mean()
    STDev = scores.std()
    Test_scores = model.score(X_test,y_test)
    
    #CV and Save Scores
    results = results.append({'Model':modelname,'Para':model.best_params_,'Test_Score':Test_scores,'CVMean':CV_scores,'CV STDEV': STDev},ignore_index =True)
    
    ensemble_methods[modelname]=model.best_estimator_
    
    #Print Evaluation
    
    print("\n Evaluation Method: {}",format(scoring))
    print("Optimal Model Parameters: {}",format(grid.best_params_))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_scores,STDev,modelname))
    print("Test_Score: ",Test_scores)
    
    #Sckit Confusion Matrix
    
    model.best_estimator_.fit(X_train,y_train)
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test,pred,title = "{} Confusion matrix".format(modelname),normalize=True,figsize=(6,6),text_fontsize='large')
    
    plt.show()
    
df1 = pd.DataFrame(columns =['PassengerId','Survived'])
def norm_save(model,score,modelname):
    global results
    global df1
    
    model.fit(X,y)
    submission = model.predict(test_df)
    df1 =df1.append({'PassengerId':test_df.index,'Survived':submission})
    
    CV_score = score.mean()
    Test_scores = model.score(X_test,y_test)
    STDev = score.std()
    
    #CV and save Scores
    
    Test_Score = model.score(X_test,y_test)
    
    results = results.append({'Model':modelname,'Para':model.best_params_,'Test_Score':Test_scores,'CVMean':CV_scores,'CV STDEV': STDev},ignore_index =True)
    
    ensemble_methods[modelname] = model
    
    print("\n Evaluation Method: {}",format(scoring))
    print("Optimal Model Parameters: {}",format(grid.best_params_))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_scores,STDev,modelname))
    print("Test_Score: ",Test_scores)
    
    #Sckit Confusion Matrix
    
    model.best_estimator_.fit(X_train,y_train)
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test,pred,title = "{} Confusion matrix".format(modelname),normalize=True,figsize=(6,6),text_fontsize='large')
    
    plt.show()
    
def eval_plot(model):
    skplt.metrics.plot_roc_curve(y_test,model.predict_proba(X_test))
    plt.show()


#Non-Pramateric models
"""
#K-Nearest Neighbors
"""
param_grid = {'n_neighbors': st.randint(1,40),
              #Increasing this value reduces the bias and increases the variance,dont overfit
              'weights': ['uniform','distance']
              }
#Hyper-prameter Tuning with Cross-Validation
grid = RandomizedSearchCV(KNeighborsClassifier(),
                      param_grid,#HyperParmeters
                      cv = cv,#crossValidations splits
                      scoring = scoring,#Vest validation selection metric
                      verbose=1,#Frequency of model updates
                      n_iter=n_iter,#Number of hyperparameters combinations tried
                      random_state=rstate
                      )

#Execute Tuning on entire training set
grid.fit(X_train,y_train)
save(grid,"KNN")    


results