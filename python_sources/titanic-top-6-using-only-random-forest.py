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


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from sklearn import preprocessing


# In[ ]:


def init():
    test_df = pd.read_csv("../input/titanic/test.csv")
    train_df = pd.read_csv("../input/titanic/train.csv")
    test_df["Survived"]=-8888
    Main_df=pd.concat((train_df, test_df), axis=0, sort=False)
    print("Columns with null values {}".format(Main_df.columns[Main_df.isnull().any()]))


    global title_map
    title_map={
    "mr": "Mr","mrs": "Mrs","miss": "Miss","master": "Master","don": "Royal", "rev": "Officer", "dr": "Officer",
    "mme": "Mrs", "dona": "Royal", "jonkheer" : "Royal", "the countess": "Royal", "capt": "Officer", "col": "Officer",
    "ms": "Mrs", "mlle": "Miss", "major": "Officer", "lady" : "Royal", "sir" : "Royal"
    }
    return(Main_df)


# **Helper Functions:**
# 
# **Get Title:** 
# > The function will get the titles of each passanger. titles are broadly divided in to following categories.
# 1. Mr
# 2. Mrs
# 3. Master
# 4. Royal families
# 5. Officers
# 6. Miss
# 
# **Is In Group:**
# > In my opinion , ticket number is most un explored feature. I have used it extensively in indentifying the patterns
# Based on ticket number, The function will calculate/find the 
# * group size : Many kernels uses Family size. i think that do not consider the fact that many people are travelling in a group.
# * actual fare : the tickets booked in groups must have common fare. it make sense to divide the fare by group size.  
# * special tickets in third class/Private cabins etc have been also considered.
# 
# 
# **Age:**
# We will impute the missing age values.
# 
# 
# **Visualization**
# * Created several visualization to show the relationship of critical features w.r.t Survival.
# 
# The code has been modularized for better reading . I have also used Grid search to fine tune the hyper parameters as well as used Cross fold validation to get the mean and standard deviation of each algorithm.
# 
# pls upvote if you like the kernel :-) 
# 

# In[ ]:


def get_title(x):
    x=x.split(",")[1].split(".")[0].strip().lower()
    return(title_map[x])

def is_in_group(df):
    gdf=pd.DataFrame({"GroupSize": df.groupby("Ticket")["Ticket"].count()}).reset_index()
    df=pd.merge(df, gdf, how="left", on="Ticket").set_index(df.index)
   
    df["Fare"].fillna(8.0, inplace=True) # Median fare for this route.
    df["ActualFare"]=df["Fare"].astype(int)/df["GroupSize"]
    df["IsMother"]= np.where((df["Sex"]=="female") & (df["Parch"] > 0) & (df["Title"] != "Miss"), 1, 0) # Mothers
    
    df['ST_third_class'] = np.where((df['Ticket'].str.match(pat='(SOTON)|(W./C.)|(A/5)')) & (df["Pclass"]==3) & (df["Cabin"].isnull()), 1, 0)
    df["PC"]= np.where((df["Pclass"]==1) & (df['Ticket'].str.match(pat='(PC)')) & (df["Cabin"].isnull()), 1, 0)
    df['ST_WC'] = np.where((df['Ticket'].str.match(pat='(W./C.)')) & (df["Cabin"].isnull()), 1, 0)
    return(df)

def adjust_columns(df):
    columns=["Title", "Sex","Pclass", "Embarked"]
    df=pd.get_dummies(df, columns=columns)
    df.drop(["Name", "Ticket", "Cabin", "Fare", "Cabin"], axis=1, inplace=True)
    df.drop(["Title_Royal"], axis=1, inplace=True) # Not significant contribution

    return(df)

def age(df):
    
    #null mothers age will be given median age of existing mothers
    cond = df["IsMother"]==1 
    median = df.loc[cond, "Age"].median()
    df.loc[cond, "Age"]=df.loc[cond, "Age"].fillna(median)

    # Median age of adults with no childrens and no parents
    cond=df["Parch"]==0 
    median = df.loc[cond, "Age"].median()
    df.loc[cond, "Age"]=df.loc[cond, "Age"].fillna(median)

 

    #remaining null values substituted by median age of Sex+Title
    df["Age"].fillna(df.groupby(['Sex','Title'])["Age"].transform("median"), inplace=True) # Fill age based on median 
    
    #Now let's see the distribution of age
    fig, (axis1) = plt.subplots(1,1, figsize=(10,5))
    sn.distplot(df["Age"], ax=axis1)
    
    
    return(df)


# In[ ]:


if __name__=="__main__":
    df=init() # Init
    
    df["Title"]=df["Name"].map(get_title) #Getting title for each person
    df=is_in_group(df) # Find group based on tickets and family members#.
    df=age(df) # get Age values for null Age

    ##Get the survival rate of key features
    sex_survival_rate=df.loc[df["Survived"]!=-8888, ["Sex", "Survived"]].groupby("Sex").mean().reset_index()
    pclass_survival_rate=df.loc[df["Survived"]!=-8888, ["Pclass", "Survived"]].groupby("Pclass").mean().reset_index()
    Group_survival_rate=df.loc[df["Survived"]!=-8888, ["GroupSize", "Survived"]].groupby("GroupSize").mean().reset_index()
    Title_survival_rate=df.loc[df["Survived"]!=-8888, ["Title", "Survived"]].groupby("Title").mean().reset_index()
    ST_survival_rate=df.loc[df["Survived"]!=-8888, ["ST_third_class", "Survived"]].groupby("ST_third_class").mean().reset_index()


    fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(1,5, figsize=(15,5))
    sn.barplot(x='Sex', y='Survived', data=sex_survival_rate, ax=axis1)
    sn.barplot(x='Pclass', y='Survived', data=pclass_survival_rate, ax=axis2)
    sn.barplot(x='GroupSize', y='Survived', data=Group_survival_rate, ax=axis3)
    sn.barplot(x='Title', y='Survived', data=Title_survival_rate, ax=axis4)
    sn.barplot(x='ST_third_class', y='Survived', data=ST_survival_rate, ax=axis5)



    
    scaler = preprocessing.StandardScaler()
    df["ActualFare"]=scaler.fit_transform(np.array(df["ActualFare"].values).reshape(-1,1))
    df["Age"]=scaler.fit_transform(np.array(df["Age"].values).reshape(-1,1))
    df["GroupSize"]=scaler.fit_transform(np.array(df["GroupSize"].values).reshape(-1,1))
    df["Parch"]=scaler.fit_transform(np.array(df["Parch"].values).reshape(-1,1))
    df["SibSp"]=scaler.fit_transform(np.array(df["SibSp"].values).reshape(-1,1))

    print("Columns with null values {}".format(df.columns[df.isnull().any()]))

    df=adjust_columns(df) # Feature encoding and selections.


# In[ ]:


#Now model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score, KFold, learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


# In[ ]:


train=df.loc[df["Survived"]!=-8888]
test=df.loc[df["Survived"]==-8888]

cols=list(train.columns)
cols.remove("Survived")
cols.remove("PassengerId")
print(cols)

X_train=train.drop(["Survived","PassengerId"], axis=1).values
Y_train=train["Survived"].values

X_test=test.drop(["Survived","PassengerId"], axis=1).values
print(df.shape, X_train.shape, Y_train.shape, X_test.shape)


# In[ ]:


#Let's use cross validation to find best model
kfold = StratifiedKFold(n_splits=5)

clf=["RandomForest", "LogisticRegression", "KNN"]

random_state = 2
classifiers = []
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(KNeighborsClassifier(n_neighbors=5))

results, means, std = [], [], []
for classifier in classifiers :
    results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

for result in results:
    means.append(result.mean())
    std.append(result.std())

result_DF=pd.DataFrame({"Classifiers":clf, "ResultMeans":means, "ResultSTD":std })
print(result_DF)


# In[ ]:


def grid_search_param(algo):
    if algo == "Logistic":
        clf=LogisticRegression()
        c_space = np.logspace(-5, 8, 15) 
        penalty=["l1", "l2"]
        param_grid = {'C': c_space, 'penalty' : penalty}
    elif algo=="RandomForest":
        clf=RandomForestClassifier()
        param_grid = { 'n_estimators': [700],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth' : [6],
                      'criterion' :['gini', 'entropy'],
                     }  
    elif algo=="KNN":
        clf=KNeighborsClassifier()

        param_grid ={'n_neighbors' : [6,7,8,9,10,11,12,14,16,18,20,22],
                    'algorithm' : ['auto'],
                    'weights' : ['uniform', 'distance'],
                    'leaf_size' : list(range(1,50,5))
                    }
    return(param_grid, clf)

def get_best_param(clf, param):
    grid = GridSearchCV(clf, param, cv = 10) 
    grid.fit(X_train, Y_train) 
    print("Tuned   Parameters: {}".format(grid.best_params_))  
    print("Best score is {}".format(grid.best_score_)) 
    return(grid)


def run_clf(grid, clf, algo):
    if algo == "Logistic":
        c_value=grid.best_params_["C"]
        penalty=grid.best_params_["penalty"]
        print("Best parameter: C is {} and Penalty is {}".format(c_value, penalty))
        clf=LogisticRegression(C=c_value, penalty=penalty)
    
    elif algo=="RandomForest":
        n_estimators=grid.best_params_["n_estimators"]
        max_features=grid.best_params_["max_features"]
        max_depth=grid.best_params_["max_depth"]
        criterion=grid.best_params_["criterion"]
        clf=RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion)
    
    elif algo=="KNN":
        n_neighbors=grid.best_params_["n_neighbors"]
        algorithm=grid.best_params_["algorithm"]
        weights=grid.best_params_["weights"]
        leaf_size=grid.best_params_["leaf_size"]
        clf=KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm,weights=weights,leaf_size=leaf_size)
        
    clf.fit(X_train, Y_train)
    print(confusion_matrix(Y_train, clf.predict(X_train)))
    print("{} score is {}".format(algo, clf.score(X_train,Y_train)))
    return(clf)

        


# In[ ]:



print("Running Logistic regression")
param, clf=grid_search_param("Logistic")
(grid)= get_best_param(clf, param)
clfLR=run_clf(grid, clf, "Logistic")
pred_values=clfLR.predict(X_test)
test["Survived"]=pd.Series(pred_values, index=test.index)
test[["PassengerId", "Survived"]].to_csv("LRPrediction.csv", index=False)

print("Running Random Forest")
param, clf=grid_search_param("RandomForest")
(grid)= get_best_param(clf, param)
clfRF=run_clf(grid, clf, "RandomForest")
importances = clfRF.feature_importances_
DF=pd.DataFrame({"Features": cols, "Importance":  importances})
print(DF.sort_values(by ='Importance', ascending = False))

pred_values=clfRF.predict(X_test)
test["Survived"]=pd.Series(pred_values, index=test.index)
test[["PassengerId", "Survived"]].to_csv("RFPrediction.csv", index=False)


#print("Running KNN")
#param, clf=grid_search_param("KNN")
#(grid)= get_best_param(clf, param)
#clfKNN=run_clf(grid, clf, "KNN")
#pred_values=clfKNN.predict(X_test)
#test["Survived"]=pd.Series(pred_values, index=test.index)
#test[["PassengerId", "Survived"]].to_csv("KNNPrediction.csv", index=False)

