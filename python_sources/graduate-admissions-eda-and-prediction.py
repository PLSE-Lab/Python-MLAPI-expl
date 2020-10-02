#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()


# In[ ]:


data.tail()


# * I would like to add some informations about our features and what do they mean.
# * GRE Scores (out of 340)
# * TOEFL Scores (out of 120)
# * University Rating (out of 5)
# * Statement of Purpose and Letter of Recommendation Strength (out of 5)
# * Undergraduate GPA (out of 10)
# * Research Experience (either 0 or 1)
# * Chance of Admit (ranging from 0 to 1)

# In[ ]:


data.info()


# * We have no missing values, this is a good thing.
# * We already have index so serial no is unnecessary at this point, so i am just going to drop it.

# In[ ]:


data=data.drop("Serial No.",axis=1)


# In[ ]:


data.head()


# In[ ]:


data.columns


# * We have to fix the column names, if you look at "LOR" and "Chance of Admit" you'll see there is a space in the end. 
# * And we dont want space between words, so i am going to add underscore between words, and switch all uppercase letters to lowercase.

# In[ ]:


data.columns = ["gre_score","toefl_score","uni_rating","sop","lor","cgpa","research","admit_chance"]


# In[ ]:


data.columns


# In[ ]:


data.describe()


# * First lets look at correlation matrix.

# In[ ]:


corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()


# * Other then research, all of our features impact chance of admit more then 0.6 rate. And most effective is cgpa with 0.88.
# * And we have some nice correlations between our features. These are:
# * toefl and gre score
# * cgpa and gre score
# * and cgpa and toefl score
# * These all have correlation higher than 0.8.
# * We'll take a look at all these correlations.

# * But before we dive into the features, i'll seperate the chance of admit to two classes:
#     * if it is 0.7 or above it's 1
#     * if lower it's 0

# In[ ]:


def modify(feature):
    if feature["admit_chance"] >= 0.7:
        return 1
    return 0 

data["admit_chance"] = data.apply(modify,axis=1)
data.head()


# * Now we can look at distributions of our data.

# ### Visualization

# In[ ]:


sns.countplot(x = "admit_chance",data=data,palette="RdBu")
plt.show()


# In[ ]:


c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"gre_score",bins=25)
plt.show()
data.groupby("admit_chance")["gre_score"].mean()


# * We can see how gre scores distributed around 300 - 310 for admit chance = 0 
# * And of course it is higher for admit chance = 1, we have peak around 325.

# In[ ]:


c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"toefl_score",bins=25)
plt.show()
data.groupby("admit_chance")["toefl_score"].mean()


# * Most of those who admitted's scores are around 110 and that is what visualization's tells us.

# In[ ]:


c = sns.FacetGrid(data,col="admit_chance",height=6)
c.map(sns.distplot,"cgpa",bins=25)
plt.show()
data.groupby("admit_chance")["cgpa"].mean()


# In[ ]:


sns.distplot(data["cgpa"])
plt.show()


# In[ ]:


sns.catplot(x="research",y="admit_chance",data=data,kind="bar")
plt.show()


# * Most of who didnt have research experience have low chance of admit as expected.

# In[ ]:


sns.catplot(x="uni_rating",y="admit_chance",data=data,kind="bar")
plt.show()


# * Uni ratings 1 and 2 have very low chance of admit. I am going to group 1-2 as 0 and the rest as 1. 

# In[ ]:


def modify(feature):
    if feature["uni_rating"] >= 3:
        return 1
    return 0 

data["uni_rating"] = data.apply(modify,axis=1)
data.head()


# In[ ]:


sns.catplot(x="uni_rating",y="admit_chance",data=data,kind="bar")
plt.show()


# In[ ]:


sns.catplot(x="sop",y="admit_chance",data=data,kind="bar")
plt.show()


# * Same goes for here too. If statement of purpose rating is lower then 3 it's highly unlikely that you'll accepted. So im going to group 1 to 2.5 as 0 and the rest as 1.

# In[ ]:


def modify(feature):
    if feature["sop"] >= 3.0:
        return 1
    return 0 

data["sop"] = data.apply(modify,axis=1)
data.head()


# In[ ]:


sns.catplot(x="sop",y="admit_chance",data=data,kind="bar")
plt.show()


# In[ ]:


sns.catplot(x="lor",y="admit_chance",data=data,kind="bar")
plt.show()


# * We have some exceptions here but same goes for letter of recommendation too. Lower then 0 to 2.5 as 0 and rest is 1.

# In[ ]:


def modify(feature):
    if feature["lor"] >= 3.0:
        return 1
    return 0 

data["lor"] = data.apply(modify,axis=1)
data.head()


# In[ ]:


sns.catplot(x="lor",y="admit_chance",data=data,kind="bar")
plt.show()


# * Now we checked and grouped our features. We can take a look at correlations between cgpa, toefl and gre scores.
# 

# In[ ]:


sns.scatterplot(data = data, x ="gre_score", y="toefl_score",hue="admit_chance")
plt.show()


# * We have some exceptions like the little blue dot in the middle, but rest is expected. Students that have higher gre score usually have high toefl scores.

# In[ ]:


sns.scatterplot(data = data, x ="gre_score", y="cgpa",hue="admit_chance")
plt.show()


# * Same goes for here too, except we have not much outliers in here. Students with higher cgpa (higher than 9) usually admitted to university. 

# In[ ]:


sns.scatterplot(data = data, x ="toefl_score", y="cgpa",hue="admit_chance")
plt.show()


# * Same trend goes for here too.

# ### Modeling

# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb
from xgboost import XGBClassifier


# In[ ]:


x = data.drop("admit_chance",axis = 1)
y = data["admit_chance"]

scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print("x_train:", len(x_train))
print("x_test:", len(x_test))
print("y_train:", len(y_train))
print("y_test:", len(y_test))


# * First i want to fit a logistic regression model and then tune the parameters and do grid search.

# In[ ]:


logreg = LogisticRegression(max_iter=500)

logreg.fit(x_train,y_train)

print("Train Accuracy:", logreg.score(x_train,y_train))
print("Test Accuracy:", logreg.score(x_test,y_test))


# * So we have 82% test accuracy, let's see if we can get it up 90%.

# ### Hyperparameter tuning and GridSearchCV

# In[ ]:


classifier = [KNeighborsClassifier(),
              DecisionTreeClassifier(random_state=42),
              LogisticRegression(random_state=42),
              SVC(random_state=42),
              RandomForestClassifier(random_state=42),
              XGBClassifier(random_state=42, objective="binary:logistic")]

knn_params = {"n_neighbors": np.linspace(1,19,10,dtype=int),
                 "weights": ["uniform","distance"],
                 "metric": ["euclidean","manhattan","minkowski"]}

dt_params = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)} 

lr_params = {"C":np.logspace(-3,3,7),
             "penalty": ["l1","l2"]}

svm_params = {"kernel" : ["rbf"],
              "gamma": [0.001, 0.01, 0.1, 1],
              "C": [1,10,50,100,200,300,1000]}

rf_params = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

xgb_params = {"learning_rate":[0.01,0.1,1],
              "n_estimators":[50,100,150],
              "max_depth":[3,5,7],
              "gamma":[1,2,3,4]}

classifier_params = [knn_params,
                     dt_params,
                     lr_params,
                     svm_params,
                     rf_params,
                     xgb_params]


# In[ ]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(estimator = classifier[i], param_grid = classifier_params[i],
                       cv = StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs= -1, verbose= 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results = pd.DataFrame({"ML Models":["KNeighborsClassifier",
                                        "Decision Tree",
                                        "Logistic Regression",
                                        "SVM",
                                        "Random Forest",
                                        "XGBClassifier"],
                           "Cross Validation Means":cv_result})
print(cv_results)


# * So here we have a winner: Logistic Regression with 87.7% accuracy.
# * We have used many classifiers but LR outperformed all.
# * If you like my notebook please upvote and if i have mistakes please tell me.
