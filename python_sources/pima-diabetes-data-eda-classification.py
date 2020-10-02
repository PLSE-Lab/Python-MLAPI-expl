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
from scipy.stats import skew,skewtest
from sklearn.impute import KNNImputer
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()


# * We can see that we have some missing values there and they all showing with 0 so we don't know how many. Let's check other things about our data and then we can replace our missing values, in this case 0's, with NaN's.

# In[ ]:


data.info()


# In[ ]:


data.describe().T


# * Yes, we have missing data but which columns? We can see here that our Glucose, BloodPressure, SkinThcikness, Insulin and BMI features all have minimum values 0. This can't be right? Now we also found which features have missing values.

# In[ ]:


data.columns
# I checked that if features has space or not in their names. We dont want to search for mistakes later.


# In[ ]:


data.tail()


# In[ ]:


data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.nan)
print(data.isnull().sum())


# * So Glucose doesnt have much values alose BloodPressure too. But SkinThickness has one forth of data missing and Insulin is almost fourty percent missing value too. Before we dive in to our models and so we need to fill these missing values.
# 
# * What are the choices we have here?
# 
# * We have few ways that filling missing values but before that let's check our data distributions to understand our data.

# In[ ]:


p = data.hist(figsize = (20,20),bins=25)
plt.show()


# * Well, it's seems like we have some skewness in our data distributions. 
# * What is skewness?
# * Skewness is asymmetry in a statistical distribution, in which the curve appears distorted or skewed either to the left or to the right. Skewness can be quantified to define the extent to which a distribution differs from a normal distribution.
# * Now let's check our distributions to outcomes, and see if we have same distribution.

# In[ ]:


c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"Glucose",bins=25)
plt.show()


# In[ ]:


c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"BloodPressure",bins=25)
plt.show()


# In[ ]:


c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"SkinThickness",bins=25)
plt.show()


# In[ ]:


c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"Insulin",bins=25)
plt.show()


# In[ ]:


c = sns.FacetGrid(data,col="Outcome",height=6)
c.map(sns.distplot,"BMI",bins=25)
plt.show()


# * We have somewhat identical distributions. 
# * Now let's check our correlation map and fill our missing values.

# In[ ]:


corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()


# ### Fill Missing Value
# * So like i said, we have a few ways to fill our values. Calculating mean or median of each missing column is a way, and mostly preferred way. But with that much skewness in data if we add more that wont be good for us.
# * Let's use a different approach and use KNNImputer, and see if we have improvement.

# In[ ]:


imputer = KNNImputer(n_neighbors=5)
data_filled = imputer.fit_transform(data)
data_filled_df = pd.DataFrame(data_filled, index=data.index,columns=data.columns)


# In[ ]:


data_filled_df.isnull().sum()
# All missing values are filled.


# In[ ]:


corr = data_filled_df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax,)
plt.show()


# * Now, we replaced all our missing values using KNNImputer. After the change, our correlation values have slightly changed. 
# * We have our missing values filled, now we can start to build our models.
# 
# ### Modeling

# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# * After we scale our data, we'll do a quick and simple KNN fit and see our accuracy.

# In[ ]:


X = data_filled_df.drop("Outcome",axis=1)
y = data_filled_df["Outcome"]


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

print("Train Accuracy: ",knn.score(X_train,y_train))
print("Test Accuracy: ",knn.score(X_test,y_test))


# * Now our results are not satisfying, like expected. We have 67% test accuracy. Let's tune our hyperparameters with Grid Search and see where it goes.

# ### Hyperparameter Tuning, Grid Search

# In[ ]:


random_state = 42
classifier = [SVC(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier(),
             xgb.XGBClassifier(seed=123, objective="reg:logistic")]

svc_params = {"kernel" : ["rbf"],
              "gamma": [0.001, 0.01, 0.1, 1],
              "C": [1,10,50,100,200,300,1000]}

logreg_params = {"C":np.logspace(-3,3,7),
                 "penalty": ["l1","l2"]}

knn_params = {"n_neighbors": np.arange(1,20,1),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan","minkowski"]}

xgb_params = {"n_estimators": np.arange(1,15,1),
              "max_depth" : np.arange(1,10,1),
              "num_boost_round": np.arange(1,10,1)}

classifier_params = [svc_params,
                    logreg_params,
                    knn_params,
                    xgb_params]


# In[ ]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_params[i], cv = 5, scoring = "accuracy", n_jobs = -1,verbose=1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results = pd.DataFrame({"ML Models":["SVM",
                                        "LogisticRegression",
                                        "KNeighborsClassifier",
                                        "XGBClassifier"],
                           "Cross Validation Means":cv_result})
print(cv_results)


g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# * So we improved our accuracy 10% with hyperparameter tuning, and the best one is XGBClassifier with 78.5% accuracy. 
# 
# * And thanks for all readers, this is my first notebook so if you've found any mistakes please tell me so i can improve myself. 
