#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import seaborn as sns
sns.set()
import os


# In[ ]:


#loading data
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train.shape


# In[ ]:


#Display top 3 rows
train.head(3)


# In[ ]:


# Getting information of non-null objects
train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


round(train.describe(include = 'all', percentiles=[0.01,0.1,0.25,0.75,0.99]))


# In[ ]:


#STATISTICS
## Mean: Average value of a column
train.Fare.sum()/len(train.Fare)
np.mean(train.Fare)
train.Fare.mean()


# In[ ]:


# Median: middle value after sorting the column
train.Fare.median()


# In[ ]:


# Mode: Value that appears most frequently in the data
train.Fare.mode()
train.Fare.value_counts()


# In[ ]:


train.Fare.std()


# In[ ]:


train.Fare.var()


# In[ ]:


train.columns


# In[ ]:


train.Pclass.value_counts()


# In[ ]:


# distribution
train.Fare.plot.hist()


# In[ ]:


#Analyzing the data
sns.countplot(x="Survived", data=train)


# In[ ]:


#checking survival rate of each Sex
sns.countplot(x="Survived", hue="Sex", data=train)


# In[ ]:


#checking survival rate of passengers from each class
sns.countplot(x="Survived", hue="Pclass", data=train)


# In[ ]:


#clasification
train.info()


# In[ ]:


train.Age.isnull().sum()


# In[ ]:


train.head()


# In[ ]:


#Data cleaning
# Drop unnecessary columns or columns with lot of missing values
train = train.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)
train.head()


# In[ ]:


#Data cleaning
# Replace missing value in age with
train['Age'] = train['Age'].fillna(train['Age'].mean())


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


# Change sex variable to have numeric values
train['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


train.head()


# In[ ]:


# importing the libraries to split the data and Splitting the data
from sklearn.model_selection import train_test_split
#training and testing data
train,test = train_test_split(train, test_size=0.3, random_state=0, stratify=train['Survived'])


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head(3)


# In[ ]:


train.columns


# In[ ]:


# separating dependent and independent variables
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]


# In[ ]:


train_X.head()


# In[ ]:


train_Y.head()


# In[ ]:


# Importing and applying logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


model.fit(train_X,train_Y)


# In[ ]:


test_X.head()


# In[ ]:


predictions=model.predict(test_X)


# In[ ]:


test_X.shape


# In[ ]:


predictions


# In[ ]:


# Importing libraries for evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


test_Y.head()


# In[ ]:


metrics.accuracy_score(test_Y,predictions)


# In[ ]:


metrics.precision_score(test_Y,predictions)


# In[ ]:


metrics.recall_score(test_Y,predictions)


# In[ ]:


pd.DataFrame(confusion_matrix(test_Y,predictions),columns=["Predicted No", "Predicted Yes"],index=["Actual No","Actual yes"] )


# In[ ]:


#Accuracy
(149+72)/(149+16+31+72)


# In[ ]:


# Precision
72/(16+72)


# In[ ]:


# Recall
72/(31+72)


# In[ ]:


#AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
# Getting prediction probabilities
probs = model.predict_proba(test_X)[:, 1]


# In[ ]:


# Calculating ROC_AUC
from sklearn.metrics import roc_curve, auc
FPR, TPR, thresholds = roc_curve(test_Y, probs)
ROC_AUC = auc(FPR, TPR)
ROC_AUC


# In[ ]:


# Plotting ROC_AUC
import matplotlib.pyplot as plt
plt.plot(FPR,TPR)
plt.xlabel('False Positive Rate: FP/(FP+TN)', fontsize = 18)
plt.ylabel('True Positive Rate (Recall)', fontsize = 18)
plt.title('ROC for survivors', fontsize= 18)
plt.show()


# In[ ]:


#Cross-Validation
# Applying cross validation
# StratifiedShuffleSplit: stratified randomized folds
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


test.head()


# In[ ]:


X = train[train.columns[1:]]
y = train[train.columns[:1]]


# In[ ]:


## saving the feature names for decision tree display
column_names = X.columns
# Scaling the data
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[ ]:


X


# In[ ]:


accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X, y, cv = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))


# In[ ]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]
penalties = ['l1','l2']
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)
param = {'penalty': penalties, 'C': C_vals}
grid = GridSearchCV(estimator=LogisticRegression(),
param_grid = param,
scoring = 'accuracy',
n_jobs = -1,
cv = cv
)
grid.fit(X, y)


# In[ ]:


## Getting the best of everything.
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


lr_grid = grid.best_estimator_
lr_grid.score(X,y)


# In[ ]:


#KNN
## Importing the model.
from sklearn.neighbors import KNeighborsClassifier
## calling on the model oject.
knn = KNeighborsClassifier(metric='minkowski', p=2)
## doing 10 fold staratified-shuffle-split cross validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)
accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))


# In[ ]:


## Search for an optimal value of k for KNN.
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')
    k_scores.append(scores.mean())
print("Accuracy scores are: {}\n".format(k_scores))
print ("Mean accuracy score: {}".format(np.mean(k_scores)))


# In[ ]:


from matplotlib import pyplot as plt
plt.plot(k_range, k_scores, c = 'green')


# In[ ]:


from sklearn.model_selection import GridSearchCV
## trying out multiple values for k
k_range = range(1,31)
weights_options=['uniform','distance']
param = {'n_neighbors':k_range, 'weights':weights_options}
## Using startifiedShufflesplit.
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available process
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)
## Fitting the model.
grid.fit(X,y)


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


### Using the best parameters from the grid-search.
knn_grid= grid.best_estimator_
knn_grid.score(X,y)


# In[ ]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gaussian = GaussianNB()
gaussian.fit(train_X, train_Y)
predictions = gaussian.predict(test_X)
gaussian_accy = round(accuracy_score(test_Y, predictions), 3)
print(gaussian_accy)


# In[ ]:


#SVM
from sklearn.svm import SVC
Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term
gammas = [0.0001,0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
## 'rbf' stands for gaussian kernel
grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv)
grid_search.fit(X,y)


# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)


# In[ ]:


# using the best found hyper paremeters to get the score.
svm_grid = grid_search.best_estimator_
svm_grid.score(X,y)


# In[ ]:


#Decision Trees
from sklearn.tree import DecisionTreeClassifier
max_depth = range(1,20)
max_feature = [20,22,24,26,28,30,'auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth,
         'max_features':max_feature,
         'criterion': criterion
        }
grid = GridSearchCV(DecisionTreeClassifier(),
                    param_grid = param,
                     verbose=False,
                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True),
                    n_jobs = -1)
grid.fit(X, y)


# In[ ]:


print( grid.best_params_)
print (grid.best_score_)
print (grid.best_estimator_)


# In[ ]:


dectree_grid = grid.best_estimator_
dectree_grid.score(X,y)


# In[ ]:


#Bagging
#Random Forest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150,155,160];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions}
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                    param_grid=parameters,
                    cv=cv,
                    n_jobs = -1)
grid.fit(X,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


rf_grid = grid.best_estimator_
rf_grid.score(X,y)


# In[ ]:


#feature importance
feature_importances = pd.DataFrame(rf_grid.feature_importances_,
index = test[test.columns[1:]].columns,
columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False)


# In[ ]:


#Boosting
#GBM
from sklearn.ensemble import GradientBoostingClassifier
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r}
grid = GridSearchCV(GradientBoostingClassifier(),
                    param_grid=parameters,
                    cv=cv,
                    n_jobs = -1)
grid.fit(X,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


GBM_grid = grid.best_estimator_
GBM_grid.score(X,y)


# In[ ]:


from xgboost import XGBClassifier
n_estimators = [100,140,145,150,200,400];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
             }
grid = GridSearchCV(XGBClassifier(),
                    param_grid=parameters,
                    cv=cv,
                    n_jobs = -1)
grid.fit(X,y)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


XGB_grid = grid.best_estimator_
XGB_grid.score(X,y)


# In[ ]:




