#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_data=pd.read_csv(r"../input/bank-train.csv")
test_data=pd.read_csv(r"../input/bank-test.csv")


# ### EDA

# In[ ]:


train_data.head(1)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


#Check individual attributes 
#id
#Check if it's a unique id for every single row 
train_data["id"].value_counts(ascending=False)[0]
#Unique for each row, not useful for classification, might remove it later 


# In[ ]:


#Age
#Age is a numeric data. 
train_data["age"].isnull().sum()/train_data.shape[0]
#No null data 


# In[ ]:


#Check the distribution of age 
train_data["age"].hist()
#right skewed --- whether should we transform it ? 


# In[ ]:


#Let us make one variable using the log transformation for age 
#Later we can compare the result of both features and see which one helps to improve classfication accuracy 
train_data["log_age"]=np.log(train_data["age"])
train_data["log_age"].hist()
#Normal distribution 


# In[ ]:


train_data["job"].value_counts()
#Many cateogiries -- > maybe merge some of the categories ? 
#Let us check if these categories have some relationships with the response variable.


# In[ ]:


train_data.groupby("job").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Job Categories")
#There are some differences between the job categories 
#If we want to limit the levels, how can we merge them? 


# In[ ]:


train_data["marital"].value_counts(normalize=True)


# In[ ]:


train_data[train_data["marital"]=="unknown"].shape


# In[ ]:


train_data.groupby("marital").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Marital Status")
#There are some differences between the marital status 
#Unkown marital status have obviously higher subscription rate, 
#We may not change it. 


# In[ ]:


train_data["education"].value_counts(normalize=True)


# In[ ]:


train_data.groupby("education").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Education")
#kind of wanna merge basic 6-yr and 9yr, and 4y and high school and professional course 
#How to justify the decision? 


# In[ ]:


train_data.groupby("default").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Credit Default Status")
#This might be a very important feature 


# In[ ]:


train_data.groupby("housing").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Housing Loan Status")


# In[ ]:


train_data.groupby("loan").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Personal Loan Status")


# In[ ]:


train_data.groupby("contact").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Contact Method")


# In[ ]:


train_data["month"].value_counts(normalize=True)


# In[ ]:


train_data.groupby("month").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Last Contact Month")
#Group some months? How to justify it? 


# In[ ]:


train_data["day_of_week"].value_counts(normalize=True)


# In[ ]:


train_data.groupby("day_of_week").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by Day of Week")
#Group some months? How to justify it? 


# In[ ]:


train_data.drop("duration",axis=1,inplace=True)


# In[ ]:


train_data["campaign"].hist()
#Highly right skewed ?? 


# In[ ]:


train_data["test_campaign"]=train_data['campaign'].apply(lambda x: np.floor(x/3)+1)


# In[ ]:


train_data.groupby("test_campaign").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
plt.title("Subscription Rate by # of Contacts (Bins=3) ")
#Group some months? How to justify it? 


# In[ ]:


train_data["test_campaign"].hist()


# In[ ]:


#drop the over 10s 
train_data=train_data[train_data["test_campaign"]<=10.0]
#DELETE OUTLIERS AT HERE 


# In[ ]:


#Transform the test campaign varaible 
train_data["test_campaign"].hist()
#How to transform it more normal 


# In[ ]:


train_data[train_data["pdays"]!=999]["pdays"].hist()
#Do we need to do something for 999? 


# In[ ]:


train_data[train_data["pdays"]!=999]["pdays"].max()


# In[ ]:


#Transform pdays to a category: 
bins=[0,7,14,21,28,1000]
train_data["pdays_category"]=pd.cut(train_data["pdays"],bins, labels=["OneWeek", "TwoWeek", "ThreeWeek","FourWeek","NoPreviousCampaign"],right=False)


# In[ ]:


train_data["pdays_category"].value_counts(normalize=True)


# In[ ]:


train_data.groupby("pdays_category").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")


# In[ ]:


train_data["previous"].hist()


# In[ ]:


train_data.groupby("previous").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
#The number of campaigns matters 
plt.title("Success Rate by # of Previous Campaigns")


# In[ ]:


train_data=train_data[train_data["previous"]!=7]
###DELETE OUTLIERS


# In[ ]:


train_data.groupby("poutcome").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
#The number of campaigns matters 
plt.title("Success Rate by # of Pevious Campaign Outcome")


# In[ ]:


train_data.groupby("emp.var.rate").y.value_counts(normalize=True).unstack()[1].plot(kind="bar")
#The number of campaigns matters 
plt.title("Success Rate by # of Economic Variation" )


# In[ ]:


sns.heatmap(train_data[["emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m"]].corr())
#These variables are highly correlated. 
#Think about how to manage them if doing logistic regression. 


# In[ ]:


train_data.head(1)


# ### Baseline

# In[ ]:


y_train=train_data["y"]
X_train=train_data.drop("y",axis=1)
cutoff=X_train.shape[0]
X_train.drop(["id","log_age","test_campaign","pdays_category"],axis=1,inplace=True)


# In[ ]:


test_data.drop("duration",axis=1,inplace=True)
test_data_new=test_data.drop("id",axis=1)
X=X_train.append(test_data_new)


# In[ ]:


X_dummies=pd.get_dummies(X,drop_first=True)


# In[ ]:


X=X_dummies.iloc[:cutoff]
X_valid=X_dummies.iloc[cutoff:]
y=y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


base_clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred=base_clf.predict(X_test)
y_score = base_clf.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
C=confusion_matrix(y_test,y_pred)
sns.heatmap(C / C.astype(np.float).sum(axis=1))
plt.title("Confusion Matrix Normalized")


# In[ ]:


def print_classfiction_metrics(testy,yhat_classes):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)
    
    


# In[ ]:


print_classfiction_metrics(y_test,y_pred)
print("Area Under ROC Curve:", roc_auc)


# In[ ]:


result=base_clf.predict(X_valid)


# In[ ]:


#Submission Score is 0.888300 


# ### Feature Transformation 

# In[ ]:


y_train=train_data["y"]
X_train=train_data.drop("y",axis=1)
cutoff=X_train.shape[0]
X_train.drop(["id","log_age","test_campaign","pdays_category"],axis=1,inplace=True)


# In[ ]:


try:
    test_data_new=test_data.drop(["id","duration"],axis=1)
except: 
    test_data_new=test_data.drop(["id"],axis=1)
X=X_train.append(test_data_new,sort=False)
X_transform=X.copy()


# In[ ]:


X_transform["age"]=np.log(X_transform["age"])
X_transform["campaign"]=X_transform['campaign'].apply(lambda x: np.floor(x/3)+1)

#Transform pdays to a category: 
bins=[0,7,14,21,28,1000]
X_transform["pdays"]=pd.cut(X_transform["pdays"],bins, labels=["OneWeek", "TwoWeek", "ThreeWeek","FourWeek","NoPreviousCampaign"],right=False)


# In[ ]:


X_dummies=pd.get_dummies(X_transform,drop_first=True)


# In[ ]:


X_dummies.head(1)


# In[ ]:


X=X_dummies.iloc[:cutoff]
X_valid=X_dummies.iloc[cutoff:]
y=y_train


# In[ ]:


numerics=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
X[numerics] = preprocessing.scale(X[numerics])
X_valid[numerics] = preprocessing.scale(X_valid[numerics])


# In[ ]:


X.head(1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


base_clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred=base_clf.predict(X_test)
y_score = base_clf.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
C=confusion_matrix(y_test,y_pred)
sns.heatmap(C / C.astype(np.float).sum(axis=1))
plt.title("Confusion Matrix Normalized")


# In[ ]:


print_classfiction_metrics(y_test,y_pred)
print("Area Under ROC Curve:", roc_auc)


# Submission reulst on this model is 0.88911

# In[ ]:


## Changing Thresholds of the logistic model 
####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(_, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])



# In[ ]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# In[ ]:


p, r, thresholds = precision_recall_curve(y_test, y_score)
plot_precision_recall_vs_threshold(p,r,thresholds)


# In[ ]:


import random
def find_best_accuracy(clf,X,y):
    result={}
    rands=[]
    for x in range(5):
        rands.append(random.randint(1,100001))
    rands.append(42)
    
    for i in rands: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        y_score=clf.decision_function(X_test)
        thresholds= np.linspace(y_score.min(), y_score.max(),1000)
        accr={}
        for thre in thresholds:
            new_y_pred=[0 if i<thre else 1 for i in y_score]
            accuracy = accuracy_score(y_test, new_y_pred)
            accr[thre]=accuracy*100    
        result[i]=sorted(accr, key=(lambda key:accr[key]), reverse=True)[:3]
    return result


# In[ ]:


find_best_accuracy(clf=LogisticRegression(random_state=0),X=X,y=y)


# In[ ]:


new_y_pred=[0 if i<0.03582943178391851 else 1 for i in y_score]


# In[ ]:


print_classfiction_metrics(y_test,new_y_pred)


# ### Tune the Logistic Model 

# In[ ]:


params = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10],'solver':['liblinear','saga']}
# Create grid search using 5-fold cross validation
best_lg = GridSearchCV(LogisticRegression(max_iter=100), params, cv=5, verbose=0,scoring='accuracy',return_train_score=True)
best_lg.fit(X_train,y_train)


# In[ ]:


best_lg.best_params_


# In[ ]:


y_pred=best_lg.predict(X_test)


# In[ ]:


y_score = best_lg.decision_function(X_test)
y_pred=best_lg.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print_classfiction_metrics(y_test,y_pred)
print("Area Under ROC Curve",roc_auc)


# Tuning submission result is 0.88830.

# ### Try Other Models Using Original Dataset

# In[ ]:


train_data=pd.read_csv("bank-train.csv")
test_data=pd.read_csv("bank-test.csv")
y_train=train_data["y"]
X_train=train_data.drop("y",axis=1)
cutoff=X_train.shape[0]
X=X_train.append(test_data)
X.drop(["duration","id"],axis=1,inplace=True)
X_transform=X.copy()
X_transform=pd.get_dummies(X_transform,drop_first=True)
X=X_transform.iloc[:cutoff]
y=y_train
X_valid=X_transform.iloc[cutoff:]

#Scaling
numerics=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
X[numerics] = preprocessing.scale(X[numerics])
X_valid[numerics] = preprocessing.scale(X_valid[numerics])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


    # iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train.ravel())
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
    y_pred=clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score)
    roc_auc = auc(fpr, tpr)
    print_classfiction_metrics(y_test.ravel(),y_pred)
    plt.plot(fpr, tpr,lw=2, label='ROC curve (area = {}) for {}'.format(roc_auc,name) )
  
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.05))
plt.show()


# ### Try Other Models on Transformed Datasets 

# In[ ]:


train_data=pd.read_csv(r"../input/bank-train.csv")
test_data=pd.read_csv(r"../input/bank-test.csv")
#Delete Outliers 
train_data=train_data[train_data["default"]!="yes"]
train_data=train_data[train_data["previous"]<=5]
train_data["campaign"]=train_data['campaign'].apply(lambda x: np.floor(x/3)+1)
test_data["campaign"]=test_data['campaign'].apply(lambda x: np.floor(x/3)+1)

train_data=train_data[train_data["campaign"]<=10]
y_train=train_data["y"]
X_train=train_data.drop("y",axis=1)
cutoff=X_train.shape[0]
X=X_train.append(test_data)
X.drop(["duration","id"],axis=1,inplace=True)
X_transform=X.copy()

X_transform["age"]=np.log(X_transform["age"])
#Transform pdays to a category: 
bins=[0,7,14,21,28,1500]
X_transform["pdays"]=pd.cut(X_transform["pdays"],bins, labels=["OneWeek", "TwoWeek", "ThreeWeek","FourWeek","NoPreviousCampaign"],right=False)
X_transform=pd.get_dummies(X_transform,drop_first=True)
X=X_transform.iloc[:cutoff]
y=y_train
X_valid=X_transform.iloc[cutoff:]

#Scaling

numerics=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
X[numerics] = preprocessing.scale(X[numerics])
X_valid[numerics] = preprocessing.scale(X_valid[numerics])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


    # iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train.ravel())
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]
    y_pred=clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score)
    roc_auc = auc(fpr, tpr)
    print_classfiction_metrics(y_test.ravel(),y_pred)
    plt.plot(fpr, tpr,lw=2, label='ROC curve (area = {}) for {}'.format(roc_auc,name) )
  
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.05))
plt.show()


# ### AdaBoost On Entire DataSet

# In[ ]:


train_data=pd.read_csv(r"../input/bank-train.csv")
test_data=pd.read_csv(r"../input/bank-test.csv")
y_train=train_data["y"]
X_train=train_data.drop("y",axis=1)
cutoff=X_train.shape[0]
X=X_train.append(test_data)
X.drop(["duration","id"],axis=1,inplace=True)


# In[ ]:


#Transform pdays to a category: 
bins=[0,7,14,21,28,1500]
X["pdays"]=pd.cut(X["pdays"],bins, labels=["OneWeek", "TwoWeek", "ThreeWeek","FourWeek","NoPreviousCampaign"],right=False)


# In[ ]:


X.head(1)


# In[ ]:


nums=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
cats=[i for i in X.columns.tolist() if i not in nums]


# In[ ]:


for i in cats:
    X[i]=X[i].astype('category')


# In[ ]:


X.head(1)


# In[ ]:


# Desired label orders for categorical columns.

educ_order = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri']


def ordered_labels(df, col, order):
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.reorder_categories(order, ordered=True)
    df[col] = df[col].cat.codes.astype(int)
    return df

X=ordered_labels(X,"education",educ_order)
X=ordered_labels(X,"day_of_week",day_order)


# In[ ]:


X=pd.get_dummies(X,drop_first=True)


# In[ ]:


X.head(1)


# In[ ]:


X_transform=X.copy()
X=X_transform.iloc[:cutoff]
y=y_train
X_valid=X_transform.iloc[cutoff:]


# In[ ]:


#Scaling
numerics=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
X[numerics] = preprocessing.scale(X[numerics])
X_valid[numerics] = preprocessing.scale(X_valid[numerics])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf=AdaBoostClassifier()


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_score = clf.decision_function(X_test)
y_pred= clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print_classfiction_metrics(y_test,y_pred)
print("Area Under ROC Curve",roc_auc)


# #### Changing Threshold  of the Ada Boost Model

# In[ ]:


####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(_, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])


# In[ ]:


p, r, thresholds = precision_recall_curve(y_test, y_score)
plot_precision_recall_vs_threshold(p,r,thresholds)


# In[ ]:


find_best_accuracy(clf,X,y)


# In[ ]:


result_scores=clf.decision_function(X_valid)


# In[ ]:


result=[0 if i<-0.0013974661683098244 else 1 for i in result_scores]


# In[ ]:


submission = pd.concat([test_data["id"], pd.Series(result)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)

