#!/usr/bin/env python
# coding: utf-8

# The German Credit data set is a publically available data set downloaded from the UCI Machine Learning Repository. The data contains data on 20 variables and the classification whether an applicant is considered a Good or a Bad credit risk for 1000 loan applicants.
# 
# ### [Data Source](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
# - Professor Dr. Hans Hofmann  
# - Institut f"ur Statistik und "Okonometrie  
# - Universit"at Hamburg  
# - FB Wirtschaftswissenschaften  
# - Von-Melle-Park 5    
# - 2000 Hamburg 13
# 
# ### Benchmark
# ![Credit Risk Classification: Faster Machine Learning with Intel Optimized Packages](https://i.imgur.com/nL1l7WI.png)
# 
# according to [1] the best model is Random Forest with balanced feature selection data. it's has Accuracy 82%, Precision 84%, Recall 82% and F1-Score 81%. 
# 
# <br>
# 
# 
# The goal of this kernel is to beat The benchmark with  :
# - Convert dataset to Machine Learning friendly (Feature Engginering)
# - Develop XGBoost model to predict whether a loan is a good or bad risk.
# - Find the Best parameter for XGBoost Model (Hyperparameter Tunning)
# - Beat the Benchmark

# 

# # Table of Content
# 
# **1. [Introduction](#Introduction)** <br>
#     - Import Library
#     - Evaluation Function
#     - XGBoost Model
# **2. [Preprocess](#Preprocess)** <br>
#     - Importing Dataset
#     - StandardScaler
#     - Encoding Categorical Feature
#     - Concate Transformed Dataset
#     - Split Training Dataset
#     - XGBoost  1a: Unbalance Dataset (Base Model: ROC_AUC:0.74)
#     - XGBoost  1b: Unbalance Dataset (ROC_AUC:0.79)
# **3. [Balanced Dataset](#Balanced Dataset)** <br>    
#     - XGBoost 2a: Balanced (Base Model: ROC_AUC:0.77)
#     - **XGBoost 2b: Balanced (ROC_AUC:0.80)**
# **4. [Others](#Others)** <br>  
#     - Lighgbm (ROC_AUC:0.73)
#     - LogisticRegression (ROC_AUC:0.77)
#     - RandomForestClassifier (ROC_AUC:0.69)
#     - ExtraTreesClassifier (ROC_AUC:0.74)
#     - DecisionTreeClassifier (ROC_AUC:0.64)
#     - GradientBoostingClassifier (ROC_AUC:0.76)
#     - AdaBoostClassifier (ROC_AUC:0.72)

# <a id="Introduction"></a> <br>
# # **1. Introduction:** 
# - Import Library
# - Evaluation Function
# - XGBoost Model

# ### Import Library

# In[ ]:


#Importing necessary packages in Python 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 

import numpy as np ; np.random.seed(sum(map(ord, "aesthetics")))
import pandas as pd

from sklearn.datasets import make_classification 
from sklearn.learning_curve import learning_curve 
#from sklearn.cross_validation import train_test_split 
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler

import seaborn 
seaborn.set_context('notebook') 
seaborn.set_style(style='darkgrid')

from pprint import pprint 
 


# ### Evaluation Function
# 

# In[ ]:


# Function for evaluation reports
def get_eval1(clf, X,y):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X, y, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X, y, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X, y, cv=2, scoring='roc_auc')
    
    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))
    
    return 

def get_eval2(clf, X_train, y_train,X_test, y_test):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X_test, y_test, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X_test, y_test, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X_test, y_test, cv=2, scoring='roc_auc')
    
    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))
    
    return  
  
# Function to get roc curve
def get_roc (y_test,y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    #Plot of a ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="upper left")
    plt.show()
    return


# #### XGBoost Model

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import xgboost as xgb
from xgboost import XGBClassifier
#print('XGBoost v',xgb.__version__)

# fit, train and cross validate Decision Tree with training and test data 
def xgbclf(params, X_train, y_train,X_test, y_test):
  
    eval_set=[(X_train, y_train), (X_test, y_test)]
    
    model = XGBClassifier(**params).      fit(X_train, y_train, eval_set=eval_set,                   eval_metric='auc', early_stopping_rounds = 100, verbose=100)
        
    #print(model.best_ntree_limit)

    model.set_params(**{'n_estimators': model.best_ntree_limit})
    model.fit(X_train, y_train)
    #print(model,'\n')
    
    # Predict target variables y for test data
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit) #model.best_iteration
    #print(y_pred)
   
    # Get Cross Validation and Confusion matrix
    #get_eval(model, X_train, y_train)
    #get_eval2(model, X_train, y_train,X_test, y_test)
    
    # Create and print confusion matrix    
    abclf_cm = confusion_matrix(y_test,y_pred)
    print(abclf_cm)
    
    #y_pred = model.predict(X_test)
    print (classification_report(y_test,y_pred) )
    print ('\n')
    print ("Model Final Generalization Accuracy: %.6f" %accuracy_score(y_test,y_pred) )
    
    # Predict probabilities target variables y for test data
    y_pred_proba = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1] #model.best_iteration
    get_roc (y_test,y_pred_proba)
    return model

def plot_featureImportance(model, keys):
  importances = model.feature_importances_

  importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(keys)})
  importance_frame.sort_values(by = 'Importance', inplace = True)
  importance_frame.tail(10).plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')


# <a id="Preprocess"></a> <br>
# # **2. Preprocess** 
# - Importing Dataset
# - StandardScaler
# - Encoding Categorical Feature
# - Concate Transformed Dataset
# - Split Training Dataset
# - XGBoost  1a: Unbalance Dataset (Base Model: ROC_AUC:0.74)
# - XGBoost  1b: Unbalance Dataset (ROC_AUC:0.79)

# ### Import Dataset
# 
# OK let's get started. We'll download the data from the UCI website.

# In[ ]:


file = '../input/germancreditdata/german.data'
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

data = pd.read_csv(file,names = names, delimiter=' ')
print(data.shape)
print (data.columns)
data.head(10)


# In[ ]:


# Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
data.classification.replace([1,2], [1,0], inplace=True)
# Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
data.classification.value_counts()


# ### StandardScaler

# In[ ]:


#numerical variables labels
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'classification']

# Standardization
numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))


# ### Encoding Categorical Feature
# 
# Labelencoding to transform categorical to numerical, Enables better Visualization than one hot encoding

# In[ ]:


from collections import defaultdict

#categorical variables labels
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
for x in range(len(catvars)):
    print(catvars[x],": ", data[catvars[x]].unique())
    print(catvars[x],": ", lecatdata[catvars[x]].unique())

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])


# ### Concate Transformed Dataset
# append the dummy variable of the initial numerical variables numvars# append 

# In[ ]:


data_clean = pd.concat([data[numvars], dummyvars], axis = 1)

print(data_clean.shape)


# ### Split Training Dataset

# In[ ]:


# Unscaled, unnormalized data
X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean,y_clean,test_size=0.2, random_state=1)


# In[ ]:


X_train_clean.keys()


# ### XGBoost  1a: Unbalance Dataset (Base Model: ROC_AUC:0.74)

# In[ ]:


params={}
xgbclf(params, X_train_clean, y_train_clean, X_test_clean, y_test_clean)


# ### XGBoost  1b: Unbalance Dataset (ROC_AUC:0.79)

# In[ ]:


params={}

params1={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'gamma':0.1,
    'subsample':0.8,
    'colsample_bytree':0.3,
    'min_child_weight':3,
    'max_depth':3,
    #'seed':1024,
    'n_jobs' : -1
}

params2={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    #'gamma':0.01,
    'subsample':0.555,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'max_depth':8,
    #'seed':1024,
    'n_jobs' : -1
}

xgbclf(params2, X_train_clean, y_train_clean, X_test_clean, y_test_clean)


# <a id="Balanced Dataset"></a> <br>
# # **3. Balanced Dataset** 
# - XGBoost 2a: Balanced (Base Model: ROC_AUC:0.77)
# - XGBoost 2b: Balanced (ROC_AUC:0.80)

# In[ ]:



from imblearn.over_sampling import SMOTE

# Oversampling
# http://contrib.scikit-learn.org/imbalanced-learn/auto_examples/combine/plot_smote_enn.html#sphx-glr-auto-examples-combine-plot-smote-enn-py

# Apply SMOTE
sm = SMOTE(ratio='auto')
X_train_clean_res, y_train_clean_res = sm.fit_sample(X_train_clean, y_train_clean)

# Print number of 'good' credits and 'bad credits, should be fairly balanced now
print("Before/After clean")
unique, counts = np.unique(y_train_clean, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(y_train_clean_res, return_counts=True)
print(dict(zip(unique, counts)))


# In[ ]:


#Great, before we do anything else, let's split the data into train/test.
X_train_clean_res = pd.DataFrame(X_train_clean_res, columns=X_train_clean.keys())
#y_train_clean_res = pd.DataFrame(y_train_clean_res)


# In[ ]:


print(np.shape(X_train_clean_res))
print(np.shape(y_train_clean_res))
print(np.shape(X_test_clean)) 
print(np.shape(y_test_clean))


# ### XGBoost 2a: Balanced (Base Model: ROC_AUC:0.77)

# In[ ]:


#BASE MODEL
params={}
xgbclf(params,X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### XGBoost 2b: Balanced (ROC_AUC:0.80)

# In[ ]:


params = {}

params1={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'gamma':0.1,
    'subsample':0.8,
    'colsample_bytree':0.3,
    'min_child_weight':3,
    'max_depth':3,
    #'seed':1024,
    'n_jobs' : -1
}

params2={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    #'gamma':0.01,
    'subsample':0.555,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'max_depth':8,
    #'seed':1024,
    'n_jobs' : -1
}

#xgbclf(params, X_train, y_train,X_test,y_test)
model = xgbclf(params2,X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)
model
#plot_featureImportance(model, X_train_clean_res.keys())


# # 4.  Feature Selection
# - XGBoost3 (Base Model:ROC_AUC:0.73)
# - GridSearchCV (ROC_AUC:0.70)

# In[ ]:


#model = xgbclf(params1,X_train_clean_res[importance_col], y_train_clean_res,X_test_clean[importance_col], y_test_clean)

importances = model.feature_importances_
importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(X_train_clean_res.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True, ascending=False)
importance_col = importance_frame.Feature.head(10).values


# ### XGBoost3 (Base Model:ROC_AUC:0.73)

# In[ ]:


params = {}

params1={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.01,
    #'gamma':0.1,
    #'subsample':0.8,
    #'colsample_bytree':0.3,
    #'min_child_weight':3,
    'max_depth':3,
    #'seed':1024,
    'n_jobs' : -1
}

xgbclf(params,X_train_clean_res[importance_col], y_train_clean_res,X_test_clean[importance_col], y_test_clean)


# ### GridSearchCV (ROC_AUC:0.70)

# In[ ]:


from sklearn.grid_search import GridSearchCV

print('XGBoost with grid search')
# play with these params
params={
    'learning_rate': [0.01, 0.02],
    'max_depth': [3], # 5 is good but takes too long in kaggle env
    #'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    #'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],
    'n_estimators': [50, 100, 200, 300, 400, 500]
    #'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]
}


xgb_clf = xgb.XGBClassifier()

rs = GridSearchCV(xgb_clf,
                  params,
                  cv=2,
                  scoring="roc_auc",
                  n_jobs=1,
                  verbose=False)
rs.fit(X_train_clean_res[importance_col], y_train_clean_res)
best_est = rs.best_estimator_
print(best_est)
print(rs.best_score_)

# Roc AUC with test data
print(rs.score(X_test_clean[importance_col],y_test_clean))

# Roc AUC with all train data
#y_pred_proba = best_est.predict_proba(X_test_clean[importance_col])[:,1]
#print("Roc AUC: ", roc_auc_score(y_test_clean, y_pred_proba))

#xgbclf(params1,X_train_clean_res[importance_col], y_train_clean_res,X_test_clean[importance_col], y_test_clean)


# <a id="Others"></a> <br>
# # 5. Others
# - Lighgbm (ROC_AUC:0.73)
# - LogisticRegression (ROC_AUC:0.77)
# - RandomForestClassifier (ROC_AUC:0.69)
# - ExtraTreesClassifier (ROC_AUC:0.74)
# - DecisionTreeClassifier (ROC_AUC:0.64)
# - GradientBoostingClassifier (ROC_AUC:0.76)
# - AdaBoostClassifier (ROC_AUC:0.72)

# ### Lighgbm (ROC_AUC:0.73)

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import lightgbm as lgb

# fit, train and cross validate Decision Tree with training and test data 
def lgbclf(X_train, y_train,X_test, y_test):

    model = lgb.LGBMClassifier().fit(X_train, y_train)
    print(model,'\n')

    # Predict target variables y for test data
    y_pred = model.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(model, X_train, y_train,y_test,y_pred)
    #get_eval2(model, X_train, y_train,X_test, y_test,y_pred)
    get_roc (y_test,y_pred)
    return

# Logistic Regression
#lgbclf(X_train, y_train,X_test,y_test)
lgbclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### LogisticRegression (ROC_AUC:0.77)

# In[ ]:


from sklearn.linear_model import LogisticRegression

# fit, train and cross validate Decision Tree with training and test data 
def logregclf(X_train, y_train,X_test, y_test):
    print("LogisticRegression")
    model = LogisticRegression().fit(X_train, y_train)
    print(model,'\n')

    # Predict target variables y for test data
    y_pred = model.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(model, X_train, y_train,y_test,y_pred)
    #get_eval2(model, X_train, y_train,X_test, y_test,y_pred)
    get_roc (y_test,y_pred)
    return

# Logistic Regression
#logregclf(X_train, y_train,X_test,y_test)
logregclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### RandomForestClassifier (ROC_AUC:0.69)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 

# fit, train and cross validate Decision Tree with training and test data 
def randomforestclf(X_train, y_train,X_test, y_test):
    print("RandomForestClassifier")
    randomforest = RandomForestClassifier().fit(X_train, y_train)
    print(randomforest,'\n')
    
    # Predict target variables y for test data
    y_pred = randomforest.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(randomforest, X_train, y_train,y_test,y_pred)
    get_roc (y_test,y_pred)
    return randomforest

# Random Forest
# Choose clean data, as tree is robust
rf = randomforestclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### ExtraTreesClassifier (ROC_AUC:0.74)

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

# fit, train and cross validate Decision Tree with training and test data 
def extratreesclf(X_train, y_train,X_test, y_test):
    print("ExtraTreesClassifier")
    extratrees = ExtraTreesClassifier().fit(X_train, y_train)
    print(extratrees,'\n')
    
    # Predict target variables y for test data
    y_pred = extratrees.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(extratrees, X_train, y_train,y_test,y_pred)
    
    get_roc (y_test,y_pred)
    return
 
# Extra Trees
# Choose clean data, as tree is robust
extratreesclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### DecisionTreeClassifier (ROC_AUC:0.64)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 
# fit, train and cross validate Decision Tree with training and test data 
def dectreeclf(X_train, y_train,X_test, y_test):
    print("DecisionTreeClassifier")
    dec_tree = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=5).fit(X_train, y_train)
    print(dec_tree,'\n')
    
    # Predict target variables y for test data
    y_pred = dec_tree.predict_proba(X_test)[:,1]

    
    # Get Cross Validation and Confusion matrix
    #get_eval(dec_tree, X_train, y_train,y_test,y_pred)
    get_roc (y_test,y_pred)
    return

# Decisiontree
dectreeclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### GradientBoostingClassifier (ROC_AUC:0.76)

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

# fit, train and cross validate GradientBoostingClassifier with training and test data 
def gradientboostingclf(X_train, y_train, X_test, y_test):  
    print("GradientBoostingClassifier")
    gbclf = GradientBoostingClassifier().fit(X_train, y_train)
    print(gbclf,'\n')
    
    # Predict target variables y for test data
    y_pred = gbclf.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(gbclf, X_train, y_train,y_test,y_pred)
    get_roc (y_test,y_pred)
    return
  
# GradientBoostingClassifier
# Choose clean data, as tree is robust
gradientboostingclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# ### AdaBoostClassifier (ROC_AUC:0.75)

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

# fit, train and cross validate GradientBoostingClassifier with training and test data 
def adaboostclf(X_train, y_train, X_test, y_test):  
    print("AdaBoostClassifier")
    abclf = AdaBoostClassifier().fit(X_train, y_train)
    print(abclf,'\n')
    
    # Predict target variables y for test data
    y_pred = abclf.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(abclf, X_train, y_train,y_test,y_pred)
    get_roc (y_test,y_pred)
    return

# AdaBoostClassifier
# Choose clean data, as tree is robust
adaboostclf(X_train_clean_res, y_train_clean_res,X_test_clean, y_test_clean)


# In[ ]:


import eli5


# In[ ]:


import xgboost as xgb
xgc = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                        objective='binary:logistic', random_state=42)
xgc.fit(X_train_clean_res, y_train_clean_res)


# In[ ]:


y_preds = xgc.predict(X_test_clean)
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
metrics.accuracy_score(y_test_clean,y_preds)
print("confusion matrix=",metrics.confusion_matrix(y_test_clean,y_preds))
print("classification report=\n",classification_report(y_test_clean,y_preds))
print("accuracy=",metrics.accuracy_score(y_test_clean,y_preds))
print("mean squared error=",mean_squared_error(y_test_clean, y_preds))
print("roc_auc score is=",roc_auc_score(y_test_clean, y_preds))


# In[ ]:


fig = plt.figure(figsize = (16, 12))
title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

ax1 = fig.add_subplot(2,2, 1)
xgb.plot_importance(xgc, importance_type='weight', ax=ax1, max_num_features=20)
t=ax1.set_title("Feature Importance - Feature Weight")

ax2 = fig.add_subplot(2,2, 2)
xgb.plot_importance(xgc, importance_type='gain', ax=ax2, max_num_features=20)
t=ax2.set_title("Feature Importance - Split Mean Gain")

ax3 = fig.add_subplot(2,2, 3)
xgb.plot_importance(xgc, importance_type='cover', ax=ax3, max_num_features=20)
t=ax3.set_title("Feature Importance - Sample Coverage")


# In[ ]:


import eli5

eli5.show_weights(xgc.get_booster())


# In[ ]:


y_test_clean.iloc[3]


# In[ ]:




doc_num = 2
print('Actual Label:', y_test_clean.iloc[doc_num])
print('Predicted Label:', y_preds[doc_num])
eli5.show_prediction(xgc.get_booster(), X_test_clean.iloc[doc_num], 
                     feature_names=list(X_test_clean.columns),
                     show_feature_values=True)


# In[ ]:


y_test_clean.head()


# In[ ]:



doc_num = 3
print('Actual Label:', y_test_clean.iloc[doc_num])
print('Predicted Label:', y_preds[doc_num])
eli5.show_prediction(xgc.get_booster(), X_test_clean.iloc[doc_num], 
                     feature_names=list(X_test_clean.columns),
                     show_feature_values=True)


# In[ ]:




from pdpbox import pdp, get_dataset, info_plots

def plot_pdp(model, df, feature, cluster_flag=False, nb_clusters=None, lines_flag=False):
    
    # Create the data that we will plot
    pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns.tolist(), feature=feature)

    # plot it
    pdp.pdp_plot(pdp_goals, feature, cluster=cluster_flag, n_cluster_centers=nb_clusters, plot_lines=lines_flag)
    plt.show()


# In[ ]:




# plot the PD univariate plot
plot_pdp(xgc, X_train_clean_res, 'creditamount')


# In[ ]:


plot_pdp(xgc, X_train_clean_res, 'age')


# In[ ]:


plot_pdp(xgc, X_train_clean_res, 'duration')


# In[ ]:


import shap

# load JS visualization code to notebook
shap.initjs()


# In[ ]:


explainer = shap.TreeExplainer(xgc)
shap_values = explainer.shap_values(X_test_clean)


# In[ ]:


X_shap = pd.DataFrame(shap_values)
X_shap.tail()


# In[ ]:


print('Expected Value: ', explainer.expected_value)


# In[ ]:


shap.summary_plot(shap_values, X_test_clean, plot_type="bar", color='blue')


# In[ ]:


shap.summary_plot(shap_values, X_test_clean)


# In[ ]:


import lime
import lime.lime_tabular


# In[ ]:


categorical_features = np.argwhere(np.array([len(set(X_train_clean_res.values[:,x]))
for x in range(X_train_clean_res.values.shape[1])]) <= 10).flatten()


# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train_clean_res.values, 
                                                   feature_names=X_train_clean_res.columns.values.tolist(), 
                                                   categorical_features=categorical_features, 
                                                   verbose=True, mode='regression')


# In[ ]:



i = 2
 
exp = explainer.explain_instance(X_test_clean.iloc[i], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)


# In[ ]:


i = 3
 
exp = explainer.explain_instance(X_test_clean.iloc[i], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)

