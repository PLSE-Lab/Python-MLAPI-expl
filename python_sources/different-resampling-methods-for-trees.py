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


# # Table of Content:
# 
# [1. Introduction](#1)
# 
# [2. Data Overview](#2)
# 
# [3. Basic Model Fitting](#3)
# * [3.1 Model evaluation](#3.1) 
# 
# [4. Models With Resampling](#4)
# * [4.1 Downsample](#4.1)
# * [4.2 Upsample](#4.2)
# * [4.3 Class weight for Random Forest (Manually)](#4.3)
#     * [4.31 Balanced Class for RF and DecisionTree](#4.31)    
# * [4.4 Smote](#4.4)
# 
# [5. Precision vs. Recall curve](#5)
# 
# [6. Final Test](#6)

# <a id="1"></a>
# # 1. Introduction

# ### Goal: the pros and cons of different techniques for handling inbalanced dataset
# 
# ### Variables: 
# * Churn: 1 if customer cancelled service, 0 if not
# * AccountWeeks: number of weeks customer has had active account
# * ContractRenewal: 1 if customer recently renewed contract, 0 if not
# * DataPlan: 1 if customer has data plan, 0 if not
# * DataUsage: gigabytes of monthly data usage
# * CustServCalls: number of calls into customer service
# * DayMins: average daytime minutes per month
# * DayCalls: average number of daytime calls
# * MonthlyCharge: average monthly bill
# * OverageFee: largest overage fee in last 12 months
# * RoamMins: average number of roaming minutes

# <a id="2"></a>
# # 2. Data Overview

# In[ ]:


#Load the data
df = pd.read_csv("/kaggle/input/telecom-churn/telecom_churn.csv")
df.sample(5)


# In[ ]:


#check missing values
df.info()


# - No missing Data - Great!
# - No Categorical variables - Great!
# 
# Ambigious varaible: Customer Service Call 
# 1. I wonder if the customer service call is the average numbers of call/month, or the most recent month's call or the total amount of calls that customer have made. 
# 2. We don't have any ideas on when the calls were made. 5 recent customers service calls can be an indicator to predict the churn, while 5 customer service calls that were made a year ago will not be an indicator
# 

# ### Baseline Churn

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math


# In[ ]:


#default churn rate
labels = 'Churn', "Stay"
sizes = [df.Churn[df['Churn'] == 1].count(), df.Churn[df['Churn'] == 0].count()]
explode = (0.1, 0)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')

plt.title("Proportion of customer churned and retained")

plt.show()


# <a id="3"></a>
# # 3. Model fitting with original data

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_curve


# In[ ]:


#dat preperation
label = df["Churn"]
df_train1 = df.iloc[:, 1:].copy()
feature_names = list(df_train1.columns.values)

#I seperate the data into train, valiation and test. We will reserve the test set till the end to test the performance of the best model.

#set, testset
X_trainval, X_test, y_trainval, y_test = train_test_split(df_train1, label, test_size = 0.2, random_state=1)
#train, validation set split
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = 0.2, random_state=1)


# ### Baseline Model

# In[ ]:


#dummy model
dummy1 = DummyClassifier(random_state=1).fit(X_train, y_train)
pred_dummy1 = dummy1.predict(X_val)
print("randomly guessing score: {:.2f}".format(dummy1.score(X_val,y_val)))
dummy2 = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
pred_dummy2 = dummy2.predict(X_val)
print("guess all customers will stay score: {:.2f}".format(dummy2.score(X_val,y_val)))


# ### Different Tree models

# In[ ]:


#tree model
tree = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
print("Decision Tree validation score: {:.2f}".format(tree.score(X_val, y_val)))
tree_crossval = cross_val_score(tree, X_trainval, y_trainval)
print("Decision Tree cross-validation score: {:.2f}".format(tree_crossval.mean()))

#tree model with simple tune
tree_tune = DecisionTreeClassifier(random_state=1, max_depth = 6).fit(X_train, y_train)
#I picked 6 by just trying out different numbers. We can also use GridSearchCV to select the best parameters but I am a bit lazy to do so. 
print("Decision Tree validation score after tune: {:.2f}".format(tree_tune.score(X_val, y_val)))
pred_tree = tree.predict(X_val)
tree_crossval_tune = cross_val_score(tree_tune, X_trainval, y_trainval)
print("Decision Tree cross-validation score (tune): {:.2f}".format(tree_crossval_tune.mean()))


# In[ ]:


print("With original dataset")
#tree model with simple tune
tree_tune = DecisionTreeClassifier(random_state=1, max_depth = 6).fit(X_train, y_train)
#I picked 6 by just trying out different numbers. We can also use GridSearchCV to select the best parameters but I am a bit lazy to do so. 
print("Decision Tree (max_depth 6) validation score: {:.2f}".format(tree_tune.score(X_val, y_val)))
pred_tree = tree.predict(X_val)
tree_crossval_tune = cross_val_score(tree_tune, X_trainval, y_trainval)
print("Decision Tree (max_depth 6) cross-validation score max_depth 6: {:.2f}".format(tree_crossval_tune.mean()))

#random forest model
forest = RandomForestClassifier(random_state=1, max_depth=8).fit(X_train, y_train)
pred_forest = forest.predict(X_val)
print("\nRandom Forest (max_depth 8) validation score: {:.2f}".format(forest.score(X_val, y_val)))
forest_crossval = cross_val_score(forest, X_trainval, y_trainval)
print("Random Forest (max_depth 8) cross-validation score: {:.2f}".format(forest_crossval.mean()))

#Gradient Boosting model
gradient = GradientBoostingClassifier(random_state=1).fit(X_train, y_train)
pred_gradient = gradient.predict(X_val)
#print("Gradient Boosting train score: {:.2f}".format(gradient.score(X_train, y_train)))
print("\nGradient Boosting validation score: {:.2f}".format(gradient.score(X_val, y_val)))
gradient_crossval = cross_val_score(gradient, X_trainval, y_trainval)
print("Grandient Boosting cross-validation score: {:.2f}".format(gradient_crossval.mean()))

#XG bossting model
xg = XGBClassifier().fit(X_train, y_train)
pred_xg = xg.predict(X_val)
#print("XGBoost train score: {:.2f}".format(xg.score(X_train, y_train)))
print("\nXGBoost validation score: {:.2f}".format(xg.score(X_val, y_val)))
xg_crossval = cross_val_score(xg, X_trainval, y_trainval)
print("XGBoost cross-validation score: {:.2f}".format(xg_crossval.mean()))


# <a id="3.1"></a>
# # 3.1 Basic model Evaluation

# In[ ]:


print("With original dataset")
#confusion matrix for dummy
confusion_dummy = confusion_matrix(y_val, pred_dummy2)
print("confusion matrix for dummy model:\n{}".format(confusion_dummy))

#confusion matrix for trees 
confusion_tree=confusion_matrix(y_val, pred_tree)
print("confusion matrix for decision tree (max_depth 6):\n{}".format(confusion_tree))

confusion_forest=confusion_matrix(y_val, pred_forest)
print("confusion matrix for random forest (max_depth 8):\n{}".format(confusion_forest))

confusion_gradient=confusion_matrix(y_val, pred_gradient)
print("confusion matrix for gradient boost:\n{}".format(confusion_gradient))

confusion_xg=confusion_matrix(y_val, pred_xg)
print("confusion matrix for xg boosting:\n{}".format(confusion_xg))


# Based on the confusion matrix, the our RF model makes more False Negative mistakes, which means customers will churn but the model says it will stay. It is because we have so many "Stay" in the sample that the model is focusing on getting the "Stay" class right, not the "Churn" class. We will need to rebalance the dataset. 

# In[ ]:


#classification_report
#dummy
print("dummy classificatoin report\n")
print(classification_report(y_val, pred_dummy2, target_names = ["Stay", "Churn"]))
print("\nrandom forest classification report\n")
print(classification_report(y_val, pred_forest, target_names = ["Stay", "Churn"]))


# From the classification report aboved, the difference between the model's ability to predict "Stay" vs. to predict "Churn is even more obvious. Here are some reminders what "precision" and "recall" are:
# * Precision: True Positive / (True Positive + False Positive). Let's translate to Englis and let's use the example of "Churn", which is 0.88. It means: The times the model rightly predicted customers will churn / (rightly predict churn + predict churn but the client stay) is 88% 
# * Recall: True Positive / (True Positive + False Negative). Let's translate to Englis and let's use the example of "Churn", which is 0.63. It means: The times the model rightly predicted customers will churn / (rightly predict churn + predict stay but the client churn) is 63% 

# <a id="4"></a>
# # 4. Models With Resamplings

# <a id="4.1"></a>
# ### 4.1 Downsample

# In[ ]:


from sklearn.utils import resample


# In[ ]:


# Separate majority and minority classes
df_maj = df[df.Churn==0]
df_min = df[df.Churn==1]

print("The minority sample size is: {}".format(len(df_min))) #483

# Downsample majority class
df_maj_ds = resample(df_maj, replace=False,    # sample without replacement
                             n_samples=483,     # to match minority class
                             random_state=1) # reproducible results
# Combine minority class with downsampled majority class
df_ds = pd.concat([df_maj_ds, df_min])
 
# Display new class counts
df_ds.Churn.value_counts()


# In[ ]:


#dat preperation
label2 = df_ds["Churn"]
df_ds_train = df_ds.iloc[:, 1:].copy()
feature_names2 = list(df_ds_train.columns.values)
#set, testset split
Xds_train, Xds_val, yds_train, yds_val = train_test_split(df_ds_train, label2, test_size = 0.2, random_state=1)


# In[ ]:


print("Downsampling")
#tree after downsample
tree_ds = DecisionTreeClassifier(random_state=1, max_depth = 4).fit(Xds_train, yds_train)
print("Tree (max_depth 4) validation score: {:.2f}".format(tree_ds.score(X_val, y_val)))
tree_ds_crossval = cross_val_score(tree_ds, X_val, y_val)
print("Tree (max_depth 4) cross-validation score after ds: {:.2f}".format(tree_ds_crossval.mean()))

#random forest model after downsampled
forest_ds = RandomForestClassifier(random_state=1).fit(Xds_train, yds_train)
print("\nRandom Forest validation score: {:.2f}".format(forest_ds.score(X_val, y_val)))
forest_ds_crossval = cross_val_score(forest_ds, X_val, y_val)
print("Random Forest cross-validation score after ds: {:.2f}".format(forest_ds_crossval.mean()))

gradient_ds = GradientBoostingClassifier(random_state=1).fit(Xds_train, yds_train)
print("\nGradient Boosting validation score: {:.2f}".format(gradient_ds.score(X_val, y_val)))
gradient_ds_crossval = cross_val_score(gradient_ds, X_val, y_val)
print("Gradient Boosting cross-validation score after ds: {:.2f}".format(gradient_ds_crossval.mean()))

xg_ds = XGBClassifier(random_state=1).fit(Xds_train, yds_train)
print("\nXG Boosting validation score: {:.2f}".format(xg_ds.score(X_val, y_val)))
xg_ds_crossval = cross_val_score(xg_ds, X_val, y_val)
print("XG Boosting cross-validation score after ds: {:.2f}".format(xg_ds_crossval.mean()))


# In[ ]:


#confusion matrix for random forest downsampled
pred_tree_ds = tree_ds.predict(X_val)
confusion_tree_ds=confusion_matrix(y_val, pred_tree_ds)
print("\nconfusion matrix for tree(max_depth 4) after downsampling:\n{}".format(confusion_tree_ds))

pred_forest_ds = forest_ds.predict(X_val)
confusion_forest_ds=confusion_matrix(y_val, pred_forest_ds)
print("\nconfusion matrix for random forest after downsampling:\n{}".format(confusion_forest_ds))

pred_gradient_ds = gradient_ds.predict(X_val)
confusion_gradient_ds=confusion_matrix(y_val, pred_gradient_ds)
print("\nconfusion matrix for gradient boosting after downsampling:\n{}".format(confusion_gradient_ds))

pred_xg_ds = xg_ds.predict(X_val)
confusion_xg_ds=confusion_matrix(y_val, pred_xg_ds)
print("\nconfusion matrix for xg boosting after downsampling:\n{}".format(confusion_xg_ds))


# False positive performance all improved but we also sacrificed for predicting class "1" due to the loss of information

# <a id="4.2"></a>
# ### 4.2 Upsample 

# In[ ]:


df_train = X_train.copy()


# In[ ]:


df_train["Churn"] = y_train


# In[ ]:


# Separate majority and minority classes
df_maj2 = df_train[df_train.Churn==0]
df_min2 = df_train[df_train.Churn==1]

print("The majority sample size is: {}".format(len(df_maj2))) #1829


# Upsample majority class
df_min_up = resample(df_min2, replace=True,    # sample without replacement
                             n_samples=1829,     # to match minority class
                             random_state=1) # reproducible results

# Combine minority class with downsampled majority class
df_up = pd.concat([df_min_up, df_maj2])
 
# Display new class counts
df_up.Churn.value_counts()


# In[ ]:


Xup_train = df_up.iloc[:, :-1].copy()

yup_train = df_up["Churn"]


# In[ ]:


tree_up = DecisionTreeClassifier(random_state=1, max_depth = 4).fit(Xup_train, yup_train)
print("Tree(max_depth 4) validation score after ups: {:.2f}".format(tree_up.score(X_val, y_val)))
tree_up_crossval = cross_val_score(tree_up, X_val, y_val)
print("Tree(max_depth 4) cross-validation score ups: {:.2f}".format(tree_up_crossval.mean()))

forest_up = RandomForestClassifier(random_state=1).fit(Xup_train, yup_train)
print("\nRandom Forest validation score after ups: {:.2f}".format(forest_up.score(X_val, y_val)))
forest_up_crossval = cross_val_score(forest_up, X_val, y_val)
print("Random Forest cross-validation score after ups: {:.2f}".format(forest_up_crossval.mean()))

gradient_up = GradientBoostingClassifier(random_state=1).fit(Xup_train, yup_train)
print("\nGradient Boosting validation score after ups: {:.2f}".format(gradient_up.score(X_val, y_val)))
gradient_up_crossval = cross_val_score(gradient_up, X_val, y_val)
print("Gradient Boosting cross-validation score: {:.2f}".format(gradient_up_crossval.mean()))

xg_up = XGBClassifier(random_state=1).fit(Xup_train, yup_train)
print("\nXG Boosting validation score after ups: {:.2f}".format(xg_up.score(X_val, y_val)))
xg_up_crossval = cross_val_score(xg_up, X_val, y_val)
print("XG Boosting cross-validation score: {:.2f}".format(xg_up_crossval.mean()))


# In[ ]:


pred_tree_up = tree_up.predict(X_val)
confusion_tree_up=confusion_matrix(y_val, pred_tree_up)
print("\nconfusion matrix for tree(max_depth 4) after upsamling:\n{}".format(confusion_tree_up))

pred_forest_up = forest_up.predict(X_val)
confusion_forest_up=confusion_matrix(y_val, pred_forest_up)
print("\nconfusion matrix for random forest after upsamling:\n{}".format(confusion_forest_up))

pred_gradient_up = gradient_up.predict(X_val)
confusion_gradient_up=confusion_matrix(y_val, pred_gradient_up)
print("\nconfusion matrix for gradient after upsamling:\n{}".format(confusion_gradient_up))

pred_xg_up = xg_up.predict(X_val)
confusion_xg_up=confusion_matrix(y_val, pred_xg_up)
print("\nconfusion matrix for xg after upsamling:\n{}".format(confusion_xg_up))


# By keep copying the data in small sample size, it overfit those data point. That's why for testing, the False Positive rate is still high

# <a id="4.3"></a>
# ### 4.3 Manually changing the weights for RF

# In[ ]:


RF5 = RandomForestClassifier(class_weight={0: 1, 1: 5}, max_depth=4, random_state=1) #change the weight of class 1 to be 5 times bigger
rf_weighted = RF5.fit(X_train, y_train)
print("Random Forest(max_depth 4) validation score after weighted 1:5: {:.2f}".format(rf_weighted.score(X_val, y_val)))
rf_weighted_crossval = cross_val_score(rf_weighted, X_val, y_val)
print("Random Forest(max_depth 4) cross-validation score after weighted 1:5: {:.2f}".format(rf_weighted_crossval.mean()))

pred_rf_weighted = rf_weighted.predict(X_val)
confusion_rf_weighted=confusion_matrix(y_val, pred_rf_weighted)
print("\nconfusion matrix for random forest (max_depth 4) after weighted 1:5:\n{}".format(confusion_rf_weighted))

print("\n classification report for weights 1:5")
print(classification_report(y_val, pred_rf_weighted, target_names = ["Stay", "Churn"]))


# In[ ]:


RF50 = RandomForestClassifier(class_weight={0: 1, 1: 50}, max_depth=4) #change it to be 50 times
rf_weighted2 = RF50.fit(X_train, y_train)

pred_rf_weighted2 = rf_weighted2.predict(X_val)
confusion_rf_weighted2=confusion_matrix(y_val, pred_rf_weighted2)
print("\nconfusion matrix for random forest(max_depth 4) after weighted 1:50:\n{}".format(confusion_rf_weighted2))

print("\nclassification report for weights 1:50")
print(classification_report(y_val, pred_rf_weighted2, target_names = ["Stay", "Churn"]))


# <a id="4.31"></a>
# ### 4.3.1 balanced class_weight for RF and Tree

# In[ ]:


RFauto = RandomForestClassifier(class_weight='balanced', random_state = 1, max_depth=8)

rf_auto = RFauto.fit(X_train, y_train)
print("Random Forest(max_depth 8) validation score after balanced: {:.2f}".format(rf_auto.score(X_val, y_val)))
rfauto_crossval = cross_val_score(rf_auto, X_val, y_val)
print("Random Forest(max_depth 8) cross-validation score after balanced: {:.2f}".format(rfauto_crossval.mean()))

tree_auto = DecisionTreeClassifier(random_state=1, class_weight = 'balanced', max_depth=6).fit(X_train, y_train)
print("\nDecision Tree(max_depth 6) validation score after balanced: {:.2f}".format(tree_auto.score(X_val, y_val)))
treeauto_crossval = cross_val_score(tree_auto, X_trainval, y_trainval)
print("Decision Tree(max_depth 6) cross-validation after balanced: {:.2f}".format(treeauto_crossval.mean()))


# In[ ]:



pred_rfauto = rf_auto.predict(X_val)
confusion_rfauto=confusion_matrix(y_val, pred_rfauto)
print("\nconfusion matrix for random forest(max_depth 8) after balanced:\n{}".format(confusion_rfauto))

#print("\nclassification report for random forest balanced classes")
#print(classification_report(y_val, pred_rfauto, target_names = ["Stay", "Churn"]))

pred_treeauto = tree_auto.predict(X_val)
confusion_treeauto=confusion_matrix(y_val, pred_treeauto)
print("\nconfusion matrix for tree(max_depth 6) after balanced classes:\n{}".format(confusion_treeauto))


# <a id="4.4"></a>
# ### 4.4 SMOTE
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=1)
Xsm_train, ysm_train = sm.fit_resample(X_train, y_train)


# In[ ]:


#tree after smote
tree_sm = DecisionTreeClassifier(random_state=1, max_depth = 5).fit(Xsm_train, ysm_train)
print("Tree(max_depth 5) validation score after smote: {:.2f}".format(tree_up.score(X_val, y_val)))
tree_sm_crossval = cross_val_score(tree_sm, X_val, y_val)
print("Tree(max_depth 5) cross-validation score after smote: {:.2f}".format(tree_up_crossval.mean()))

#random forest model after downsampled
forest_sm = RandomForestClassifier(random_state=1, max_depth=8).fit(Xsm_train, ysm_train)
print("\nRandom Forest(max_depth 8) validation score after smote: {:.2f}".format(forest_sm.score(X_val, y_val)))
forest_sm_crossval = cross_val_score(forest_sm, X_val, y_val)
print("Random Forest(max_depth 8) cross-validation score after smote: {:.2f}".format(forest_sm_crossval.mean()))

gradient_sm = GradientBoostingClassifier(random_state=1).fit(Xsm_train, ysm_train)
print("\nGradient Boosting validation score after smote: {:.2f}".format(gradient_up.score(X_val, y_val)))
gradient_sm_crossval = cross_val_score(gradient_sm, X_val, y_val)
print("Gradient Boosting cross-validation score after smote: {:.2f}".format(gradient_sm_crossval.mean()))

xg_sm = XGBClassifier(random_state=1).fit(Xsm_train, ysm_train)
print("\nXG Boosting validation score after smote: {:.2f}".format(xg_sm.score(X_val, y_val)))
xg_sm_crossval = cross_val_score(xg_sm, X_val, y_val)
print("XG Boosting cross-validation score after smote: {:.2f}".format(xg_sm_crossval.mean()))


# In[ ]:


print("SMOTE")
pred_tree_sm = tree_sm.predict(X_val)
confusion_tree_sm=confusion_matrix(y_val, pred_tree_sm)
print("\nconfusion matrix for tree(max_depth 5) :\n{}".format(confusion_tree_sm))

pred_forest_sm = forest_sm.predict(X_val)
confusion_forest_sm=confusion_matrix(y_val, pred_forest_sm)
print("\nconfusion matrix for forest(max_depth 8) :\n{}".format(confusion_forest_sm))

pred_gradient_sm = gradient_sm.predict(X_val)
confusion_gradient_sm=confusion_matrix(y_val, pred_gradient_sm)
print("\nconfusion matrix for gradient:\n{}".format(confusion_gradient_sm))

pred_xg_sm = xg_sm.predict(X_val)
confusion_xg_sm=confusion_matrix(y_val, pred_xg_sm)
print("\nconfusion matrix for xg:\n{}".format(confusion_xg_sm))


# Performance for gradient and xg are very consistent. It's due to how their algorithms were designed. For the other tree methods, you can see the False positive rate improved a lot compare to when we used upsample. 

# <a id="5"></a>
# # 5. precision vs. recall curve

# In[ ]:


fig= plt.subplots(figsize=(8, 6))
precision_gd, recall_gd, thresholds_gd = precision_recall_curve(y_val, gradient_sm.predict_proba(X_val)[:, 1])
plt.plot(precision_gd, recall_gd, label="gd")

close_default_gd = np.argmin(np.abs(thresholds_gd - 0.5))
plt.plot(precision_gd[close_default_gd], recall_gd[close_default_gd], '^', c='k', markersize=10, label='threshold 0.5 rf', fillstyle="none", mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# Looks like we can maintain the same recall rate and improve the overall accuracy with an adjustment of the rate of threshold.

# In[ ]:


threshold = 0.38 #after different trails, this is the best

predicted_proba = gradient_sm.predict_proba(X_val)
predicted = (predicted_proba [:,1] >= threshold).astype('int')

#compare the accuracy scores
accuracy_adj = accuracy_score(y_val, predicted)
print("accurcy rate with 0.38 threshold {}".format(str(round(accuracy_adj,4,)*100))+"%")

accuracy = accuracy_score(y_val, pred_gradient_sm)
print("accurcy rate with 0.5 threshold {}".format(str(round(accuracy,4,)*100))+"%")

#confusion matrix compare
confusion_gd=confusion_matrix(y_val, predicted)
print("confusion matrix with new threshold:\n{}".format(confusion_gd))

pred_gradient_sm = gradient_sm.predict(X_val)
confusion_gd_sm=confusion_matrix(y_val, pred_gradient_sm)
print("\nconfusion matrix original:\n{}".format(confusion_gd_sm))

#classification_report
print("\nrandom forest classification report with adjuested threshold\n")
print(classification_report(y_val, predicted, target_names = ["Stay", "Churn"]))


# In[ ]:


fig= plt.subplots(figsize=(8, 6))
precision_gd, recall_gd, thresholds_gd = precision_recall_curve(y_val, gradient_sm.predict_proba(X_val)[:, 1])
plt.plot(precision_gd, recall_gd, label="gd")

close_default_gd = np.argmin(np.abs(thresholds_gd - 0.4))
plt.plot(precision_gd[close_default_gd], recall_gd[close_default_gd], '^', c='k', markersize=10, label='threshold 0.5 rf', fillstyle="none", mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# <a id="6"></a>
# # 6. final test

# In[ ]:


print("Random Forest final test score: {:.2f}".format(forest_ds.score(X_test, y_test)))
foresttest_crossval = cross_val_score(forest_ds, df_train1, label)
print("Random Forest final cross-validation test score: {:.2f}".format(foresttest_crossval.mean()))

test_forest = forest_ds.predict(X_test)
confusion_foresttest=confusion_matrix(y_test, test_forest)
print("\nconfusion matrix:\n{}".format(confusion_foresttest))

foresttest_proba = forest_ds.predict_proba(X_test)

foresttest = (foresttest_proba [:,1] >= 0.56).astype('int')

forestaccuracy_test = accuracy_score(y_test, foresttest)
#test_crossval2 = cross_val_score(test, df_train1, label)
print("\naccurcy rate with test data with 0.56 threshold is {}".format(str(round(forestaccuracy_test,4,)*100))+"%")
#print("cross-validation rate with test data is {}".format(str(round(test_crossval2,4,)*100))+"%")
      
print("\nrandom forest classification report with adjuested threshold\n")
print(classification_report(y_test, foresttest, target_names = ["Stay", "Churn"]))


# In[ ]:


print("Random Forest final test score: {:.2f}".format(forest_sm.score(X_test, y_test)))
forestsmtest_crossval = cross_val_score(forest_sm, df_train1, label)
print("Random Forest final cross-validation test score for smote: {:.2f}".format(forestsmtest_crossval.mean()))

test_forestsm = forest_sm.predict(X_test)
confusion_forestsmtest=confusion_matrix(y_test, test_forestsm)
print("\nconfusion matrix:\n{}".format(confusion_forestsmtest))

forestsmtest_proba = forest_sm.predict_proba(X_test)

forestsmtest = (forestsmtest_proba [:,1] >= 0.56).astype('int')

forestsmaccuracy_test = accuracy_score(y_test, forestsmtest)
#test_crossval2 = cross_val_score(test, df_train1, label)
print("\naccurcy rate with test data with 0.56 threshold is {}".format(str(round(forestsmaccuracy_test,4,)*100))+"%")
#print("cross-validation rate with test data is {}".format(str(round(test_crossval2,4,)*100))+"%")
      
print("\nrandom forest classification report with adjuested threshold\n")
print(classification_report(y_test, forestsmtest, target_names = ["Stay", "Churn"]))


# In[ ]:


print("Gradient Boosting final test score: {:.2f}".format(gradient_sm.score(X_test, y_test)))
gdtest_crossval = cross_val_score(gradient_sm, df_train1, label)
print("Gradient Boosting final cross-validation test score: {:.2f}".format(gdtest_crossval.mean()))

test_gd = gradient_sm.predict(X_test)
gdconfusion_test=confusion_matrix(y_test, test_gd)
print("\nconfusion matrix:\n{}".format(gdconfusion_test))

test_gd_original = gradient.predict(X_test)
gdoconfusion_test=confusion_matrix(y_test, test_gd_original)
print("\nconfusion matrix without sm:\n{}".format(gdoconfusion_test))

gdtest_proba = gradient_sm.predict_proba(X_test)

gdtest = (gdtest_proba [:,1] >= 0.38).astype('int')

gdconfusion_test2=confusion_matrix(y_test, gdtest)
print("\nconfusion matrix with 0.38 threshold:\n{}".format(gdconfusion_test2))

gdaccuracy_test = accuracy_score(y_test, gdtest)
print("\naccurcy rate with test data with 0.38 threshold is {}".format(str(round(gdaccuracy_test,4,)*100))+"%")

      
print("\ngradient boost classification report with adjuested threshold\n")
print(classification_report(y_test, gdtest, target_names = ["Stay", "Churn"]))

test_gdo = gradient.predict(X_test)
print("\ngradient boost classification report original\n")
print(classification_report(y_test, test_gdo, target_names = ["Stay", "Churn"]))


# In[ ]:




