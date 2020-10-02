#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


new_data=pd.read_csv("/kaggle/input/new-data/new_data (1).csv")


# In[ ]:


print(new_data.shape)


# In[ ]:


X_safe=new_data.drop(['Client Retention Flag'], axis=1)
Y_safe=new_data['Client Retention Flag']
print(X_safe.shape)
print(Y_safe.shape)


# # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# In[ ]:


X=pd.get_dummies(X_safe)
Y=Y_safe

from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_tr,y_tr)
y_pre=model.predict(x_val)

from sklearn.metrics import f1_score
print(y_pre)
f1=f1_score(y_val,y_pre,pos_label='Yes')
print(f1)


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.9, gamma=0,
                              learning_rate=0.1, max_delta_step=0, max_depth=10,
                              min_child_weight=1, missing=None, n_estimators=200, n_jobs=-1,
                              nthread=None, objective='binary:logistic', random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                              silent=None, subsample=0.9, verbosity=0)
classifier.fit(x_tr, y_tr)

# Predicting the Test set results
y_pred = classifier.predict(x_val)
f1=f1_score(y_val,y_pred,pos_label='Yes')
print(f1)


# # ____ ____AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA___

# In[ ]:


df=new_data
print(df.isnull().sum())
print(df.describe())
#print(df.isnull().sum().max())
df=df.drop_duplicates()


# In[ ]:


df['Client Retention Flag'].value_counts()


# In[ ]:



print('Retained', round(df['Client Retention Flag'].value_counts()['Yes']/len(df) * 100,2), '% of the dataset')
print('Not Retained', round(df['Client Retention Flag'].value_counts()['No']/len(df) * 100,2), '% of the dataset')
print('Retained', round(df['Client Retention Flag'].value_counts()['Yes']))
print('Not Retained', round(df['Client Retention Flag'].value_counts()['No']))


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

flag_2= df['Flag 2'].values
flag_5= df['Flag 5'].values

sns.distplot(flag_2, ax=ax[0], color='r')
ax[0].set_title('Distribution of Flag 2', fontsize=14)
ax[0].set_xlim([min(flag_2), max(flag_2)])

sns.distplot(flag_5, ax=ax[1], color='b')
ax[1].set_title('Distribution of Flag 5', fontsize=14)
ax[1].set_xlim([min(flag_5), max(flag_5)])



plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(18,4))

ac_1_9= df['Activity 1 Time Period 9'].values
ac_1_10= df['Activity 1 Time Period 10'].values
ac_1_11=df['Activity 1 Time Period 11'].values

sns.distplot(ac_1_9, ax=ax[0], color='r',rug=True, hist=False)
ax[0].set_title('Activity 1 Time Period 9', fontsize=14)
ax[0].set_xlim([min(ac_1_9),200])

sns.distplot(ac_1_10, ax=ax[1], color='b',rug=True, hist=False)
ax[1].set_title('Activity 1 Time Period 10', fontsize=14)
ax[1].set_xlim([min(ac_1_10), 200])
                
sns.distplot(ac_1_11, ax=ax[2], color='b',rug=True, hist=False)
ax[2].set_title('Activity 1 Time Period 11', fontsize=14)
ax[2].set_xlim([min(ac_1_11), 200])


plt.show()


# ## Splitting into train-test data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X = df.drop('Client Retention Flag', axis=1)
y = df['Client Retention Flag']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


# ## Random Undersampling

# In[ ]:


df = df.sample(frac=1,random_state=3)

# amount of fraud classes 492 rows.
not_retained= df.loc[df['Client Retention Flag'] == 'No']
retained = df.loc[df['Client Retention Flag'] == 'Yes'][:3000]

normal_distributed_df = pd.concat([retained, not_retained])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

print(new_df.shape)
new_df.head()
new_df.drop_duplicates().shape


# In[ ]:


cols=['Client Contract Starting Month', 'Flag 1', 'Flag 2', 'Flag 3',
       'Flag 5', 'Activity 1 Time Period 11', 'Activity 1 Time Period 10',
       'Activity 1 Time Period 9', 'Activity 1 Time Period 8',
       'Activity 1 Time Period 7', 'Activity 1 Time Period 6',
       'Activity 1 Time Period 5', 'Activity 1 Time Period 4',
       'Activity 1 Time Period 3', 'Activity 1 Time Period 2',
       'Activity 1 Time Period 1', 'Activity 1 Time Period 0','Flag 6', 'Activity 2 to 4',
       'Activity 5 to 8','Client Retention Flag']
new_df=new_df[cols]
new_df.head()


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['Yes','No'])
Class=le.transform(new_df['Client Retention Flag'])
new_df['Class']=Class
new_df['Class'].head()
new_df.drop(['Client Retention Flag'],axis=True,inplace=True)
new_df.head()


# In[ ]:


print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[ ]:


f, (ax1) = plt.subplots(1, 1, figsize=(10,10))

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


# In[ ]:


Correalation_Matrix = new_df[new_df.columns[1:]].corr()['Class'][:]
Correlation_Matrix = pd.DataFrame(Correalation_Matrix)
Correlation_Matrix.Class.plot(figsize=(10,5))
plt.ylabel('Correlation Score')
plt.xlabel('Features')

new_df.corrwith(new_df.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with class", fontsize = 15,
        rot = 45, grid = True)


# In[ ]:


pos_corr=sub_sample_corr.index[sub_sample_corr['Class'] >0.172].tolist()#these eatures
#pos_corr conains 3 highest +vely correlated features


# In[ ]:


f, axes = plt.subplots(ncols=3, figsize=(20,10))

sns.boxplot(x="Class", y="Activity 1 Time Period 5", data=new_df, ax=axes[0])
axes[0].set_title('Activity 1 Time Period 5 vs Class +ve Correlation')

sns.boxplot(x="Class", y="Activity 1 Time Period 3", data=new_df, ax=axes[1])
axes[1].set_title('Activity 1 Time Period 3 vs Class +ve Correlation')


sns.boxplot(x="Class", y="Activity 1 Time Period 1", data=new_df,  ax=axes[2])
axes[2].set_title('Activity 1 Time Period 1 vs Class +ve Correlation')


plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(18,4))

ac_1_5= new_df['Activity 1 Time Period 5'].loc[new_df['Class']==0].values
ac_1_3= new_df['Activity 1 Time Period 3'].loc[new_df['Class']==0].values
ac_1_1=new_df['Activity 1 Time Period 1'].loc[new_df['Class']==0].values

sns.distplot(ac_1_5, ax=ax[0], color='r',rug=True, hist=False)
ax[0].set_title('Activity 1 Time Period 5 Not retained', fontsize=14)
ax[0].set_xlim([min(ac_1_9),400])

sns.distplot(ac_1_5, ax=ax[1], color='b',rug=True, hist=False)
ax[1].set_title('Activity 1 Time Period 3 Not retained', fontsize=14)
ax[1].set_xlim([min(ac_1_10), 400])
                
sns.distplot(ac_1_1, ax=ax[2], color='b',rug=True, hist=False)
ax[2].set_title('Activity 1 Time Period 1 Not retained', fontsize=14)
ax[2].set_xlim([min(ac_1_11), 400])


plt.show()


# In[ ]:


new_df2 = new_df.drop(new_df[(new_df['Activity 1 Time Period 5'] > 275) | (new_df['Activity 1 Time Period 3'] > 275)| (new_df['Activity 1 Time Period 1'] > 275)].index)
print(new_df2.shape)
print(new_df.shape)


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(18,4))

ac_1_5= new_df2['Activity 1 Time Period 5'].loc[new_df2['Class']==0].values
ac_1_3= new_df2['Activity 1 Time Period 3'].loc[new_df2['Class']==0].values
ac_1_1=new_df2['Activity 1 Time Period 1'].loc[new_df2['Class']==0].values

sns.distplot(ac_1_5, ax=ax[0], color='r',rug=True, hist=False)
ax[0].set_title('Activity 1 Time Period 5 Not retained', fontsize=14)
ax[0].set_xlim([min(ac_1_9),400])

sns.distplot(ac_1_5, ax=ax[1], color='b',rug=True, hist=False)
ax[1].set_title('Activity 1 Time Period 3 Not retained', fontsize=14)
ax[1].set_xlim([min(ac_1_10), 400])
                
sns.distplot(ac_1_1, ax=ax[2], color='b',rug=True, hist=False)
ax[2].set_title('Activity 1 Time Period 1 Not retained', fontsize=14)
ax[2].set_xlim([min(ac_1_11), 400])


plt.show()


# In[ ]:


new_df2.head()


# In[ ]:


X.head()


# In[ ]:


X = new_df2.drop('Class', axis=1)
y = new_df2['Class']

X=pd.get_dummies(X)
"""
#Effect Coding Scheme################################################################### f1 decreased
gen_onehot_features=pd.get_dummies(X['Client Contract Starting Month'])
gen_effect_features = gen_onehot_features.iloc[:,:-1]
gen_effect_features.loc[np.all(gen_effect_features == 0, axis=1)] = -1
X.drop(['Client Contract Starting Month'],axis=1,inplace=True)
X=pd.concat([X, gen_effect_features], axis=1)

gen_onehot_features=pd.get_dummies(X['Flag 1'])
gen_effect_features = gen_onehot_features.iloc[:,:-1]
gen_effect_features.loc[np.all(gen_effect_features == 0, axis=1)] = -1
X.drop(['Flag 1'],axis=1,inplace=True)
X=pd.concat([X, gen_effect_features], axis=1)

gen_onehot_features=pd.get_dummies(X['Flag 6'])
gen_effect_features = gen_onehot_features.iloc[:,:-1]
gen_effect_features.loc[np.all(gen_effect_features == 0, axis=1)] = -1
X.drop(['Flag 6'],axis=1,inplace=True)
X=pd.concat([X, gen_effect_features], axis=1)

gen_onehot_features=pd.get_dummies(X['Flag 3'])
gen_effect_features = gen_onehot_features.iloc[:,:-1]
gen_effect_features.loc[np.all(gen_effect_features == 0, axis=1)] = -1
X.drop(['Flag 3'],axis=1,inplace=True)
X=pd.concat([X, gen_effect_features], axis=1)
"""

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# In[ ]:


classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}
# Wow our scores are getting even high scores even when applying cross validation.
from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 3) * 100, "% accuracy score")


# In[ ]:


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=1000,learning_rate=0.85,random_state=6)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(f1_score(y_test,y_pred))


# In[ ]:


#XGBoost Algorithm
from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.9, gamma=0,
                              learning_rate=0.12, max_delta_step=0, max_depth=8,
                              min_child_weight=1, missing=None, n_estimators=200, n_jobs=-1,
                              nthread=None, objective='binary:logistic', random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                              silent=None, subsample=0.9, verbosity=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
f1=f1_score(y_test,y_pred)
print(f1)


# In[ ]:


#CatBoost Algorithm

X_cat = new_df2.drop('Class', axis=1)
y_cat = new_df2['Class']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
X_train_cat = X_train_cat.values
X_test_cat = X_test_cat.values
y_train_cat = y_train_cat.values
y_test_cat = y_test_cat.values

from catboost import CatBoostClassifier
cat = CatBoostClassifier(random_state=6,logging_level='Silent',eval_metric='AUC',
                         depth=6,learning_rate=0.1,max_leaves=26)
cat.fit(X_train_cat,y_train_cat)
y_pred_cat=cat.predict(X_test_cat)
print(f1_score(y_test_cat,y_pred_cat))


# In[ ]:


#Ensembling doest improve result
from sklearn.ensemble import VotingClassifier

model1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.9, gamma=0,
                              learning_rate=0.12, max_delta_step=0, max_depth=8,
                              min_child_weight=1, missing=None, n_estimators=200, n_jobs=-1,
                              nthread=None, objective='binary:logistic', random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                              silent=None, subsample=0.9, verbosity=0)
model2 = AdaBoostClassifier(n_estimators=1000,learning_rate=0.85,random_state=6)
model = VotingClassifier(estimators=[('xgb', model1), ('adc', model2)],voting='soft')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f1_score(y_test,y_pred))


# In[ ]:


new_df3=new_df2
new_df3['Client Contract Starting Month'] = new_df2['Client Contract Starting Month'].groupby(new_df2['Client Contract Starting Month']).transform('count')
(new_df3.head(20))


# # BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

# In[ ]:


"""
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_
"""


# In[ ]:


log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')


# In[ ]:


cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.45, 0.90), cv=cv, n_jobs=4)


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)


# In[ ]:


from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))


# # BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
