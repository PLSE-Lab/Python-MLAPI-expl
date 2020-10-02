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


# Load the dataframe
dataframe = pd.read_csv("../input/santander-customer-satisfaction/train.csv")

# Print the first 20 rows
dataframe.head(20)


# In[ ]:


# We print the sum of NaNs in each coloumn
np.isnan(dataframe).sum()


# In[ ]:


# Assigning data to the train dataframe
train_data = dataframe


# In[ ]:


# Dropping Target as we are supposed to predict that
# Dropping ID as it is unique to all rows
train_data = train_data.drop(['TARGET'], axis = 1)
train_data = train_data.drop(['ID'], axis = 1)


# ## Removing redundant and low variance features

# In[ ]:


# We append the coloumn names that have 0 standard deviation as we can't gather much info from these due to low variance
remove_col_std = []
for i in train_data.columns:
    if(train_data[i].std() == 0):
        remove_col_std.append(i)


# In[ ]:


# Redefining train dataframe by removing the 0 standard deviation coloumns
train_data = train_data.drop(remove_col_std, axis = 1)


# In[ ]:


# Removing columns that are identical to one another
remove_col_redund = []
count = 0
for i in range(len(train_data.columns)):
    i_values = train_data[train_data.columns[i]].values
    for j in range(i+1, len(train_data.columns)):
        if(np.array_equal(i_values, train_data[train_data.columns[j]].values)):
            remove_col_redund.append(train_data.columns[j])


# In[ ]:


# Redefining the train dataframe once more by dropping redundant coloumns
train_data = train_data.drop(remove_col_redund, axis = 1)


# ## Feature Selection using Correlation-Heatmap

# Correlation of first 10 features with TARGET

# In[ ]:


# We select the first 20 features and the target coloumn
first_df = pd.concat([train_data.iloc[:, :20], train_data.iloc[:, 305]], axis = 1)


# In[ ]:


# We print the correlation heatmap for these 20 features with the target variable
plt.figure(figsize = (20 ,20))
corrmat = first_df.corr()
top_corr_features = corrmat.index
sns.heatmap(first_df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')


# Intresting features:
# <ol>
#     <li>var15</li>
# </ol>

# Implementing the above strategy for the rest of the features. <br>We take the correlation boundary as 0.020. This means all features above or equal to this correlation value will be considered as an intresting feature.
# 
# NOTE: Since the scale for correlation changes, evaluate by the value in each box rather than the colour.

# Ideally, try to take features which are having either a correlation score closer to 1 or -1

# We may have to implement the same process for all the features to check how each feature correlates with the target variable which can be quite tedious given the number of features.

# In[ ]:


first_df = pd.concat([train_data.iloc[:, 296:306], train_data.iloc[:, 305]], axis = 1)


# In[ ]:


plt.figure(figsize = (20 ,20))
corrmat = first_df.corr()
top_corr_features = corrmat.index
sns.heatmap(first_df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')


# Seems that ind_var2_0, ind_var2, ind_var27_0, ind_var28_0, ind_var28, ind_var27 don't have any correlation with any feature.
# 
# Intresting features:
# <ol>
#     <li>var15 = 0.1</li>
#     <li>ind_var8_0 = 0.047</li>
#     <li>ind_var8 = 0.028</li>
#     <li>ind_var26_cte = 0.024</li>
#     <li>ind_var25_cte = 0.023</li>
#     <li>num_var8_0 = 0.047</li>
#     <li>num_var8 = 0.028</li>
#     <li>var36 = 0.1</li>
#     <li>num_var22_ult1 = 0.025</li>
#     <li>num_meses_var8_ult3 = 0.026</li>
#     <li>num_op_var41_efect_ult1 = 0.021</li>
#     <li>num_op_var41_efect_ult3 = 0.02</li>
#     <li>num_op_var39_efect_ult1 = 0.022</li>
#     <li>num_op_var39_efect_ult3 = 0.02</li>
# </ol>

# ## Feature Selection using K-Best

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# We segregate the data features into 'X' dataframe and the target variable in the 'y' dataframe

# In[ ]:


X = train_data.iloc[:, :306]
y = dataframe.iloc[:, 370]


# We use the ANOVA test to determine the feature importance. This is done through the 'f_classif' method.
# 
# 'k' determines how many features are we going to select

# In[ ]:


bestfeatures = SelectKBest(score_func = f_classif, k = 40)
fit = bestfeatures.fit(X, y)


# We create two dataframes. One containing the scores for each feature and the second dataframe, the feature itself

# In[ ]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[ ]:


featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Features', 'Score']


# In[ ]:


featureScores


# We sort the top 40 features

# In[ ]:


print(featureScores.nlargest(40, 'Score'))


# Plotting the top 30 features

# In[ ]:


plt.figure(figsize = (25, 6))
sns.barplot(x = featureScores.nlargest(30, 'Score')['Features'], y = featureScores.nlargest(30, 'Score')['Score'])
plt.xticks(rotation = 45)
ax = plt.gca()
plt.show()


# ## Feature Selection using Feature Importance

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


model = ExtraTreesClassifier()
selector = model.fit(X, y)


# In[ ]:


plt.figure(figsize = (60, 40))
feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(40).plot(kind = 'barh')
plt.show()


# ## Class distribution

# In[ ]:


plt.figure(figsize = (10, 6))
sns.countplot(dataframe['TARGET'].values)


# Clearly there's a lot of class imbalance

# This means we may have to use algorithms such as XGBoost or AdaBoost as they are known to be resilient to such class imbalance problem.

# Satisfied customers:-

# In[ ]:


dataframe[dataframe['TARGET'] == 0].shape[0]


# Unsatisfied customers:-

# In[ ]:


dataframe[dataframe['TARGET'] == 1].shape[0]


# ## Selecting a feature set based on ExtraTreesClassifier

# In[ ]:


# List of the top 40 features selected by the ExtraTreesClassifier
list(feat_importances.nlargest(40).index)


# In[ ]:


X_feat_1 = X[list(feat_importances.nlargest(40).index)]


# We determine the shape of the matrix and confirm if the shape of rows for the input matrix 'X' is same as that of 'y' 

# In[ ]:


print("X shape: " +  str(X_feat_1.shape) + " || y shape: " + str(y.shape))


# Next we need to implement  train_test_split and feature scaling

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_feat_1, y, test_size = 0.25, random_state = 4)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   
}


# In[ ]:


# Instantiating the XGBClassfier
classifier_etc = XGBClassifier()


# Using the 'roc_auc' scoring parameter as that is the metrics to be evaluated in the problem statement

# In[ ]:


random_search = RandomizedSearchCV(classifier_etc, param_distributions = params, n_iter = 5, scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)


# We implement random search CV technique as this will search the search space randomly and will attempt to find the best set of hyperparameters

# In[ ]:


random_search.fit(X_train,y_train)


# Once we have fit the data, we can directly get the best model parameters through the 'best_estimator_' method

# In[ ]:


random_search.best_estimator_


# The optimal parameters can be extracted through the model

# In[ ]:


random_search.best_params_


# In[ ]:


classifier_etc = random_search.best_estimator_


# Attempting to get the cross validation score

# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier_etc,X,y,cv=10)


# In[ ]:


score


# In[ ]:


score.mean()


# In[ ]:


classifier_etc.fit(X_train, y_train)


# In[ ]:


y_pred = classifier_etc.predict(X_test)


# Printing the confusion matrix for the same

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# And our ROC curve score is...

# In[ ]:


from sklearn.metrics import roc_auc_score
print("Roc AUC: ", roc_auc_score(y_test, classifier_etc.predict_proba(X_test)[:,1], average='macro'))


# ## Selecting a feature set based on K-Best

# We repeat the same process as before

# In[ ]:


list(featureScores.nlargest(40, 'Score')['Features'])


# In[ ]:


X_feat_2 = X[list(featureScores.nlargest(40, 'Score')['Features'])]


# In[ ]:


print("X shape: " +  str(X_feat_2.shape) + " || y shape: " + str(y.shape))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_feat_2, y, test_size = 0.20, random_state = 4)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   
}


# In[ ]:


classifier_k_best = XGBClassifier()


# In[ ]:


random_search = RandomizedSearchCV(classifier_k_best, param_distributions = params, n_iter = 5, scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)


# In[ ]:


random_search.fit(X_train,y_train)


# In[ ]:


random_search.best_estimator_


# In[ ]:


random_search.best_params_


# In[ ]:


classifier_k_best = random_search.best_estimator_


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier_k_best,X,y,cv=10)


# In[ ]:


score


# In[ ]:


score.mean()


# In[ ]:


classifier_k_best.fit(X_train, y_train)


# In[ ]:


y_pred = classifier_k_best.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import roc_auc_score
print("Roc AUC: ", roc_auc_score(y_test, classifier_k_best.predict_proba(X_test)[:,1], average='macro'))


# ## Finalizing

# Performing the final steps for result submission

# In[ ]:


# Creating the test dataframe
test_df = pd.read_csv("../input/santander-customer-satisfaction/test.csv")


# In[ ]:


test_df_X = test_df[list(feat_importances.nlargest(40).index)]


# In[ ]:


test_df_X.shape


# In[ ]:


test_df_X = sc.fit_transform(test_df_X)


# In[ ]:


test_ID = test_df.ID


# In[ ]:


probs = classifier_etc.predict_proba(test_df_X)


# In[ ]:


submission = pd.DataFrame({"ID":test_ID, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)


# In[ ]:




