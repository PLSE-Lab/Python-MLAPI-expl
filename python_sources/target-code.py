# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Importing the libraries
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict


# importing file excel file and converting to csv
data_xl = pd.read_excel('Data for classification.xlsx', 'Sheet1', index = None)
data_xl.to_csv('data_for_classification.csv', encoding='utf-8')

# converting to pandas dataframe
data = pd.read_csv('data_for_classification.csv', index_col = None)
# data.drop(data.ix[:,'Unnamed: 0'].head(0).columns, axis=1)
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Imputing of missing values with zeros
# data = data.replace(np.nan, 0)
data.replace(np.nan, 0, inplace=True)

## Encoding dummy variables
# data['feature_10'].unique().tolist()
# d={}
# for col in data:
#     d[col] = data[col].unique().tolist()
data = pd.get_dummies(data, prefix=['feature_10_dum'], columns=['feature_10'])

# sorting variables in order
tar_df = data['Target']
data.drop(['Target'], axis=1, inplace=True)
new_data = pd.concat([data, tar_df], axis = 1, ignore_index = False)

# removing dummy variable which has high multicollinearity among other input features
new_data.drop(['feature_10_dum_0.0'], axis = 1, inplace = True)
# seperating input and output variables
X = new_data.iloc[:, 1:24].values
# X = new_data.iloc[:, new_data.columns != 'feature_10_dum_0'].values
y = new_data.iloc[:, -1].values
pd.factorize(new_data['Target'].values)[0].reshape(-1, 1)
# print(new_data) # print(X) # print(y) # new_data.head(25) # X.tolist(25) # y.tolist(25)

# # checking for outliers
# down_quantiles = data.quantile(0.05)
# outliers_low = (data.iloc[:, 1:] < down_quantiles)
# high_quantiles = data.quantile(.95)
# outliers_high = (data.iloc[:, 1:] > high_quantiles)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 0)

# Feature Scaling using statndardisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Multicollinearity check using correlation matrix and vif method
# correlation matrix
new_data.corr()
# new_data.corr(cutoff = .75)

# vif values
X_df = pd.DataFrame(X)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
# print(vif)

# finding important variables using p-values by fitting Decision tree
classifier_dtree_pval = DecisionTreeClassifier()
classifier_dtree_pval.fit(X_train, y_train)
p_values = classifier_dtree_pval.feature_importances_

# finding coefficients of variables by fitting Logistic regression
classifier_log_reg_coef = LogisticRegression(class_weight="balanced", random_state = 0)
classifier_log_reg_coef.fit(X_train, y_train)
coefficients =  classifier_log_reg_coef.coef_[0]

# Fitting Logistic Regression to the Training set for classification
class_weight = {0: 1.,
                1: 6.}
classifier_log_reg = LogisticRegression(class_weight = class_weight , random_state = 0) # class_weight = 'balanced' class_weight={0:1,1:1}
classifier_log_reg.fit(X_train, y_train)

# Predicting the train set results
y_pred_log_reg_train = classifier_log_reg.predict(X_train)
# misclassified_samples = X_train[y_train != y_pred_log_reg_train]
# Predicting the Test set results
y_pred_log_reg = classifier_log_reg.predict(X_test)
# y_pred_log_reg = (y_pred_log_reg > 0.4)

# Making the Confusion Matrix for Logistic Regression
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
cm_log_reg_train = confusion_matrix(y_train, y_pred_log_reg_train)
# evaluating results/output
print('Logistic regression report: \n', classification_report(y_test, y_pred_log_reg))
accuracy_log_reg = classifier_log_reg.score(X_test, y_test)
print('accuracy of Logistic regression: ', accuracy_log_reg, '\n')
log_reg_roc_auc = roc_auc_score(y_test, classifier_log_reg.predict(X_test))
fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, classifier_log_reg.predict_proba(X_test)[:,1])

# fitting Decision tree to the Training set for classification
classifier_dtree = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = None, max_leaf_nodes = None,
                                          min_weight_fraction_leaf = 0.0, min_impurity_decrease = 0.0, min_samples_split= 2)
classifier_dtree.fit(X_train, y_train)
# Predicting the train set results
y_pred_dt_train = classifier_dtree.predict(X_train)
# Predicting the test set results
y_pred_dt_test  = classifier_dtree.predict(X_test)

# Making the Confusion Matrix for Decision tree
cm_dtree_train = confusion_matrix(y_train, y_pred_dt_train)
cm_dtree_test = confusion_matrix(y_test, y_pred_dt_test)
# evaluating results/output
print('Decision tree report: \n', classification_report(y_test, y_pred_dt_test))
accuracy_dtree = classifier_dtree.score(X_test, y_test)
print('accuracy of Decisin tree: ', accuracy_dtree, '\n')
dtree_roc_auc = roc_auc_score(y_test, classifier_dtree.predict(X_test))
# fpr_dtree, tpr_dtree, thresholds_dtree = roc_curve(y_test, classifier_dtree.predict_proba(X_test)[:,1])

# Fitting Support Vector Machine to the Training set
classifier_svm = SVC(kernel = 'linear', class_weight = class_weight, random_state = 0) # class_weight = 'balanced'
classifier_svm.fit(X_train, y_train)

# Predicting the train set results
y_pred_svm_train = classifier_svm.predict(X_train)
# Predicting the Test set results
y_pred_svm_test = classifier_svm.predict(X_test)

# Making the Confusion Matrix for Support Vector Machine
cm_svm = confusion_matrix(y_test, y_pred_svm_test)
cm_svm_train = confusion_matrix(y_train, y_pred_svm_train)
# evaluating results/output
print('Support vector machine report: \n', classification_report(y_test, y_pred_svm_test))
accuracy_svm = classifier_svm.score(X_test, y_test)
print('accuracy of SVM: ', accuracy_svm)
svm_roc_auc = roc_auc_score(y_test, classifier_svm.predict(X_test))
# svm_dtree, tpr_svm, thresholds_svm = roc_curve(y_test, classifier_svm.predict_proba(X_test)[:,1])

# K-fold cross-validation to find bias-variance tradeoff, overfitting/underfitting

# for logistic regression model
accuracies_log_reg = cross_val_score(estimator = classifier_log_reg, X = X_train, y = y_train, cv = 10, scoring= 'accuracy')
# if our goal is area under curve-- scoring='roc_auc' instead of accuaracy
accuracies_log_reg.mean()
accuracies_log_reg.std()

# for Decision tree model
accuracies_dtree = cross_val_score(estimator = classifier_dtree, X = X_train, y = y_train, cv = 10, scoring= 'accuracy')
accuracies_dtree.mean()
accuracies_dtree.std()

# for SVM
accuracies_svm = cross_val_score(estimator = classifier_svm, X = X_train, y = y_train, cv = 10, scoring= 'accuracy')
accuracies_svm.mean()
accuracies_svm.std()

# ROC Curve for Logistic regression model
# plt.figure()
# plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression (area = %0.2f)' % log_reg_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()

# Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
# {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
# 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
# grid_search = GridSearchCV(estimator = classifier_svm,
# param_grid = parameters,scoring = 'accuracy',
# cv = 10,n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_



