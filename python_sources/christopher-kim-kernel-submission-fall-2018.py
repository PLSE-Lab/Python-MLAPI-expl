# Christopher Kim
# MaKeene Learning 2018 (Fall)
# Kaggle Contest Kernel Submission

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

''' 
Many experiments were conducted for this kaggle competition, from data preprocessing to hyperparameter turning. 
I primarily focused on xgboost and random forest, since I believed that those would give me the best results. 
I had a lot of fun with this contest, and I've combined the required methods used into one kernel and showed an 
additional experiment conducted for this competition. In terms of results, I am probably missing an approach in
which the others are getting very successful results. Maybe it is not using xgboost or random forest, or maybe it
is a pre-procssing touch that I have not considered which is why my results are inferior. Nevertheless, I am 
excited to see the top-solution implementation. 

In terms of the code, all of the requirements have been sorted in the order listed in the email, and each code segment
is commnted out with triple strings that can simply be deleted to run a specific segment. Many of my other experiments
involving hyperparameter tuning and different combination approaches are not shown here.
'''

# Data preprocessing
train_set = pd.read_csv('../input/trainFeatures.csv')
test_set = pd.read_csv('../input/testFeatures.csv')
train_label = pd.read_csv('../input/trainLabels.csv')

# Obtain the ids for later processing the output to csv
test_ids = test_set['ids']

# Experiments with taking out features that didn't seem to have anything to do with the label were conducted.
# This is one example. However, in the end it was most effective to just keep all of the data except for the 'erkey' column
#train_set = train_set.drop(columns=['ids','RatingID','erkey','AccountabilityID'])
#test_set = test_set.drop(columns=['ids','RatingID','erkey','AccountabilityID'])

train_set = train_set.drop(columns=['erkey'])
test_set = test_set.drop(columns=['erkey'])

X = train_set.fillna(0) # fill the nans with 0 value
y = train_label['OverallScore'].fillna(0) # fill the nans with 0 value
X_test = test_set.fillna(0) # fill the nans with 0 value

''' #Basic Linear Regression | Kaggle RMSE: 6.933 (Standard data preprocessing)
linear_regr_model = linear_model.LinearRegression(fit_intercept=True, normalize=False)
linear_regr_model.fit(X,y)
preds = linear_regr_model.predict(X_test)
'''

''' #Lasso Regression | Kaggle RMSE: 6.669 (Standard data preprocessing)
lasso_regr_model = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=100000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso_regr_model.fit(X,y)
preds = lasso_regr_model.predict(X_test)
'''

''' #Ridge Regression | Kaggle RMSE: 6.933 (Standard data preprocessing)
ridge_regr_model = linear_model.Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
   tol=0.001, solver='auto', random_state=None)
ridge_regr_model.fit(X,y)
preds = ridge_regr_model.predict(X_test)
'''

''' #Random Forest Regression | Kaggle RMSE: 4.0212 (Standard data preprocessing)
RF_model = RandomForestRegressor(n_estimators = 350, bootstrap = True, verbose=0, min_samples_split=2)
RF_model.fit(X,y)
preds = RF_model.predict(X_test)
'''

''' #XGB Regression | Kaggle RMSE: 3.96132 (Standard data preprocessing)
xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05,n_estimators=387, 
	booster='gbtree', base_score=0.5, random_state=0, reg_lambda=1)
xgb_model.fit(X,y)
preds = xgb_model.predict(X_test)
'''

''' #Support Vector Regression "not covered in class"| Kaggle RMSE: 6.7569 (Standard data preprocessing)
svm_model = SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1)
svm_model.fit(X,y)
preds = svm_model.predict(X_test)
'''

''' #Unsupervised Dimensionality Reduction with XGB Regression | Kaggle RMSE: 5.07 (PCA)
pca = PCA(n_components=40)
pca.fit(X)
X_train_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)

xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05,n_estimators=387,
	booster='gbtree', base_score=0.5, random_state=0, reg_lambda=1)
xgb_model.fit(X_train_pca,y)
preds = xgb_model.predict(X_test_pca)
'''

''' #Feature Selection Method with XGB Regression | Kaggle RMSE: 4.9457 (Select K-Best with f_regression)
feat_sel = SelectKBest(f_regression, k=20).fit(X,y)
X_train_new = feat_sel.transform(X)
X_test_new = feat_sel.transform(X_test)

xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05,n_estimators=387)
xgb_model.fit(X_train_new,y)
preds = xgb_model.predict(X_test_new)
'''

'''
#Format submission
submission = pd.DataFrame()
submission['Id'] = test_ids
submission['OverallScore'] = preds
submission.to_csv('submission_chriskim.csv',index=False)
'''