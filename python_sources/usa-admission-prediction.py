import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import norm

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../input/Admission_Predict.csv')

data.info() #No missing values
data.shape
data.head()

col_names = data.columns.tolist()
no_of_columns = len(col_names)
[i for i in col_names]

#No use with serial no. column
data.drop(['Serial No.'], axis = 1, inplace = True)

data.describe()

#rug-divisions, fit-normal distribution, kde-distribution of data, 
sns.distplot(data['GRE Score'], rug = True, rug_kws={"color": "black"}, fit=norm, kde_kws={"color": "green", "lw": 2, "label": "kde"})

sns.distplot(data['TOEFL Score'], rug = True, rug_kws={"color": "black"}, fit=norm, kde_kws={"color": "green", "lw": 2, "label": "kde"})


features = data.iloc[:, :-1]
label = data.iloc[:, (len(data.columns) - 1)]

from sklearn.model_selection import train_test_split
features_train, features_test, label_train, label_test = train_test_split(features,label,test_size = 0.30)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_train.columns
features_train[features_train.columns] = scaler.fit_transform(features_train[features_train.columns])
features_test[features_test.columns] = scaler.transform(features_test[features_test.columns])

#Model BUilding

#1. Multiple_Linear_Regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(features_train, label_train)
label_pred_test = linear_regressor.predict(features_test)

label_pred_train = linear_regressor.predict(features_train)

#Evaluation Metrics
from sklearn.metrics import mean_squared_error
MSE_Test = mean_squared_error(label_test, label_pred_test)

from sklearn.metrics import mean_squared_error
MSE_Train = mean_squared_error(label_train, label_pred_train)

from sklearn.metrics import r2_score
R2_Score_lr = r2_score(label_test, label_pred_test)

from sklearn.metrics import r2_score
R2_Score_lr_Train = r2_score(label_train, label_pred_train)

#2.POlynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2)
features_train_poly = poly_regressor.fit_transform(features_train)
features_test_poly = poly_regressor.fit_transform(features_test)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(features_train_poly, label_train)
labels_pred_poly = linear_regressor_2.predict(features_test_poly)

R2_Score_poly = r2_score(labels_pred_poly, label_test)

#3.Support Vector Regression
from sklearn.svm import SVR

# from sklearn.model_selection import GridSearchCV
# parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']}, 
# {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001]}]                                          ]
# grid_search = GridSearchCV(estimator = svc_classifier, param_grid = parameters, scoring = 'accuracy')
# grid_search = grid_search.fit(features_train, label_train)
# Label shldnt be cont. otherwise we get fitting error

SVM_regressor = SVR('rbf')
SVM_regressor.fit(features_train, label_train)
labels_pred_svm = SVM_regressor.predict(features_test)
R2_Score_svm = r2_score(labels_pred_svm, label_test)

#4. DecisionTree Regression
from sklearn.tree import DecisionTreeRegressor
DT_regressor = DecisionTreeRegressor()

DT_regressor.fit(features_train, label_train)
labels_pred_DT = DT_regressor.predict(features_test)
R2_Score_DT = r2_score(labels_pred_DT, label_test)

#5. RandomForest Regression
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 50)

RF_regressor.fit(features_train, label_train)
labels_pred_RF = RF_regressor.predict(features_test)
R2_Score_RF = r2_score(labels_pred_RF, label_test)


#6. Knn Regression
from sklearn.neighbors import KNeighborsRegressor
Knn_regressor = KNeighborsRegressor()

Knn_regressor.fit(features_train, label_train)
labels_pred_Knn = Knn_regressor.predict(features_test)
R2_Score_knn = r2_score(labels_pred_RF, label_test)

#6. AdaBoost Regression
from sklearn.ensemble import AdaBoostRegressor
Adaboost_regressor = AdaBoostRegressor()

Adaboost_regressor.fit(features_train, label_train)
labels_pred_adaboost = Adaboost_regressor.predict(features_test)
R2_Score_adaboost = r2_score(labels_pred_adaboost, label_test)

#6. GradientBoosting Regression
from sklearn.ensemble import GradientBoostingRegressor
GB_regressor = GradientBoostingRegressor()
GB_regressor.fit(features_train, label_train)
labels_pred_GB = GB_regressor.predict(features_test)
R2_Score_GB = r2_score(labels_pred_GB, label_test)

#6. XGboost Regression
from xgboost import XGBRegressor
XGBRegressor = XGBRegressor()

XGBRegressor.fit(features_train, label_train)
labels_pred_XGB = XGBRegressor.predict(features_test)
R2_Score_XGBoost = r2_score(labels_pred_XGB, label_test)

#6. CatBoost Regression
from catboost import CatBoostRegressor
CatBoostRegressor = CatBoostRegressor()

CatBoostRegressor.fit(features_train, label_train)
labels_pred_CatBoost = CatBoostRegressor.predict(features_test)
R2_Score_catBoost = r2_score(labels_pred_CatBoost, label_test)

#6. Other Regressions
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,SGDRegressor
Lasso_regressor = Lasso()
Ridge_regressor = Ridge()
BayesianRidge_regressor = BayesianRidge()
ElasticNet_regressor = ElasticNet()
HuberRegressor = HuberRegressor()
SGDRegressor = SGDRegressor()

#lasso
Lasso_regressor.fit(features_train, label_train)
labels_pred_lasso = Lasso_regressor.predict(features_test)
R2_Score_lasso = r2_score(labels_pred_lasso, label_test)

#Ridge
Ridge_regressor.fit(features_train, label_train)
labels_pred_Ridge = Ridge_regressor.predict(features_test)
R2_Score_ridge = r2_score(labels_pred_Ridge, label_test)

#BayesianRidge
BayesianRidge_regressor.fit(features_train, label_train)
labels_pred_BayesianRidge = BayesianRidge_regressor.predict(features_test)
R2_Score_bayesian = r2_score(labels_pred_BayesianRidge, label_test)

#ElasticNet
ElasticNet_regressor.fit(features_train, label_train)
labels_pred_ElasticNet = ElasticNet_regressor.predict(features_test)
R2_Score_ElasticNet = r2_score(labels_pred_ElasticNet, label_test)

#HuberRegressor
HuberRegressor.fit(features_train, label_train)
labels_pred_HuberRegressor = HuberRegressor.predict(features_test)
R2_Score_huber = r2_score(labels_pred_HuberRegressor, label_test)

#SGDRegressor
SGDRegressor.fit(features_train, label_train)
labels_pred_SGDRegressor = SGDRegressor.predict(features_test)
R2_Score_sgd = r2_score(labels_pred_SGDRegressor, label_test)

dict1 = {'Algorithm' : ['linear_regressor', 'poly_regressor','SVM_regressor','DT_regressor',
'RF_regressor','Knn_regressor','Adaboost_regressor','GB_regressor','XGBRegressor',
'CatBoostRegressor','Lasso_regressor','Ridge_regressor','BayesianRidge_regressor',
'ElasticNet_regressor','HuberRegressor','SGDRegressor'],
    'R2_Score' : [R2_Score_lr,R2_Score_poly,R2_Score_svm,R2_Score_DT,R2_Score_RF,
    R2_Score_knn,R2_Score_adaboost,R2_Score_GB,R2_Score_XGBoost,
    R2_Score_catBoost,R2_Score_lasso,R2_Score_ridge,R2_Score_bayesian,R2_Score_ElasticNet,
    R2_Score_huber,R2_Score_sgd]
}

pd.DataFrame.from_dict(dict1)

