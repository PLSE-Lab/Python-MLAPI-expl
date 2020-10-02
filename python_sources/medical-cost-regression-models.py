#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/insurance/insurance.csv")
df = data.copy()


# # General Information about the data
# 
# - There are 7 features/variables in the data namely:
# 
#     - age: age of primary beneficiary
#     - sex: insurance contractor gender (female/male)
#     - bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
#     - children: Number of children covered by health insurance (Number of dependents)
#     - smoker: Smoking (whether the beneficiary is a smoker or not)
#     - region: the beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
#     - charges: Individual medical costs billed by health insurance
# - The age, bmi index, children and charges are ratio variables.
# - The sex, smoker and region are nominal categorical variables.
#     
# # Purpose of the study
# 
# - The data give information about the profile of the medical insurance beneficiaries and their charged medical costs. The final aim is to predict potential medical costs of a beneficiary.
# 
# - I have already done EDA and data visualization in another Kaggle notebook. 
# https://www.kaggle.com/brooksrattigan/medical-cost-eda-data-visualization
# 
# - The dependent variable is medical costs of the beneficiaries and the rest of the variables will be used as independent variables.
# 
# - The dependent variable is a continous ratio variable. In this respect, the aim of this study is to conduct different regression models. For this purpose;
#     - The outliers will be handled. 
#     - The categorical variables will be transformed so that the categories will have numerical values.
#     - The selected regression models will be used and the RMSE and R2 scores of each model will be compared.

# In[ ]:


display(df.head())
display(df.tail())


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


sns.swarmplot(x="sex", y="charges", data=df)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="charges", data=df)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="bmi", data=df)
plt.title('Bmi by Smoking', color = 'blue', fontsize=15)
plt.show()


# In[ ]:


sns.boxplot(x="sex", y="charges",hue="smoker", data=df)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="smoker", y="bmi", data=df)
plt.title('Bmi by Smoking', color = 'blue', fontsize=15)
plt.show()


# # Data preprocessing

# ## Converting categorical variables

# In[ ]:


df['sex'] = [1 if each=='male' else 0 for each in df.sex]
df['smoker']=[1 if each=='yes' else 0 for each in df.smoker]
df = pd.get_dummies(df, columns = ["region"], prefix = ["region"], drop_first=False)
df.head()


# 
# ## Outliers
# - As it can be seen from the general statistical information of the data set that min & max values of charges and bmi, the standard deviation of charges and the difference between the mean and media of the charges give an impression that there may be some outliers in the data set. The swarm plots and box plots show also that there are some outliers. 
# - I will use Local Outlier Factor method to detect the outliers and replace them with an observation which is selected as the threshold.

# In[ ]:


from sklearn.neighbors import LocalOutlierFactor


# In[ ]:


clf = LocalOutlierFactor()
clf.fit_predict(df)


# In[ ]:


df_scores = clf.negative_outlier_factor_


# In[ ]:


# When we sort the outlier scores, it is seen that the top 5 values seem to outstand compared to the rest.
# In this framework, the 6th observation is selected as the threshold and the outliers will be replaced by this threshold.
np.sort(df_scores)[:20]


# In[ ]:


# The threshold score is set as a filter.
threshold_score = np.sort(df_scores)[5]
df[df_scores == threshold_score]
outlier_tf = df_scores < threshold_score


# In[ ]:


# the threshold observation
threshold_observation = df[df_scores == threshold_score]
threshold_observation


# In[ ]:


outliers = df[outlier_tf]
outliers


# In[ ]:


# We convert these into an array to handle the outliers with to_records() method.
res = outliers.to_records(index = False)
res


# In[ ]:


# We replace the values of outlier observations with the values of the threshold observation. 
res[:] = threshold_observation.to_records(index = False)
df[outlier_tf] = pd.DataFrame(res, index = df[outlier_tf].index)
df[outlier_tf]


# In[ ]:


df.head()


# In[ ]:


import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict 
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ## Train-test splitting

# In[ ]:


X = df.drop(["charges"],axis = 1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)


# ## Multilinear Regression

# In[ ]:


# Multilinear Regression model with skilearn 
lr = LinearRegression()
model1 = lr.fit(X_train, y_train)


# In[ ]:


# Coefficients
model1.coef_ 


# In[ ]:


# Intercept
model1.intercept_


# In[ ]:


# R2 score of test data
model1.score(X_test,y_test)


# In[ ]:


# RMSE score of test data
rmse = np.sqrt(mean_squared_error(y_test, model1.predict(X_test)))
rmse


# In[ ]:


y_pred = model1.predict(X_test)
y_test_ =np.array(range(0,len(y_test)))
plt.plot(y_test_,y_test,color="r")
plt.plot(y_test_,y_pred,color="blue")
plt.show()


# ### Multilinear Regression CV

# In[ ]:


# R2 average of test data after cross validation
mlin_final_r2 = cross_val_score(model1, X_train, y_train, cv = 10, scoring = "r2").mean()
mlin_final_r2


# In[ ]:


# RMSE average score of test data after cross validation
mlin_final_rmse = np.sqrt(-cross_val_score(model1, 
                X_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
mlin_final_rmse


# ## PLS Regression

# In[ ]:


from sklearn.cross_decomposition import PLSRegression, PLSSVD


# In[ ]:


pls_model = PLSRegression().fit(X_train, y_train)


# In[ ]:


# PLS model coefficients
pls_model.coef_


# In[ ]:


# PLS model predictions based on train data
y_pred = pls_model.predict(X_train)


# In[ ]:


# PLS RMSE score for train data
np.sqrt(mean_squared_error(y_train, y_pred))


# In[ ]:


# PLS R2 for train data
r2_score(y_train, y_pred)


# In[ ]:


# PLS prediction based on test data
y_pred = pls_model.predict(X_test)


# In[ ]:


# PLS RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# PLS R2 for test data
r2_score(y_test, y_pred)


# ### PLS Model Tuning

# In[ ]:


# Illustraion of change in RMSE score as the model adds one additional component to the model in each loop.
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('Components and RMSE');


# In[ ]:


# PLS model with two components
pls_model2 = PLSRegression(n_components = 2).fit(X_train, y_train)


# In[ ]:


# PLS prediction based on test data after cross validation
y_pred2 = pls_model2.predict(X_test)


# In[ ]:


# PLS RMSE test score after cross validation
pls_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
pls_final_rmse


# In[ ]:


# PLS R2 test score after cross validation
pls_final_r2 = r2_score(y_test, y_pred2)
pls_final_r2


# ## Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)
ridge_model


# In[ ]:


ridge_model.coef_


# In[ ]:


# Illustration of how weights of independent variables approaches to 0 as the alpha value increases. 

lambdas = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
coefficients = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train) 
    coefficients.append(ridge_model.coef_)
        
ax = plt.gca()
ax.plot(lambdas, coefficients) 
ax.set_xscale('log') 

plt.xlabel('Lambda(Alpha) Values')
plt.ylabel('Coefficients')
plt.title('Ridge Coefficients');


# In[ ]:


# Ridge prediction based on test data
y_pred = ridge_model.predict(X_test)


# In[ ]:


# Ridge RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# Ridge R2 
r2_score(y_test, y_pred)


# ### Ridge Regression CV

# In[ ]:


from sklearn.linear_model import RidgeCV


# In[ ]:


# Ridge instantiation of cross validation model and model details
ridge_cv = RidgeCV(alphas = lambdas, 
                   scoring = "neg_mean_squared_error",
                   normalize = True)
ridge_cv.fit(X_train, y_train)
ridge_model


# In[ ]:


# Ridge cross validation alpha score
ridge_cv.alpha_


# In[ ]:


# Ridge tuned model after cross validation
ridge_tuned = Ridge(alpha = ridge_cv.alpha_, 
                   normalize = True).fit(X_train,y_train)


# In[ ]:


# Ridge model coefficients after cross validation
ridge_tuned.coef_


# In[ ]:


# Ridge RMSE test score after cross validation
ridge_final_rmse = np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test)))
ridge_final_rmse


# In[ ]:


# Ridge R2 after cross validation
ridge_final_r2 = r2_score(y_test, ridge_tuned.predict(X_test))
ridge_final_r2


# ## Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso_model = Lasso(alpha = 1.0).fit(X_train, y_train)
lasso_model


# In[ ]:


# Lasso model coefficients
lasso_model.coef_


# In[ ]:


# The weight of independent variables comes to value of zero as the alpha score changes. 

lasso = Lasso()
lambdas = 10**np.linspace(10,-2,100)*0.5 
coefficients = []

for i in lambdas:
    lasso.set_params(alpha=i)
    lasso.fit(X_train, y_train)
    coefficients.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(lambdas*2, coefficients)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[ ]:


# Lasso model prediction based on test data
y_pred = lasso_model.predict(X_test)


# In[ ]:


# Lasso RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# Lasso R2
r2_score(y_test, y_pred)


# ### Lasso Regression CV

# In[ ]:


from sklearn.linear_model import LassoCV


# In[ ]:


lasso_cv_model = LassoCV(alphas = None, 
                         cv = 10, 
                         max_iter = 10000, 
                         normalize = True)


# In[ ]:


# Lasso cross validation model details
lasso_cv_model.fit(X_train,y_train)


# In[ ]:


# Lasso cross validation model alpha score
lasso_cv_model.alpha_


# In[ ]:


# Lasso tuned model after cross validation
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(X_train, y_train)


# In[ ]:


# Lasso predictions of tuned model base on test data
y_pred = lasso_tuned.predict(X_test)


# In[ ]:


# Lasso model coefficients after cross validation
lasso_tuned.coef_


# In[ ]:


# Lasso RMSE test score after cross validation
lasso_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_final_rmse


# In[ ]:


# Lasso R2 after cross validation
lasso_final_r2 = r2_score(y_test, y_pred)
lasso_final_r2


# In[ ]:


print(f"""Multilinear Regression RMSE: {mlin_final_rmse}, R2: {mlin_final_r2}
PLS Regression RMSE: {pls_final_rmse}, R2: {pls_final_r2}
Ridge Regression RMSE: {ridge_final_rmse}, R2: {ridge_final_r2}
Lasso Regression RMSE: {lasso_final_rmse}, R2: {lasso_final_r2}""")


# ### As it is seen above that different multilinear regression models have very close RMSE and R2 scores and they are really not bad. On the other hand, it would be wise to check some of the non-linear models as well since the realation between the charges, and age and bmi are not completely linear. This can be seen through the pairplots and regression lines in my study. 
# ### In this framework, I will look into some of the selected non-linear models below. 

# ## Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


# we can change the degree value for model tuning, but it is worthwhile to note that higher degree levels may lead to overfitting.
poly_features = PolynomialFeatures(degree=3)


# In[ ]:


x_train_poly = poly_features.fit_transform(X_train)


# In[ ]:


poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)


# In[ ]:


y_train_pred = poly_model.predict(x_train_poly)


# In[ ]:


# Polynomial Regression RMSE and R2 score for train data
rmse_train = np.sqrt(mean_squared_error(y_train,y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
print(rmse_train,r2_train)


# In[ ]:


y_test_pred = poly_model.predict(poly_features.fit_transform(X_test))


# In[ ]:


# Polynomial Regression RMSE and R2 score for test data
poly_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred))
poly_r2_final = r2_score(y_test, y_test_pred)
print(poly_rmse_final,poly_r2_final)


# In[ ]:


y_test_ =np.array(range(0,len(y_test_pred)))
plt.plot(y_test_,y_test,color="r")
plt.plot(y_test_,y_test_pred,color="blue")
plt.show()


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score


# In[ ]:


rf_model = RandomForestRegressor(random_state = 42)


# In[ ]:


rf_model.fit(X_train, y_train)


# In[ ]:


rf_model.predict(X_test)[0:5]


# In[ ]:


y_pred = rf_model.predict(X_test)


# In[ ]:


# RF RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# RF R2 score for test data
r2_score(y_test, y_pred)


# ### RF Model Tuning

# In[ ]:


rf_params = {'max_depth': list(range(1,10)),
            'max_features': [2,3,5,7],
            'n_estimators' : [100, 200, 500, 1000, 1500]}


# In[ ]:


rf_model = RandomForestRegressor(random_state = 42)


# In[ ]:


rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                            n_jobs = -1,
                            verbose = 2)


# In[ ]:


rf_cv_model.fit(X_train, y_train)


# In[ ]:


rf_cv_model.best_params_


# In[ ]:


rf_tuned = RandomForestRegressor(max_depth  = 5, 
                                 max_features = 7, 
                                 n_estimators =1000)


# In[ ]:


rf_tuned.fit(X_train, y_train)


# In[ ]:


y_pred = rf_tuned.predict(X_test)


# In[ ]:


# RF RMSE test score after model tuning
rf_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
rf_rmse_final


# In[ ]:


# RF R2 for test data after model tuning
rf_r2_final = r2_score(y_test, y_pred)
rf_r2_final


# In[ ]:


# Importance level of independent variables through RF
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")


# ### As it can be seen from the graph above that smoking, bmi and age are the most important independent variables according to the RF regression results. Actually, this result is not a surprise since we can also observe these in the graphs of EDA.
# ### An alternative study can be conducted only with these 3 variables and the results can be compared with the study including all of the variables. 

# ## XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb_model = XGBRegressor().fit(X_train, y_train)


# In[ ]:


# XGBoost RMSE test score
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# XGBoost R2 for test data
r2_score(y_test, y_pred)


# ### XGBoost Model Tuning

# In[ ]:


xgb_model


# In[ ]:


xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1], 
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]
}


# In[ ]:


xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb, 
                      param_grid = xgb_grid, 
                      cv = 10, 
                      n_jobs = -1,
                      verbose = 2)


xgb_cv.fit(X_train, y_train)


# In[ ]:


xgb_cv.best_params_


# In[ ]:


xgb_tuned = XGBRegressor(colsample_bytree = 0.9, 
                         learning_rate = 0.01, 
                         max_depth = 3, 
                         n_estimators = 500) 

xgb_tuned = xgb_tuned.fit(X_train,y_train)


# In[ ]:


# XGBoost RMSE test score after model tuning
y_pred = xgb_tuned.predict(X_test)
xg_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
xg_rmse_final


# In[ ]:


# XGBoost R2 after model tuning
xg_r2_final = r2_score(y_test, y_pred)
xg_r2_final


# ## Light GBM

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train, y_train)


# In[ ]:


y_pred = lgbm_model.predict(X_test, 
                            num_iteration = lgbm_model.best_iteration_)


# In[ ]:


# LGBM RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


# LGBM R2 for test data
r2_score(y_test, y_pred)


# ### Light GBM Model Tuning

# In[ ]:


lgbm_model


# In[ ]:


lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8] }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 2)


# In[ ]:


lgbm_cv_model.fit(X_train, y_train)


# In[ ]:


lgbm_cv_model.best_params_


# In[ ]:


lgbm_tuned = LGBMRegressor(learning_rate = 0.01, 
                           max_depth = 3, 
                           n_estimators = 500,
                          colsample_bytree = 0.9)

lgbm_tuned = lgbm_tuned.fit(X_train,y_train)


# In[ ]:


y_pred = lgbm_tuned.predict(X_test)


# In[ ]:


# LGBM RMSE test score after model tuning
lgbm_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
lgbm_rmse_final


# In[ ]:


# LGBM R2 for test data after model tuning
lgbm_r2_final = r2_score(y_test, y_pred)
lgbm_r2_final


# In[ ]:


print(f"""Polynomial Regression RMSE: {poly_rmse_final}, R2: {poly_r2_final}
RF Regression RMSE: {rf_rmse_final}, R2: {rf_r2_final}
XGBoost Regression RMSE: {xg_rmse_final}, R2: {xg_r2_final}
LightGBM Regression RMSE: {lgbm_rmse_final}, R2: {lgbm_r2_final}""")


# ### As it can be seen from above that non-linear models give better RMSE and R2 scores compared to linear models. Although the selected 4 models have close scores, Light GBM has the highest R2 and lowest RMSE scores. 

# # Conclusion:
# ### In this study, I have worked on different regression models to find out the best model which is able predict the medical costs of a medical insurance beneficiary with the given information. 
# ### Multilinear regression models give fair results with respect to RMSE and R2 scores. 
# ### On the other hand, non-linear regression models give better results. In fact, EDA of the data has some indications that the relation between charges, and age and bmi is not completely linear. Among the selected 4 non-linear regression models Light GBM has given the best RMSE and R2 scores.
# ### Random Forest analysis presents us another fact that smoking, bmi and age are the top 3 features among 8 independent variables. This is not really suprising tough as we can sense that through EDA as well. 
# ### In this framework, an alternative analysis may be conducted only with these 3 independent variables and the results can be compared. 

# In[ ]:




