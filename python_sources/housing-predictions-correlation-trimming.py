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
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", encoding= 'unicode_escape')
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", encoding= 'unicode_escape')
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv", encoding= 'unicode_escape')


# In[ ]:


corr_matrix = train_data.corr()
corr_matrix


# In[ ]:


corr_matrix["SalePrice"].sort_values(ascending=False)


# In[ ]:


train_data.describe()


# # Prepare Data

# In[ ]:


train_df = train_data.fillna(0)
test_df = test_data.fillna(0)


# **Outlier Detection**

# In[ ]:


### check for any possible aoutliers in the train dataset
fig, ax = plt.subplots()
ax.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'],color='green',alpha=0.3)
plt.ylabel('SalsPrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

# plot the train data again
fig, ax = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'],color='green',alpha=0.3)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# **Feature Drops**

# In[ ]:


feature_drop = ['BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']


# In[ ]:


train_df_clean = train_df.drop(feature_drop[0],axis=1)
for i in range(len(feature_drop)-1):
    train_df_clean = train_df_clean.drop(feature_drop[i+1],axis=1)
    i = i+1


# In[ ]:


test_df_clean = test_df.drop(feature_drop[0],axis=1)
for i in range(len(feature_drop)-1):
    test_df_clean = test_df_clean.drop(feature_drop[i+1],axis=1)
    i = i+1


# In[ ]:


feature = ["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope"
           ,"Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl"
           ,"Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual"
           ,"BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir"
           ,"Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual"
           ,"GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]


# In[ ]:


i=0
for i in range(len(feature)):
    train_coded_df = pd.DataFrame(train_df_clean, columns=[feature[i]])# generate binary values using get_dummies
    dum_df = pd.get_dummies(train_coded_df, columns=[feature[i]])# merge with main df bridge_df on key values
    train_df_clean = train_df_clean.join(dum_df)
    train_df_clean = train_df_clean.drop(feature[i],axis=1)
    i=i+1


# In[ ]:


i=0
for i in range(len(feature)):
    test_coded_df = pd.DataFrame(test_df_clean, columns=[feature[i]])# generate binary values using get_dummies
    dum_df = pd.get_dummies(test_coded_df, columns=[feature[i]])# merge with main df bridge_df on key values
    test_df_clean = test_df_clean.join(dum_df)
    test_df_clean = test_df_clean.drop(feature[i],axis=1)
    i=i+1


# In[ ]:


feature_drop_train = ['Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','Electrical_0','Electrical_Mix','Exterior1st_ImStucc'
                      ,'Exterior1st_Stone','Exterior2nd_Other','GarageQual_Ex','Heating_Floor','Heating_OthW','HouseStyle_2.5Fin'
                      ,'MiscFeature_TenC','PoolQC_Fa','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Utilities_NoSeWa','Id']


# In[ ]:


train_df_clean = train_df_clean.drop(feature_drop_train[0],axis=1)
for i in range(len(feature_drop_train)-1):
    train_df_clean = train_df_clean.drop(feature_drop_train[i+1],axis=1)
    i = i+1


# In[ ]:


feature_drop_test = ['Exterior1st_0','Exterior2nd_0','Functional_0','KitchenQual_0','MSZoning_0','SaleType_0','Utilities_0','Id']


# In[ ]:


test_df_clean = test_df_clean.drop(feature_drop_test[0],axis=1)
for i in range(len(feature_drop_test)-1):
    test_df_clean = test_df_clean.drop(feature_drop_test[i+1],axis=1)
    i = i+1


# In[ ]:


train_y = pd.DataFrame(train_df_clean['SalePrice'])
train_df_clean = train_df_clean.drop('SalePrice',axis=1)
train_X = train_df_clean
test_X = test_df_clean


# In[ ]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)


# # Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)


# # ML Modeling

# **Lasso Regressor**

# In[ ]:


from sklearn import linear_model
clflasso = linear_model.Lasso(alpha=0.1)
clflasso.fit(X_train, np.ravel(y_train))
y_pred_Lasso = clflasso.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_Lasso, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, y_pred_Lasso))))
plt.show()


# **HuberRegressor**

# In[ ]:


#from sklearn.linear_model import HuberRegressor
#reghub = HuberRegressor()
#reghub.fit(X_train, np.ravel(y_train))
#y_pred_Hub = reghub.predict(X_test)


# In[ ]:


#plt.figure(figsize=(5,4))
#sns.distplot(y_test, label='test')
#sns.distplot(y_pred_Hub, hist_kws={'alpha':0.3}, label='prediction')
#plt.legend()
#plt.title((np.sqrt(mean_squared_error(y_test, y_pred_Hub))))
#plt.show()


# **Linear Regression**

# In[ ]:


regl = LinearRegression()
regl.fit(X_train, np.ravel(y_train))
y_pred_Lin = regl.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_Lin, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, y_pred_Lin))))
plt.show()


# **Decision Tree Regressor**

# In[ ]:


regd = DecisionTreeRegressor()
regd.fit(X_train, np.ravel(y_train))
y_pred_D = regd.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_D, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, y_pred_D))))
plt.show()


# **Random Forest Regressor**

# In[ ]:


regr = RandomForestRegressor()
regr.fit(X_train, np.ravel(y_train))
y_pred_RF = regr.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_RF, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, y_pred_RF))))
plt.show()


# **KNN Regressor**

# In[ ]:


regknn = KNeighborsRegressor()
regknn.fit(X_train, np.ravel(y_train))
y_pred_KNN = regknn.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_KNN, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, y_pred_KNN))))
plt.show()


# **Training Model Predictions Averaged**

# In[ ]:


y_pred_avg = (y_pred_Lin + y_pred_RF)/2

plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(y_pred_avg, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title(np.sqrt(mean_squared_error(y_test, y_pred_avg)))
plt.show()


# **Model Comparisons**

# In[ ]:


def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[ ]:


scores = cross_val_score(regl,y_test,y_pred_Lin,scoring="neg_mean_squared_error",cv=10)
Lin_rmse_scores = np.sqrt(-scores)

scores = cross_val_score(regd,y_test,y_pred_D,scoring="neg_mean_squared_error",cv=10)
DR_rmse_scores = np.sqrt(-scores)

scores = cross_val_score(regr,y_test,y_pred_RF,scoring="neg_mean_squared_error",cv=10)
RF_rmse_scores = np.sqrt(-scores)

scores = cross_val_score(regknn,y_test,y_pred_KNN,scoring="neg_mean_squared_error",cv=10)
KNN_rmse_scores = np.sqrt(-scores)

#scores = cross_val_score(reghub,y_test,y_pred_Hub,scoring="neg_mean_squared_error",cv=10)
#Huber_rmse_scores = np.sqrt(-scores)

scores = cross_val_score(clflasso,y_test,y_pred_Lasso,scoring="neg_mean_squared_error",cv=10)
Lasso_rmse_scores = np.sqrt(-scores)


# In[ ]:


print('Linear Regression Scores')
display_scores(Lin_rmse_scores)


# In[ ]:


print('Decision Tree Regressor Scores')
display_scores(DR_rmse_scores)


# In[ ]:


print('Random Forest Regressor Scores')
display_scores(RF_rmse_scores)


# In[ ]:


print('KNN Regressor Scores')
display_scores(KNN_rmse_scores)


# In[ ]:


#print('Huber Regressor Scores')
#display_scores(Huber_rmse_scores)


# In[ ]:


print('Lasso Regressor Scores')
display_scores(Lasso_rmse_scores)


# # Boosting

# **Ada Booster**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

ada_clf = AdaBoostRegressor(
RandomForestRegressor())
boost = ada_clf.fit(X_train, np.ravel(y_train))


# In[ ]:


best_fit_pred_boost = boost.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(best_fit_pred_boost, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, best_fit_pred_boost))))
plt.show()


# In[ ]:


scores = cross_val_score(ada_clf,y_test,best_fit_pred_boost,scoring="neg_mean_squared_error",cv=10)
boost_rmse_scores = np.sqrt(-scores)


# In[ ]:


display_scores(boost_rmse_scores)


# **Gradient Booster**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor()
grad_boost = gbrt.fit(X_train, np.ravel(y_train))


# In[ ]:


best_fit_pred_grad_boost = grad_boost.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(best_fit_pred_grad_boost, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, best_fit_pred_grad_boost))))
plt.show()


# In[ ]:


scores = cross_val_score(ada_clf,y_test,best_fit_pred_grad_boost,scoring="neg_mean_squared_error",cv=10)
grad_boost_rmse_scores = np.sqrt(-scores)


# In[ ]:


display_scores(grad_boost_rmse_scores)


# In[ ]:


gbrt_ensm = GradientBoostingRegressor(n_estimators=3000)
gbrt_ensm.fit(X_train, np.ravel(y_train))

errors = [mean_squared_error(y_test,best_fit_pred_grad_boost)
         for best_fit_pred_grad_boost in gbrt.staged_predict(X_test)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(n_estimators = bst_n_estimators)
gbrt_best.fit(X_train, np.ravel(y_train))


# In[ ]:


best_fit_pred_grad_boost = gbrt_best.predict(X_test)


# In[ ]:


plt.figure(figsize=(5,4))
sns.distplot(y_test, label='test')
sns.distplot(best_fit_pred_grad_boost, hist_kws={'alpha':0.3}, label='prediction')
plt.legend()
plt.title((np.sqrt(mean_squared_error(y_test, best_fit_pred_grad_boost))))
plt.show()


# # Grid Search

# In[ ]:


#param_grid = [
#    {'n_estimators': [3,10,30,40,50,60,70,80,90,100], 'max_features': [2,4,6,8,10,12]},
#    {'bootstrap':[False],'n_estimators': [3,10,30,40,50,60,70,80,90,100], 'max_features': [2,3,4]},
#]
#
#forest_reg = RandomForestRegressor()
#
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
#
#grid_search.fit(X_train, np.ravel(y_train))
#
#grid_search.best_estimator_


# In[ ]:


#final_model = grid_search.best_estimator_


# # Optimization Evaluation

# In[ ]:


#final_model.fit(X_train, np.ravel(y_train))
#final_pred = regr.predict(X_test)

#plt.figure(figsize=(5,4))
#sns.distplot(y_test, label='test')
#sns.distplot(final_pred, hist_kws={'alpha':0.3}, label='prediction')
#plt.legend()
#plt.title((np.sqrt(mean_squared_error(y_test, final_pred))))
#plt.show()


# # Submission

# In[ ]:


grad_boost = gbrt_best.fit(train_X, np.ravel(train_y))
best_fit_pred_grad_boost_submission = gbrt_best.predict(test_X)


# In[ ]:


submission = sample_submission.join(pd.DataFrame(best_fit_pred_grad_boost_submission))
submission = submission.drop('SalePrice',axis=1)
submission = submission.rename(columns={0:'SalePrice'})


submission.to_csv('submission.csv',index=False)

