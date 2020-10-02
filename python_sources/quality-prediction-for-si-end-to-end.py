#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.listdir('../input')


# In[ ]:


df=pd.read_csv('../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv',decimal=',').drop_duplicates()
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# # Date in Mining Ores

# In[ ]:


df.groupby(['date'])


# In[ ]:


plt.figure(figsize=(30,10))
df.groupby(['date']).mean()['% Silica Concentrate'].plot()
plt.show()


# In[ ]:


plt.figure(figsize=(30,10))
df.groupby(['date']).mean()['% Iron Concentrate'].plot()
plt.show()


# In[ ]:


df.groupby(['% Silica Concentrate']).mean()


# In[ ]:


df.groupby(['% Iron Concentrate']).mean()


# Since dataset is very huge .Considering to split the data into train and test 

# In[ ]:


# deleting date column


# In[ ]:


df=df.drop(['date'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train,test=train_test_split(df,test_size=0.3)


# In[ ]:


print('Train size',train.shape)
print('Test size',test.shape)


# In[ ]:





# In[ ]:


import missingno
missingno.matrix(train,figsize=(20,5))


# In[ ]:


import missingno
missingno.matrix(test,figsize=(20,5))


# 

# In[ ]:


plt.figure(figsize=(30,30))
sns.heatmap(train.corr(),annot=True,linewidths=0.3)
plt.show()


# In[ ]:


train.columns[-1]


# In[ ]:


train.info()


# In[ ]:


train.describe()


# ## % for Silicon Concentration

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Deleting date columns and % Iron Concentration
y=train['% Silica Concentrate']
X=train.drop(['% Silica Concentrate'],axis=1)


# In[ ]:


# Backward Elimination
cols=list(X.columns)
pmax=1
while len(cols)>0:
    p=[]
    C=X[cols]
    xc=sm.add_constant(C)
    model=sm.OLS(y,xc).fit()
    p=pd.Series(model.pvalues.values[1:],index=cols)
    pmax=max(p)
    feature_with_p_max=p.idxmax()
    if pmax>0.05:
        cols.remove(feature_with_p_max)
    else:
        break
        
selected_cols=cols
print(selected_cols)


# ## 1. Base Model

# In[ ]:


import statsmodels.api as sm
xc=sm.add_constant(X[selected_cols])
xc=xc.drop([],axis=1)
model=sm.OLS(y,xc).fit()
model.summary()


# In[ ]:


residuals=model.resid
sns.distplot(residuals)


# In[ ]:


import scipy.stats as stats
stats.probplot(residuals,plot=plt)
plt.show()


# ### Autocorrelation Check

# In[ ]:


import statsmodels.tsa.api as smt
acf=smt.graphics.plot_acf(residuals,lags=40)
acf.show()


# ### Multicollinearity Check

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
pd.DataFrame({'vif':vif},index=X.columns).T


# In[ ]:


from sklearn import metrics


# ### 2.Linear Regression -ML

# In[ ]:


y=train['% Silica Concentrate']
X=train.drop(['% Silica Concentrate'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=0.30, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_train_pred=lr.predict(X_train)
print ("intercept:",lr.intercept_)
print ("n_coefficients:         ",lr.coef_)


# In[ ]:


print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=lr.predict(X_test)


# In[ ]:


print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# ### Model using Backward Elimination Model

# ### 3.Lasso/Ridge/ElasticNet

# ## 3.1 Lasso

# In[ ]:


from sklearn.linear_model import Lasso,LassoCV
lasso=Lasso(alpha=0.001,normalize=True)


# In[ ]:


lasso.fit(X_train,y_train)


# In[ ]:


y_train_pred=lasso.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=lasso.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# In[ ]:


n_alphas = 50
alphas = np.linspace(0.1,4.5, n_alphas)
coefs=[]
lasso = Lasso()
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)
    
plt.plot(alphas, coefs)
plt.xlabel('alphas')
plt.ylabel('coefs')
plt.show()


# In[ ]:


n_alphas=50
alphas=np.linspace(0.1,1, n_alphas)

lasso_cv = LassoCV(alphas=alphas, cv=3, random_state=22)
lasso_cv.fit(X,y)


# In[ ]:


lasso_cv.alpha_


# In[ ]:


lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)
lasso.coef_
pd.DataFrame(lasso.coef_, X.columns, columns=['coefs'])


# In[ ]:


y_train_pred=lasso.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=lasso.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# ### 3.2 Ridge

# In[ ]:


from sklearn.linear_model import Ridge,RidgeCV
ridge=Ridge(alpha=0.05)


# In[ ]:


ridge.fit(X_train,y_train)


# In[ ]:


y_train_pred=ridge.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=ridge.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# In[ ]:


ridge.fit(X, y)
ridge.coef_
pd.DataFrame(ridge.coef_, X.columns, columns=['coefs'])


# In[ ]:


n_alphas = 200
alphas = np.logspace(-3, 2, n_alphas)
coefs=[]
model = Ridge()
for a in alphas:
    model.set_params(alpha=a)
    model.fit(X, y)
    coefs.append(model.coef_)
    
plt.plot(alphas, coefs)
plt.xlabel('alphas')
plt.ylabel('coefs')
plt.show()


# In[ ]:


n_alphas = 1000
alphas = np.logspace(-2, 0)

ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X, y)

ridge_cv.alpha_


# In[ ]:


ridge=Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train,y_train)


# In[ ]:


y_train_pred=ridge.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=ridge.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# ### 3.3 Elastic Net/Elastic Net CV

# In[ ]:


from sklearn.linear_model import ElasticNet, ElasticNetCV
enet = ElasticNet(alpha=0.1)
enet.fit(X_train, y_train)


# In[ ]:


y_train_pred=enet.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=enet.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# In[ ]:


pd.DataFrame(enet.coef_, X.columns, columns=['coefs'])


# In[ ]:


n_alphas = 100
alphas = np.logspace(-3, -1, n_alphas)
coefs=[]
enet = ElasticNet()
for a in alphas:
    enet.set_params(alpha=a)
    enet.fit(X, y)
    coefs.append(model.coef_)
    
plt.plot(alphas, coefs)
plt.xlabel('alphas')
plt.ylabel('coefs')
plt.show()


# In[ ]:


n_alphas = 100
alphas = np.logspace(-3, 1, n_alphas)

en_cv = ElasticNetCV(alphas=alphas, cv=3)
en_cv.fit(X, y)
en_cv.alpha_


# In[ ]:


enet = ElasticNet(alpha=en_cv.alpha_)
enet.fit(X_train,y_train)


# In[ ]:


y_train_pred=enet.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_train_pred=enet.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# ## 4. Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# ### Changing max depth

# In[ ]:


train_accuracy=[]
test_accuracy=[]

for depth in range(5,20):
    dt_model=DecisionTreeRegressor(max_depth=depth,random_state=42)
    dt_model.fit(X_train,y_train)
    train_accuracy.append(dt_model.score(X_train,y_train))
    test_accuracy.append(dt_model.score(X_test,y_test))


# In[ ]:


frame=pd.DataFrame({'max_depth':range(5,20),'train_accuracy':train_accuracy,'test_accuracy':test_accuracy})
print(frame)


# In[ ]:


plt.figure(figsize=(13,6))
plt.plot(frame['max_depth'],frame['train_accuracy'],marker='o')
plt.plot(frame['max_depth'],frame['test_accuracy'],marker='o')
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy Performance')
plt.show()


# In[ ]:


dtr=DecisionTreeRegressor()


# In[ ]:


dtr.fit(X_train,y_train)


# In[ ]:


print ("feature_importances:",dtr.feature_importances_)
print ("Best params: \n        ",dtr.get_params)
print('n feature',dtr.n_features_)
print(dtr.n_outputs_)


# In[ ]:


y_train_pred=dtr.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=dtr.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# ## 5. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rfr=RandomForestRegressor()


# In[ ]:


rfr.fit(X_train,y_train)


# In[ ]:


print ("feature_importances:",rfr.feature_importances_)
print ("n_coefficients:         ",rfr.get_params)
print('n feature',rfr.n_features_)
print(rfr.n_outputs_)


# In[ ]:


y_train_pred=rfr.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=rfr.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# ## Considering Random Forest Regressor Base Model as the final model. Implementing the model in the test data set

# In[ ]:


test


# In[ ]:


expected_output=train['% Silica Concentrate']
expected_input=train.drop(['% Silica Concentrate'],axis=1)


# In[ ]:


y_pred=rfr.predict(expected_input)


# In[ ]:


print('R2 of Output: ', metrics.r2_score(expected_output,y_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(expected_output,y_pred))
print('Mean square Error: ',metrics.mean_squared_error(expected_output,y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(expected_output,y_pred)))


# ## Gradient Boosting

# In[ ]:


params = {
    'n_estimators': 80,
    'max_depth': 12,
    'learning_rate': 0.1,
    'criterion': 'mse'
    }


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(**params)


# In[ ]:


gbr.fit(X_train,y_train)


# In[ ]:


y_train_pred=gbr.predict(X_train)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))


# In[ ]:


y_test_pred=gbr.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))
print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))


# TEST DATASET

# In[ ]:


expected_output=train['% Silica Concentrate']
expected_input=train.drop(['% Silica Concentrate'],axis=1)


# In[ ]:


y_pred=gbr.predict(expected_input)


# In[ ]:


print('R2 of Output: ', metrics.r2_score(expected_output,y_pred))
print('Mean absolute Error: ',metrics.mean_absolute_error(expected_output,y_pred))
print('Mean square Error: ',metrics.mean_squared_error(expected_output,y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(expected_output,y_pred)))


# In[ ]:




