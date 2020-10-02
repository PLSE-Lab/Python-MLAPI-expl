#!/usr/bin/env python
# coding: utf-8

# ***Please UpVote if you like the work!!!***

# # Boston Dataset

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.datasets import load_boston
df = load_boston()
X = pd.DataFrame(df.data,columns=df.feature_names)


# In[ ]:


X


# In[ ]:


y = df.target


# In[ ]:


for i in X.columns:
    sns.scatterplot(i,y,data = X)
    plt.show()


# In[ ]:


plt.figure(figsize = (10,7))
sns.heatmap(X.corr(),annot = True)


# In[ ]:


import statsmodels.api as sm
Xc = sm.add_constant(X)
linreg = sm.OLS(y,Xc).fit()
linreg.summary()


# In[ ]:


Xc.drop(['INDUS','AGE'],axis = 1,inplace = True)


# In[ ]:


linreg = sm.OLS(y,Xc).fit()
linreg.summary()


# In[ ]:


linreg.rsquared


# In[ ]:


np.sqrt(linreg.mse_resid)


# ##### Giving a range estimate rather than giving a point estimate is always a more believable strategy. This can be achieved by using k-fold cross validation.

# In[ ]:


X = Xc.drop('const',axis = 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)


# In[ ]:


model.score(X,y)


# In[ ]:


y = pd.DataFrame(y,columns=['medv'])


# In[ ]:


from sklearn.model_selection import KFold
from sklearn import metrics
kf = KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([model],['Linear_Regression']):
    rmse = []
    for train_idx,test_idx in kf.split(X,y):
        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test,y_pred)
        print(np.sqrt(mse))
        rmse.append(np.sqrt(mse))
    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.var(rmse,ddof = 1), name))


# Bias error : 4.829
# 
# Variance error = 0.53924

# ##### Objective of REGULARIZATION : To avoid the model to overfit, i.e, to control/reduce the variance error at the cost of bias error. So we need to decide whether the bias error sacrifice is worth or not.

# So consider after regularization,
# 
# Bias error : 4.958
# 
# Variance error : 0.4236

# 1. Percentage of bias error the 1st model is better than the 2nd model : ((4.958 - 4.829)/4.958)*100 = 2.60%
# 2. Percentage of variance error the 2nd model is better than the 1st model : ((0.53924 - 0.4236)/0.53924)*100 = 21.44%

# So here we need to sacrifice just 2.6% of bias error to get a benifit of 21.44% variance error. So we would choose the 2nd model to be better than the 1st one.

# # Ridge Regression (L2)
# Cost function Ridge = min( MSE + alpha * {sum_i=1_to_p(beta_i ** 2)} )
# 
# p : Number of features 
# 
# alpha : range(0-1) - typically we play between 0 to 0.5
# 
# Ridge will add higher penalty to those features which contribute less to predict the dependent variable. It will scale down the coefficients of those weaker features which is creating a larger residues (features with low coefficients) comparable to those features creating a lower residues (features with high coefficients). Those features which are highly correlated to dependent variables will create less residues, whereas those features which are weakly correlated to dependent variables will create higher residues. So the Ridge will scale down the magnitude in such a way that highly correlated features, it will scale down less and the least correlated features it will scale down more.
# 
# Example : 
# 
# Before Ridge regularization :
# 
# y = 0.836 * x1 + 0.438 * x2 - 0.386 * x3 + 0.158 * x4
# 
# After Ridge regularization :
# 
# y = 0.81 * x1 + 0.399 * x2 - 0.170 * x3 + 0.023 * x4

# # Lasso Regression (L1)
# Cost function Lasso = min( MSE + alpha * { sum_i=1_to_p(abs(beta_i)) } )
# 
# p : Number of features 
# 
# alpha : range(0-1) - typically we play between 0 to 0.5
# 
# Example : 
# 
# Before Lasso regularization :
# 
# y = 0.836 * x1 + 0.438 * x2 - 0.386 * x3 + 0.158 * x4
# 
# After Lasso regularization :
# 
# y = 0.81 * x1 + 0.399 * x2 - 0.170 * x3 + 0 * x4
# 
# If we overdo lasso, it will underfit as most of the features will be eliminated. So the value of alpha should not be high.

# # ElasticNet Regression
# Cost function Lasso = min( MSE + a * { sum_i=1_to_p(abs(beta_i))} + b * {sum_i=1_to_p(beta_i ** 2)} )
# 
# l1_ratio = a/(a+b)
# 
# p : Number of features
# 
# If, l1_ratio = 1 ->Lasso
# 
# If, l1_ratio = 0 ->Ridge

# In[ ]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


m1 = LinearRegression()
m2 = Ridge(alpha=0.5,normalize=True) # Scaling is mandatory for all distance based calculations
m3 = Lasso(alpha=0.1,normalize=True)
m4 = ElasticNet(alpha=0.01,l1_ratio=0.92,normalize=True)


# In[ ]:


# from sklearn.model_selection import GridSearchCV
# params = { 'alpha' : np.arange(0.01,1,0.01) }
# #           ,'l1_ratio' : np.arange(0.1,1,0.01)}
# gscv = GridSearchCV(m3,params,cv = 5,scoring = 'neg_mean_squared_error')
# gscv.fit(X,y)
# gscv.best_params_


# In[ ]:


model = m1.fit(X_train,y_train)
sns.barplot(x = X.columns,y = sorted(model.coef_[0]))
plt.title('LR coefficients')


# In[ ]:


model = m2.fit(X_train,y_train)
sns.barplot(x = X.columns,y = sorted(model.coef_[0]))
plt.title('Ridge coefficients')


# In[ ]:


model = m3.fit(X_train,y_train)
sns.barplot(x = X.columns,y = sorted(model.coef_))
plt.title('LASSO coefficients')


# In[ ]:


model = m4.fit(X_train,y_train)
sns.barplot(x = X.columns,y = sorted(model.coef_))
plt.title('ElasticNet coefficients')


# In[ ]:


from sklearn.model_selection import KFold
from sklearn import metrics
kf = KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([m1,m2,m3,m4],['Linear_Regression','Ridge','LASSO','ElasticNet']):
    rmse = []
    for train_idx,test_idx in kf.split(X,y):
        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test,y_pred)
        rmse.append(np.sqrt(mse))
    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.var(rmse,ddof = 1), name))
    print()


# In[ ]:


print('Bias error increased after Lasso : ',(5.875-4.829)/5.875 * 100,"%")


# In[ ]:


print('Variance error decreased after Lasso : ',(0.53924 - 0.41470)/0.53924 * 100,"%")


# # Polynomial Linear Regression

# In[ ]:


mpg_df = pd.read_csv('../input/auto-mpg-pratik.csv')


# In[ ]:


mpg_df.head()


# In[ ]:


sns.pairplot(mpg_df)


# In[ ]:


X_update = mpg_df.drop(['mpg', 'cylinders', 'displacement', 'horsepower',
       'acceleration', 'car name'],axis = 1)
y = mpg_df['mpg']


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
qr = PolynomialFeatures(degree=2)
x_qr = qr.fit_transform(X_update[['weight']])
x_qr = x_qr[:,2:]
x_qr_df = pd.DataFrame(x_qr,columns=['weight_square'])


# In[ ]:


df_final = pd.concat([X_update,x_qr_df,y],axis = 1)
df_final


# In[ ]:


import statsmodels.api as sm
X = df_final.drop('mpg',axis = 1)
y = df_final['mpg']
Xc = sm.add_constant(X)
lr = sm.OLS(y,Xc).fit()
lr.summary()


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
m2 = Ridge(alpha=0.06,normalize=True) # Scaling is mandatory for all distance based calculations
m3 = Lasso(alpha=0.37,normalize=True)
m4 = ElasticNet(alpha=0.01,l1_ratio=0.1,normalize=True)


# In[ ]:


y = pd.DataFrame(y,columns=['mpg'])


# In[ ]:


from sklearn.model_selection import KFold
from sklearn import metrics
kf = KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([model,m2,m3,m4],['Quadratic_Regression','Ridge','Lasso','ElasticNet']):
    rmse = []
    for train_idx,test_idx in kf.split(X,y):
        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test,y_pred)
        rmse.append(np.sqrt(mse))
    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.std(rmse,ddof = 1), name))
    print()


# In[ ]:


print('Bias error increased after Ridge : ',(3.409-3.021)/3.409 * 100,"%")


# In[ ]:


print('Variance error decreased after Ridge : ',(0.36898 - 0.31105)/0.36898 * 100,"%")


# ***Please UpVote if you like the work!!!***
