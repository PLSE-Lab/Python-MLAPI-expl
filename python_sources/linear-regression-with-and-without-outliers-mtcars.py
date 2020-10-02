#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# A data frame with 32 observations on 11 (numeric) variables.
# 
# * [, 1]	mpg	Miles/(US) gallon
# * [, 2]	cyl	Number of cylinders
# * [, 3]	disp	Displacement (cu.in.)
# * [, 4]	hp	Gross horsepower
# * [, 5]	drat	Rear axle ratio
# * [, 6]	wt	Weight (1000 lbs)
# * [, 7]	qsec	1/4 mile time
# * [, 8]	vs	Engine (0 = V-shaped, 1 = straight)
# * [, 9]	am	Transmission (0 = automatic, 1 = manual)
# * [,10]	gear	Number of forward gears
# * [,11]	carb	Number of carburetors

# In[ ]:


mtcars=pd.read_csv('../input/linear-regression-eda-python/mtcars.csv')
mtcars.head()


# In[ ]:


mtcar1=mtcars
mtcar1.head()


# In[ ]:


mtcar1.info()


# In[ ]:


mtcar1.nunique()


# In[ ]:


mtcar1.cyl.unique()


# In[ ]:


mtcar1.vs.unique()


# In[ ]:


mtcar1.am.unique()


# In[ ]:


mtcar1.gear.unique()


# In[ ]:


mtcar1.carb.unique()


# In[ ]:


mtcar1.describe()


# In[ ]:


plt.boxplot(mtcar1.mpg)
plt.show()


# In[ ]:


plt.boxplot(mtcar1.disp)
plt.show()


# In[ ]:


plt.boxplot(mtcar1.hp)
plt.show()


# In[ ]:


plt.boxplot(mtcar1.drat)
plt.show()


# In[ ]:


plt.boxplot(mtcar1.wt)
plt.show()


# In[ ]:


plt.boxplot(mtcar1.qsec)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
mtcar1.vs.value_counts().plot(kind='bar',ax=ax[0])
mtcar1.am.value_counts().plot(kind='bar',ax=ax[1])
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
mtcar1.cyl.value_counts().plot(kind='bar',ax=ax[0])
mtcar1.gear.value_counts().plot(kind='bar',ax=ax[1])
plt.show()


# In[ ]:


sns.countplot(mtcar1.gear,hue=mtcar1.cyl)
plt.show()


# * There are no missing values
# * Total 32 observations are present
# * There are 5 categorical variables
# * mpg,hp,wt and qsec has outliers

# In[ ]:


mtcar2=mtcar1.drop(['cyl','vs','am','gear','carb'],axis=1)
mtcar2.head()


# In[ ]:


mtcar1.groupby(['cyl'])['carb'].value_counts()[8].plot(kind='bar')
plt.show()


# In[ ]:


mtcar1.groupby(['vs','am'])['gear'].value_counts()[1][1].plot(kind='bar')
plt.show()


# In[ ]:


mtcar1.groupby(['vs','am'])['gear'].value_counts()[0][1].plot(kind='bar')
plt.show()


# In[ ]:


mtcar1.groupby(['vs','am'])['carb'].value_counts()[1][1].plot(kind='bar')
plt.show()


# In[ ]:


mtcar1.groupby(['vs','am'])['carb'].value_counts()[1][0].plot(kind='bar')
plt.show()


# In[ ]:


sns.pairplot(mtcar2)
plt.show()


# ### Assumption 1: No autocorrelation

# In[ ]:


X=mtcar1.drop(['mpg','model'],axis=1)
Y=mtcar1.mpg
import statsmodels.api as sm
X_constant = sm.add_constant(X)
model = sm.OLS(Y,X_constant).fit()
model.summary()


# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(X,Y)
lin_reg1.score(X,Y)


# * Durbin-Watson value Its value ranges from 0-4. If the value of Durbin- Watson is between 0-2, it's known as Positive Autocorrelation between residuals.
# * If the value ranges from 2-4, it is known as Negative autocorrelation.
# * If the value is exactly 2, it means No Autocorrelation.
# * It is 1.86. Hence we can say that there is very small/neglible Positive Autocorrelation between residuals

# In[ ]:


import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(model.resid, alpha=0.05) # ACF is auto correlation function
acf.show()


# ### Assumption 2- Normality of Residuals

# The higher the value of Jarque Bera test , the lesser the residuals are normally distributed.
# We generally prefer a lower value of jarque bera test.

# * Null Hypothesis - Error terms are normally distributed.
# * Alternate Hypothesis - Error terms are NOT normally distributed.

# In[ ]:


from scipy.stats import jarque_bera
name=['ch-stat','p-value']
values=jarque_bera(model.resid)
from statsmodels.compat import lzip
jb=lzip(name,values)
print(jb)


# * The critical chi square value at the 5% level of significance is 5.99. If the computed value is below this value the null hypothesis is not rejected.
# 
# * In this case the computed value of the JB statistic 1.74 is lesser than 5.99. Thus we fail to reject the null hypothesis and conclude that the error terms are normally distributed.

# In[ ]:


sns.distplot(model.resid)
plt.show()


# ### Assumption 3 - Linearity of Residuals

# In[ ]:


mean_res=model.resid.mean()
print('Mean of residuals is %.6f'%mean_res)

To detect nonlinearity one can inspect plots of observed vs. predicted values or residuals vs. predicted values. 
The desired outcome is that points are symmetrically distributed around a diagonal line in the former plot or 
around horizontal line in the latter one. 


# In[ ]:


y_pre=model.predict(X_constant)
f,ax=plt.subplots(1,2,figsize=(10,8))
sns.regplot(Y,y_pre,ax=ax[0])
sns.regplot(model.resid,y_pre,ax=ax[1])
plt.show()


# #### Rainbow Test

# * The Null hypothesis is that the regression is correctly modelled as linear.
# * The alternative for which the power might be large are convex, check
# 

# In[ ]:


test = sm.stats.diagnostic.linear_rainbow(res=model)
print(test)


# * In both cases good linearity of residuals can be seen
# * Also mean of residuals is 0.
# * Also p value for null hypothesis is more than 0.05. So we fail to reject null hypothesis. So we conclude regression is correctly modelled as linear
# * We conclude that residuals are linear

# ### Assumption 4 -	Homoscedasticity test

# This test is based on the hytpothesis testing where null and alternate hypothesis are:
# * H0 = constant variance among residuals. (Homoscedacity)
# * Ha = Heteroscedacity.
# 
# The residuals should be homoscedacious.
# 

# In[ ]:


name = ['F statistic', 'p-value']
import statsmodels.stats.api as sms
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)


# In[ ]:


sns.set_style('whitegrid')
sns.residplot(y_pre,model.resid,lowess=True,color = 'g')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual vs Predicted')
plt.show()


# * p value is less than 0.05. So we reject null hypothesis
# * Also plot for Residual vs Predicted show that points are scattered unequally to some extent
# * We conclude there is some heteroscedasticity

# ### Assumption 5- multicollinearity

# In[ ]:


sns.heatmap(mtcar1.corr(),annot=True)
plt.show()


# ### Observations

# * mpg has strong correlation with disp,hp and wt. All are negatively correlated to mpg
# * mpg also has moderate correlation with drat
# * disp has strong negative correlation with drat
# * hp has strong negative correlation with qsec
# * wt has strong negative correlation with drat
# * multicollinearity is present in our dataset

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
col=X_constant.shape[1]
vif=[variance_inflation_factor( X_constant.values,i) for i in range(col)]
vif_pd=pd.DataFrame({'vif':vif[1:]},index=X.columns).T
vif_pd


# In[ ]:


X1=mtcar1.drop(['cyl','disp','hp','wt','qsec','gear','model','mpg','carb'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.3,random_state=1)
X1.head()


# In[ ]:


X_constant1 = sm.add_constant(x_train)
model1 = sm.OLS(y_train,X_constant1).fit()
model1.summary()


# In[ ]:


from sklearn import metrics
x_cont_train = sm.add_constant(x_train)
x_cont_test = sm.add_constant(x_test)
y_tr_pred=model1.predict(x_cont_train)
y_tst_pred=model1.predict(x_cont_test)
print('R2 for train:',metrics.r2_score(y_train,y_tr_pred))
print('R2 for test:',metrics.r2_score(y_test,y_tst_pred))


# In[ ]:


lin_reg2=LinearRegression()
lin_reg2.fit(x_train,y_train)
print('R2 for train:',lin_reg2.score(x_train,y_train))
print('R2 for test:',lin_reg2.score(x_test,y_test))


# * The difference between train score and test score is more. Here we have problem of overfit

# ### Removing Outliers

# In[ ]:


q1=mtcar1.mpg.quantile(0.25)
q3=mtcar1.mpg.quantile(0.75)
iqr=q3-q1
ll=q1-1.5*iqr
ul=q3+1.5*iqr
print(mtcar1.shape)
mtcar3 = mtcar1[~((mtcar1['mpg']<ll) | (mtcar1['mpg']>ul))]
print(mtcar3.shape)


# In[ ]:


X_wo=mtcar3.drop(['mpg','model'],axis=1)
Y_wo=mtcar3['mpg'].values
X_const_wo=sm.add_constant(X_wo)
model_wo=sm.OLS(Y_wo,X_const_wo).fit()
model_wo.summary()


# In[ ]:


name = ['F statistic', 'p-value']
import statsmodels.stats.api as sms
test_wo = sms.het_goldfeldquandt(model_wo.resid, model_wo.model.exog)
lzip(name, test_wo)


# * The goldfeld test show that it is still heteroscedasticity

# In[ ]:


vif1=[variance_inflation_factor( X_const_wo.values,i) for i in range(X_const_wo.shape[1])]
vif_pd1=pd.DataFrame({'vif':vif1[1:]},index=X_wo.columns).T
vif_pd1


# In[ ]:


lin_reg_wo=LinearRegression()
X_wo1=X_wo[['drat','vs','am']]

x_train1,x_test1,y_train1,y_test1=train_test_split(X_wo1,Y_wo,test_size=0.3,random_state=2)
lin_reg_wo.fit(x_train1,y_train1)

print(lin_reg_wo.score(x_train1,y_train1))
print(lin_reg_wo.score(x_test1,y_test1))


# * After removing outliers and selecting non collinear variables, we can see that out R^2 is close for training and test data set

# ### -----------------------------------------------------------------------END-------------------------------------------------------------------

# In[ ]:




