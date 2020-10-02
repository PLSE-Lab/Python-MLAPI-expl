#!/usr/bin/env python
# coding: utf-8

# ***Please UpVote if you like the work!!!***

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Univariate Analysis

# In[ ]:


df = pd.DataFrame({'bmi' : [4.0,5.5,6.8,7.2,7.8,9.8,9.7,8.8,11.0,13.0],
                'glucose' :[60,135,90,175,240,220,300,370,360,365]})


# In[ ]:


df


# In[ ]:


sns.scatterplot(df['bmi'],df['glucose'])
plt.plot([4,15],[300,550],color = 'r',linestyle = 'dashed')
plt.plot([4,16],[10,300],color = 'r',linestyle = 'dotted')


# glucose = beta0 + beta1 * bmi

# y -> actual points 
# 
# y' -> predicted points on the line

# residual = y - y'
# 
# The difference between actual(y) and the predicted(y') is called a residual

# For dashed line, all residuals will be negative.
# 
# For dotted line, all residuals will be positive.
# 
# SSE (sum of squared errors) = sum((residual of each point)**2)
# 
# MSE (Mean squared error) = SSE/total num of data points
# 
# RMSE (Root mean squared error) = sqrt(MSE)
# 
# Cost function = min(MSE)
# 
# Our intention/objective is to minimize our cost function. The line which gives the minimum cost is our best fit line.
# 
# beta1 = covariance of X and y divided by variance of X
# 
# beta1 = cov(X,y)/var(X)
# 
# cov(X,y) = sum_1_to_n((X-Xbar)(y-ybar))/n-1
# 
# var(X) = sum_1_to_n((X-Xbar)**2)/n-1
# 
# beta0 = ybar - beta1 * Xbar

# In[ ]:


n_bmi = len(df['bmi'])
n_glucose = len(df['glucose'])


# In[ ]:


cov_bmi_glu = np.sum((df['bmi'] - np.mean(df['bmi']))*(df['glucose'] - np.mean(df['glucose'])))/(n_bmi-1)


# In[ ]:


var_bmi = np.sum((df['bmi'] - np.mean(df['bmi']))**2)/(n_glucose-1)


# In[ ]:


beta1 = cov_bmi_glu/var_bmi


# In[ ]:


beta1


# In[ ]:


ybar = np.mean(df['glucose'])
Xbar = np.mean(df['bmi'])


# In[ ]:


beta0 = ybar - beta1 * Xbar


# In[ ]:


beta0


# In[ ]:


glu_predict = beta0 + beta1 * df['bmi']


# In[ ]:


glu_predict


# In[ ]:


sns.scatterplot(df['bmi'],df['glucose'])
sns.lineplot(df['bmi'],glu_predict,color = 'g')


# In[ ]:


sse = np.sum((df['glucose'] - glu_predict)**2)
mse = sse/df.count()[0]
rmse = np.sqrt(mse)


# In[ ]:


rmse


# On an average my model predicts the blood glucose level plus or minus 54.88 miligrams/decilitres

# ### Verification using the sklearn model

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


lr.fit(df[['bmi']],df['glucose'])


# In[ ]:


lr.coef_


# In[ ]:


lr.intercept_


# In[ ]:


lr.predict(df[['bmi']])


# For feature selection, correlation matrix will just give us how strongly the variables are correlated with each other, but the final decision whether to keep a variable or drop a variable should be taken by p-value in the ols summary.

# For Regreesion model, we calculate the measures like rmse, r-squared and adjusted rsquared.
# 
# For binary classification, we calculate accuracy,confusion matrix,classification report and ROC AUC curve.
# 
# For multiclass classification, we calculate accuracy,confusion matrix,classification report and f1-score.

# ##### We will be working on differnt models for regression and classification, but we need to choose the best model for our clients. 
# 
# ##### So for final performance validation, for regression we use RMSE.
# 
# ##### For binary classification we use AUC and for multiclass classification we use F1-SCORE

# ## R-squared
# 
# It tells how close our predicted y is from our actual y.

# r-square => 1 - ( sum((actual_y - predicted_y)** 2) / sum((actual_y - ybar)** 2) )

# #### Base line Estimator
# If predicted_y = ybar
# 
# r-square -> 0
# 
# If our data is scattered like below:
# <img src="img_scatterplot.png" style="width:200px;height;200px">
# 
# So slope(beta1) will be 0.
# 
# So predicted_y = beta0 + beta1 * X, will become predicted_y = beta0
# 
# As we know beta0 = ybar - beta1 * Xbar, so,
# ##### predicted_y = y_bar as beta1 is 0.
# 
# So rsquare = 0, using the above formula

# In[ ]:


r = np.corrcoef(df['glucose'],glu_predict)[0][1]
r


# In[ ]:


sns.scatterplot(df['glucose'],glu_predict)
plt.ylabel('predicted glucose values')
plt.xlabel('actual glucose values')


# This tells how close our predicted y is from the actual y

# In[ ]:


r_square = r**2
r_square


# In[ ]:


lr.score(df[['bmi']],df['glucose'])


# ##### So having the actual spread of X how well we can predict y is what rsquare tells us.
# ##### Higher the rsquare, better the model. Similarly, lower the RMSE, better the model.

# #### Without independent variable if our model is trying to predict something, it will end up with poor performance and the worst model.

# ## Hypothesis for Regression
# #### H0 : beta1 = 0
# #### H1 : beta1 != 0

# # Multivariate Regression

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


df_mtcars = pd.read_csv('../input/mtcars.csv')


# In[ ]:


df_mtcars.head()


# In[ ]:


df_mtcars['cyl'].value_counts()


# Consider variable 'am':
# 
# 1 - Automatic transmission
# 0 - Manual transmission
# 
# Even if one of the 2 variables in the statistical test is continous, we'll do TEST OF MEAN.
# 
# So for 'am', we'll do a 2 sample independent t-test for automatic vs manual transmission to check if there is significant difference in 'mpg' wrt auto vs manual transmission.

# In[ ]:


df_mtcars.corr()


# In[ ]:


df_mtcars.columns


# In[ ]:


model = ols('mpg~cyl+disp+hp+drat+wt+qsec+vs+am+gear+carb',df_mtcars).fit()


# In[ ]:


model.params


# ## For multivariate the formulas for betas differ.
# 
# (beta1) cyl   =>       -0.111440
# 
# (beta2) disp   =>       0.013335
# 
# (beta3) hp    =>       -0.021482
# 
# (beta4) drat    =>      0.787111
# 
# (beta5) wt    =>       -3.715304
# 
# (beta6) qsec    =>      0.821041
# 
# (beta7) vs    =>        0.317763
# 
# (beta8) am   =>         2.520227
# 
# (beta9) gear   =>       0.655413
# 
# (beta10)carb   =>      -0.199419
# 
# Intercept(beta0) = mean(mpg) - beta1 * mean(cyl) - beta2 * mean(disp) - beta3 * mean(hp) - beta4 * mean(drat) - beta5 * mean(wt) - beta6 * mean(qsec) - beta7 * mean(vs) - beta8 * mean(am) - beta9 * mean(gear) - beta10 * mean(carb)

# In[ ]:


model.summary()


# None of the variables are significant. So we will consider different combinations of variables and try to find the best possible features using rmse score.

# In[ ]:


model = ols('mpg~hp+wt',df_mtcars).fit()


# In[ ]:


model.params


# In[ ]:


model.summary()


# In[ ]:


mpg_pred = model.predict(df_mtcars[['hp','wt']])


# In[ ]:


rmse = np.sqrt(np.sum(((df_mtcars['mpg'] - mpg_pred)**2))/len(df_mtcars['mpg']))
rmse


# In[ ]:


import statsmodels.api as sm
sm.stats.diagnostic.linear_rainbow(model)


# In[ ]:


import statsmodels.stats.api as smi
smi.het_goldfeldquandt(model.resid,model.model.exog)


# ### Observations on bivariate linear regression model
# 1. For a perfect linear reg model, there should not be any autocorrelation effect. Durbin Watson score = 2 (No Autocorrelation effect). But, our model has a slight positive autocorrelation effect, which inmplies there is certain redundency in the data(multicollinear effect).
# 2. pvalue of JB score fails to reject the H0, which implies residuals are normally distributed.
# 3. Check the scatterplot of y and y_pred. The pattern should be linear. Statistically use rainbow test. The rainbow test confirms that our model is linear.
# 4. Check Goldfeld Quantile Distribution test. Our model shows that the pvalue is very less than 0.05 so we reject the null hypothesis that the data is homoskedstic

# ##### To check whether 'cyl' feature affects the 'mpg' or not...

# In[ ]:


df_mtcars['cyl'].value_counts()


# In[ ]:


mpg_cyl_4 = df_mtcars[df_mtcars['cyl'] == 4]['mpg']
mpg_cyl_8 = df_mtcars[df_mtcars['cyl'] == 8]['mpg']
mpg_cyl_6 = df_mtcars[df_mtcars['cyl'] == 6]['mpg']


# In[ ]:


from scipy.stats import f_oneway
f_oneway(mpg_cyl_4,mpg_cyl_6,mpg_cyl_8)


# ***Please UpVote if you like the work!!!***
