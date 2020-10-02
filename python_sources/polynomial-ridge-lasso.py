#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


# In[ ]:


Data=pd.read_csv('../input/housing (1).csv',index_col=0)


# In[ ]:


Data.head()


# In[ ]:


corr=Data.corr()


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(corr,annot=True)


# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


Data.columns


# In[ ]:


m1=smf.ols('medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat',Data).fit()


# In[ ]:


m1.summary()

there is no significance for age and indus  hence we can drope the variables in our model
# In[ ]:


m2=smf.ols('medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat',Data).fit()


# In[ ]:


m2.summary()


# In[ ]:


#sns.pairplot(Data)


# In[ ]:


sns.pairplot(Data[['indus','age','medv']])


# ## quadritic (non linear form)

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=Data.drop(['medv','age','indus'],axis=1)
y=Data[['medv']]


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


x_train=X_train[['lstat','rm']]


# In[ ]:


qr=PolynomialFeatures(degree=2,include_bias=False)


# In[ ]:


x_qr=qr.fit_transform(x_train)
x_qr


# In[ ]:


x_qr_df=pd.DataFrame(x_qr)


# In[ ]:


x_qr_df.shape


# In[ ]:


Y_train.shape


# In[ ]:


idx=np.arange(len(Y_train))


# In[ ]:


Y_train.index=idx


# In[ ]:


x_qr_df=pd.concat([x_qr_df,Y_train],axis=1)


# In[ ]:


x_qr_df.shape


# In[ ]:


x_qr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','medv']


# In[ ]:


x_qr_df.head()


# In[ ]:


m2=smf.ols('medv~rm+lstat2+lstatXrm+rm2',x_qr_df).fit()


# In[ ]:


m2.summary()

lstat in not pass the test so lstat is removed from the model
# In[ ]:


x_test=X_test[['lstat','rm']]


# In[ ]:


xtest_qr=qr.fit_transform(x_test)


# In[ ]:


xtest_qr_df=pd.DataFrame(xtest_qr)


# In[ ]:


xtest_qr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2']


# In[ ]:


xtest_qr_df.head()


# In[ ]:


QR_pred=m2.predict(xtest_qr_df)


# In[ ]:


plt.plot(QR_pred,Y_test,'*')


# In[ ]:


from sklearn import metrics


# In[ ]:


MSE=metrics.mean_squared_error(QR_pred,Y_test)


# In[ ]:


QR_RMSE=np.sqrt(np.mean(MSE))
QR_RMSE

trying with ptratio inplace of rm hence rmse is more rm has high significance compared to ptratio
# ## linear model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression().fit(x_train,Y_train)


# In[ ]:


li_pred=model.predict(x_test)


# In[ ]:


plt.plot(li_pred,Y_test,'*')


# In[ ]:


MSE=metrics.mean_squared_error(li_pred,Y_test)


# In[ ]:


li_RMSE=np.sqrt(np.mean(MSE))
li_RMSE


# In[ ]:


R2=model.score(x_test,Y_test)
R2


# ## polinomial model

# In[ ]:


pr=PolynomialFeatures(degree=3,include_bias=False)


# In[ ]:


x_pr=pr.fit_transform(x_train)
x_pr


# In[ ]:


x_pr_df=pd.DataFrame(x_pr)


# In[ ]:


x_pr_df.head()


# In[ ]:


x_pr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','lstat3','lstat2Xrm','lstatXrm2','rm3']


# In[ ]:


x_pr_df.head()


# In[ ]:


x_pr_df=pd.concat([x_pr_df,Y_train],axis=1)


# In[ ]:


x_pr_df.head()


# In[ ]:


x_pr_df.columns


# In[ ]:


mp1=smf.ols('medv~lstat+rm+lstatXrm+rm2+lstatXrm2+rm3',x_pr_df).fit() # from summarey we drop lstat2 lastat3 lsatat2rm
mp1.summary()

Validate the polinomial model X_test 
# In[ ]:


xtest_pr=pr.fit_transform(x_test)


# In[ ]:


xtest_pr_df=pd.DataFrame(xtest_pr)


# In[ ]:


xtest_pr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','lstat3','lstat2Xrm','lstatXrm2','rm3']


# In[ ]:


xtest_pr_df.head()


# In[ ]:


PR_pred=mp1.predict(xtest_pr_df)


# In[ ]:


plt.plot(PR_pred,Y_test,'*')


# In[ ]:


MSE=metrics.mean_squared_error(PR_pred,Y_test)


# In[ ]:


PR_RMSE=np.sqrt(np.mean(MSE))
PR_RMSE


# ## RDGE and LASSO punishment for regression 

# In[ ]:


from sklearn.linear_model import Ridge,Lasso


# In[ ]:


rd=Ridge(alpha=0.5,normalize=True)
rd.fit(X_train,Y_train)


# In[ ]:


rd_pred=rd.predict(X_test)


# In[ ]:


ls=Lasso(alpha=0.05,normalize=True)
ls.fit(X_train,Y_train)


# In[ ]:


ls_pred=ls.predict(X_test)


# In[ ]:


rd.coef_


# In[ ]:


ls.coef_


# In[ ]:


Variable=X_test.columns
Variable


# In[ ]:


ridge=pd.Series(rd.coef_,Variable).sort_values()


# In[ ]:


ridge.plot(kind='bar')


# In[ ]:


lasso=pd.Series(ls.coef_,Variable).sort_values()


# In[ ]:


lasso.plot(kind='bar')


# ## compairing the model with K- fold Crosss validation method

# In[ ]:


from sklearn import metrics
from sklearn.model_selection import KFold
LR=LinearRegression(normalize=True)
ridge_R=Ridge(alpha=0.5,normalize=True)
lasso_L=Lasso(alpha=0.1,normalize=True)


# In[ ]:


kf=KFold (n_splits=3, shuffle=True, random_state=2)
for model, name in zip([LR,ridge_R,lasso_L],['MVLR','RIDGE','LASSO']):
    rmse=[]
    for train,test in kf.split(x,y):
        X_train,X_test=x.iloc[train,:],x.iloc[test,:]
        Y_train,Y_test=y[train],y[test]
        model.fit(X_train,Y_train)
        Y_pred=model.predict(X_test)
        rmse.append(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
    print(rmse)
    print("cross VALIDATE RMSE score %0.03f (+/-%0.05f)[%s]"% (np.mean(rmse),np.var(rmse,ddof=1),name))


# In[ ]:





# In[ ]:




