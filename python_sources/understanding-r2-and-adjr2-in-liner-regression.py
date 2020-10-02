#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Basics

# In[ ]:


x=np.array([6.5,6.8,7.0,8.0,8.5,9.8,10.5,5.5,5.0])
y=np.array([180,195,210,235,280,350,400,170,160])


# In[ ]:


sns.scatterplot(x,y)


# In[ ]:


b1=((x-np.mean(x))*(y-np.mean(y))).sum()/((x-np.mean(x))**2).sum()


# In[ ]:


b1==np.cov(x,y,ddof=1)[1,0]/np.var(x,ddof=1)


# In[ ]:


b0=np.mean(y)-(b1*np.mean(x))
b0


# In[ ]:


y_pred=b0+(b1*x)
y_pred


# In[ ]:


sum_of_residues= (y-y_pred).sum()
sum_of_residues


# In[ ]:


sse=((y-y_pred)**2).sum()
sse


# In[ ]:


mse=sse/len(x)
mse


# In[ ]:


sns.scatterplot(x,y)
sns.lineplot(x,y_pred,)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


df=pd.DataFrame(x,columns=['x'])


# In[ ]:


lr=LinearRegression()
lr.fit(df,y)
ypred=lr.predict(df)


# In[ ]:


lr.coef_


# In[ ]:


lr.intercept_


# In[ ]:


ypred,y_pred


# In[ ]:


lr.score(df,y)


# In[ ]:


1-(((y-ypred)**2).sum()/((y-y.mean())**2).sum())


# In[ ]:


(np.corrcoef(y,ypred)[1,0])**2


# In[ ]:


from statsmodels.formula.api import ols


# In[ ]:


df['y']=y


# In[ ]:


df


# In[ ]:


model=ols('y~x',df).fit()


# In[ ]:


model.params


# In[ ]:


model.predict(df.x)


# In[ ]:


model.summary()


# In[ ]:


n,p=df.drop('y',1).shape
r2=1-(((y-ypred)**2).sum()/((y-y.mean())**2).sum())
adjusted_r2= 1 - ((1-r2) * (n-1) / (n-p-1))
adjusted_r2


# In[ ]:


df.shape


# # car-mpg

# In[ ]:


df=pd.read_csv('../input/learn-ml-datasets/car-mpg (1).csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


[(i,v) for i,v in enumerate(df.hp) if not v.isdigit()]


# In[ ]:


df.shape


# In[ ]:


df.drop('car_name',1,inplace=True)
df.shape


# In[ ]:


df.replace('?',np.nan,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.fillna(df.hp.median(),inplace=True)


# In[ ]:


df.info()


# In[ ]:


df['hp']=df.hp.astype(int)


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)


# In[ ]:


sns.pairplot(df,diag_kind='kde')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=df.drop(['mpg'],1)
y=df.mpg


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)


# In[ ]:


lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
lr.score(xtest,ytest),lr.score(xtrain,ytrain)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(ytest,ypred)


# In[ ]:


r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())
n,p=x.shape
adj_r2=1-((1-r2)*(n-1)/(n-p-1))
r2,adj_r2


# In[ ]:


#correlation of ytest and ypred as cr
cr=((ytest-(np.mean(ytest)))*(ypred-(np.mean(ypred)))).sum()/(((((ytest-(np.mean(ytest)))**2).sum())**0.5)*(((ypred-(np.mean(ypred)))**2).sum())**0.5)
cr**2


# In[ ]:


mse=np.mean((ytest-ypred)**2)
mse


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,ypred)


# In[ ]:


rmse=mse**0.5
rmse


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(ytest,ypred)
sns.scatterplot(ytest,ytest)


# In[ ]:


for i in df.columns:
  print(i,df[i].nunique())


# In[ ]:


for i in ['cyl','origin','car_type']:
  df[i]=df[i].apply(str)


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)


# In[ ]:


x=df.drop(['mpg'],1)
x=pd.get_dummies(x,drop_first=True)
y=df.mpg


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)


# In[ ]:


lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
lr.score(xtest,ytest),lr.score(xtrain,ytrain)


# In[ ]:


# We can see how the adjusted_r2 is being affected by increasing the number of features using dummification 
r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())
n,p=x.shape
adj_r2=1-((1-r2)*(n-1)/(n-p-1))
r2,adj_r2


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


for i in range(x.shape[1]):
  print(x.columns[i],variance_inflation_factor(x.values, i))


# In[ ]:


# we eliminate features to see the change in adj R2. It is our decision to choose inbetween higher r2 and no multicollinearity.
# Higher r2 means better results, no multicollienarity means stable relationship between y and x variables.
x=df.drop(['mpg','origin','car_type','acc'],1)
x=pd.get_dummies(x,drop_first=True)
y=df.mpg
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())
n,p=x.shape
adj_r2=1-((1-r2)*(n-1)/(n-p-1))
print(r2,adj_r2)
for i in range(x.shape[1]):
  print(x.columns[i],variance_inflation_factor(x.values, i))

