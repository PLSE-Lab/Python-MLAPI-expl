#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 


# In[ ]:


df = pd.read_csv("../input/realestate.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns=df.columns.str.lower()


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


plt.figure(1)
plt.subplot(121)
sns.distplot(df['unit_area']);
plt.subplot(122)
df['unit_area'].plot.box(figsize=(16,5))
plt.show()


# In[ ]:


# univariate analyis : ---
plt.figure(1)
plt.subplot(121)
sns.distplot(df['houseage']);
plt.subplot(122)
df['houseage'].plot.box(figsize=(16,5))
plt.show()


# In[ ]:


plt.figure(1)
plt.subplot(121)
sns.distplot((df['distance']));
plt.subplot(122)
(df['distance']).plot.box(figsize=(16,5))
plt.show()


# In[ ]:


df.columns


# In[ ]:


plt.figure(1)
plt.subplot(121)
sns.distplot(df['stores']);
plt.subplot(122)
df['stores'].plot.box(figsize=(16,5))
plt.show()


# In[ ]:


# checking na 
df.isna().sum()


# In[ ]:


df['transactiondate']=df['transactiondate'].astype(int)


# In[ ]:


sns.catplot(x = 'transactiondate' , y = 'unit_area' , data =df)


# In[ ]:


sns.lineplot(x ='transactiondate' , y ='unit_area' , data =df)


# In[ ]:


df['transactiondate'].value_counts()


# In[ ]:


for col in df.columns:
    print("--------****-------")
    print (df[col].value_counts().head())
    


# In[ ]:


df.head()


# In[ ]:


df['distance'].describe()


# In[ ]:


df['distance'].plot.box()


# In[ ]:


df['distance']=df['distance'].clip(23,1453)


# In[ ]:


df['distance'].plot.box()


# In[ ]:


df['distance'].plot.hist()


# In[ ]:


df['houseage']=df['houseage'].astype(int)


# In[ ]:


df.head()


# In[ ]:


df['distance']=df['distance'].astype(int)


# In[ ]:


# collect x and y
df.columns
X = df[['houseage', 'distance', 'stores']]
y = df['unit_area']


# In[ ]:


df.columns


# In[ ]:


import statsmodels.formula.api as smf # for regression model


# In[ ]:


ml1 = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df).fit() # regression model


# In[ ]:


ml1.summary()


# In[ ]:


import statsmodels.api as sm
sm.graphics.influence_plot(ml1)


# In[ ]:


df_new=df.drop(df.index[270],axis=0)


# In[ ]:


df_new


# In[ ]:


ml2 = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_new).fit() # regression model


# In[ ]:


ml2.summary()


# In[ ]:


sm.graphics.influence_plot(ml2,)


# In[ ]:


rsquared=smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_new).fit().rsquared


# In[ ]:


vif_sp = 1/(1-rsquared)
vif_sp  # vif value


# In[ ]:


rsquared2=smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df).fit().rsquared


# In[ ]:


vif_sp = 1/(1-rsquared2)
vif_sp  # vif value


# In[ ]:


# Added varible plot 
sm.graphics.plot_partregress_grid(ml2)


# In[ ]:


# predict the data 
pred=ml2.predict(df_new)


# In[ ]:


# checking linearity
# Observed values VS Fitted values
plt.scatter(df_new.unit_area,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# In[ ]:


#Residuals VS Fitted Values 
plt.scatter(pred,ml2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[ ]:


########    Normality plot for residuals ######
# histogram
plt.hist(ml2.resid_pearson) # Checking the standardized residuals are normally distributed


# In[ ]:


# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml2.resid_pearson, dist="norm", plot=pylab)


# In[ ]:


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(pred,ml2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[ ]:


### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
df_train,df_test  = train_test_split(df_new,test_size = 0.2) # 20% size


# In[ ]:


df_train.shape,df_test.shape


# In[ ]:


model_train = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_train).fit()


# In[ ]:


model_train


# In[ ]:


# train_data prediction
train_pred = model_train.predict(df_train)


# In[ ]:


train_pred.head()


# In[ ]:


# trian residual values 
train_resid  = train_pred - df_train.unit_area


# In[ ]:


train_resid.plot.hist()


# In[ ]:


# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse


# In[ ]:


# prediction on test data set 
test_pred = model_train.predict(df_test)


# In[ ]:


# test residual values 
test_resid  = test_pred - df_test.unit_area


# In[ ]:


# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))


# In[ ]:


test_rmse,train_rmse

