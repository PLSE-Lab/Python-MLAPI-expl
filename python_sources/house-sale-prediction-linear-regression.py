#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import metrics
import scipy.stats as stats
import pylab
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True


# In[ ]:


df=pd.read_csv("../input/housesalesprediction/kc_house_data.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['date'] = df.date.str.strip('T000000')
df['date'] = pd.to_datetime(df.date , format='%Y%m%d')


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe().T


# In[ ]:


sns.distplot(df.price)


# In[ ]:


df['log_price'] = np.log(df.price)


# In[ ]:


sns.distplot(df.log_price)


# In[ ]:


corr = df.corr()
corr.style.background_gradient()


# In[ ]:


plt.subplots(figsize=(17,14))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt="1.1f")
plt.title("Data Correlation",fontsize=50)
plt.show()


# In[ ]:


# Drop variables based on low correlation
df=df.drop(['id','condition','yr_built','yr_renovated','zipcode','long','date'],axis=1)


# In[ ]:


df.head()


# In[ ]:


feature_columns=df.columns.difference(['price','log_price'])
feature_columns


# In[ ]:


train, test= train_test_split(df,test_size=0.3,random_state=12345)


# In[ ]:


print('train data :: ',train.shape)
print('test data :: ',test.shape)


# In[ ]:


lm=smf.ols('log_price ~ bathrooms + bedrooms + floors + grade + lat + sqft_above + sqft_basement + sqft_living + sqft_living15 + sqft_lot + sqft_lot15 + view + waterfront',train).fit()
lm.summary()


# In[ ]:


train['pred_price'] = np.exp(lm.predict(train))
train['error'] = train['price'] - train['pred_price']
train.head()


# In[ ]:


test['pred_price'] = np.exp(lm.predict(test))
test['error'] = test['price'] - test['pred_price']
test.head()


# In[ ]:


# Accuracy metrices
MAPE_train = np.mean(np.abs(train.error) / train.price) * 100
MAPE_test = np.mean(np.abs(test.error) / test.price) * 100
print(MAPE_train)
print(MAPE_test)


# In[ ]:


lm.resid.hist(bins=10)


# In[ ]:


lm.resid.mean()


# In[ ]:


sns.distplot(lm.resid)


# In[ ]:


sns.distplot(test.error)


# In[ ]:


sns.jointplot(train.price,train.error)


# In[ ]:


stats.probplot(train.error,dist='norm',plot=pylab)
pylab.show()

