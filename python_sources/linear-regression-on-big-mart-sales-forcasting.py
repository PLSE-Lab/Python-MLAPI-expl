#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('/kaggle/input/big-mart-sales-forcasting/train.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


nulldf = df[df['Item_Weight'].isnull()]


# In[ ]:


nulldf.head()


# In[ ]:


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Outlet_Size'].value_counts()


# In[ ]:


df['Outlet_Size'] = df['Outlet_Size'].fillna('NA')
df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


a = df.groupby('Item_Fat_Content', as_index=False)['Item_Identifier'].count()
a


# In[ ]:


import numpy as np
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='LF', 'Low Fat', df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='low fat', 'Low Fat', df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='reg', 'Regular', df['Item_Fat_Content'])


# In[ ]:


df['Count'] = 1


# In[ ]:


Fat_Content = df.groupby('Item_Fat_Content', as_index=False)['Count'].sum()
Fat_Content


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(x = 'Item_Fat_Content', y = 'Count', data=Fat_Content)
plt.show()


# In[ ]:


item_type = df.groupby('Item_Type', as_index=False)['Count'].count()
item_type.sort_values('Count', ascending=False)


# In[ ]:


ax2 = sns.barplot(x = 'Item_Type', y = 'Count', data = item_type)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90)
plt.show()


# In[ ]:


ax3 = sns.distplot(df['Item_MRP'])


# In[ ]:


ax3 = sns.distplot(df['Item_MRP'], bins=10)


# In[ ]:


box = sns.boxplot(df['Item_MRP'])


# In[ ]:


box2 = sns.boxplot(df['Item_Outlet_Sales'])


# In[ ]:


from scipy.stats import iqr

q1 = df['Item_Outlet_Sales'].quantile(0.25)
q3 = df['Item_Outlet_Sales'].quantile(0.75)
inter_qr = iqr(df['Item_Outlet_Sales'])
print(q1)
print(q3)
print(inter_qr)


# In[ ]:


df['Outliers'] = 0
df['Outliers'] = np.where(df['Item_Outlet_Sales']>(q3+1.5*inter_qr), 1, df['Outliers'])
df['Outliers'] = np.where(df['Item_Outlet_Sales']<(q1-1.5*inter_qr), 1, df['Outliers'])
df.head()


# In[ ]:


box2 = sns.boxplot(df['Item_Visibility'])


# In[ ]:


vq1 = df['Item_Visibility'].quantile(0.25)
vq3 = df['Item_Visibility'].quantile(0.75)
vinter_qr = iqr(df['Item_Visibility'])
print(vq1)
print(vq3)
print(vinter_qr)


# In[ ]:


df['Outliers'] = np.where(df['Item_Visibility']>(vq3+1.5*vinter_qr), 1, df['Outliers'])
df['Outliers'] = np.where(df['Item_Visibility']<(vq1-1.5*vinter_qr), 1, df['Outliers'])
df.head()


# In[ ]:


df.Outliers.value_counts()


# In[ ]:


df.Outlet_Establishment_Year.describe()


# In[ ]:


df['Outlet_Age'] = 2015 - df['Outlet_Establishment_Year']
df.head()


# In[ ]:


df['Outlet_Age'].describe()


# In[ ]:


dfclean = df[df['Outliers']==0]
print(df.shape)
print(dfclean.shape)


# In[ ]:


dfclean = dfclean.drop(columns=['Item_Identifier', 'Count', 'Outliers'])
dfclean.shape


# In[ ]:


dfclean = pd.get_dummies(dfclean)
dfclean.shape


# In[ ]:


dfclean.head()


# In[ ]:


x = dfclean.drop(columns='Item_Outlet_Sales')
y = dfclean['Item_Outlet_Sales']

print(x.shape)
print(y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[ ]:


import statsmodels.api as sm

model = sm.OLS(ytrain, xtrain).fit()


# In[ ]:


print(model.summary())


# In[ ]:


pred = model.predict(xtest)

data = list(zip(ytest, pred))


# In[ ]:


comptab = pd.DataFrame(data, columns=['Actual', 'Predicted'])
comptab.head()


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(ytest, pred))
print(rmse)


# In[ ]:




