#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)


# In[ ]:


train.head()


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[ ]:


train.SalePrice.describe()


# In[ ]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# In[ ]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[ ]:


corr = numeric_features.corr()
corr


# In[ ]:


print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])


# In[ ]:


train['OverallQual'].unique()


# In[ ]:


quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)


# In[ ]:


quality_pivot


# In[ ]:


quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# In[ ]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[ ]:


train = train[train['GarageArea'] < 1200]


# In[ ]:


plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[ ]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
tnulls.index.name = 'Feature'
nulls


# In[ ]:


print ("Unique values are:", train.MiscFeature.unique())


# In[ ]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# In[ ]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[ ]:


print ('Encoded: \n') 
print (train.enc_street.value_counts())


# In[ ]:


def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# In[ ]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[ ]:


data


# In[ ]:


y = np.log(train.SalePrice)


# In[ ]:


X = data.drop(['SalePrice', 'Id'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[ ]:


model = lr.fit(X_train, y_train)


# In[ ]:


print ("R^2 is: \n", model.score(X_test, y_test))


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[ ]:


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test.Id


# In[ ]:


feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()


# In[ ]:


predictions = model.predict(feats)


# In[ ]:


final_predictions = np.exp(predictions)


# In[ ]:


print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])


# In[ ]:


submission['SalePrice'] = final_predictions
submission.head()


# In[ ]:


submission.to_csv('submission1.csv', index = False)


# In[ ]:




