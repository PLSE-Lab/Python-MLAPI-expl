#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries
import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


print("Train data shape:",train.shape)
print("Test data shape:",test.shape)


# In[ ]:


train.head()


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)


# In[ ]:


train.SalePrice.describe()


# In[ ]:


print("Skew is:",train.SalePrice.skew())
plt.hist(train.SalePrice,color='orange')
plt.show()


# In[ ]:


target = np.log(train.SalePrice)
print("Skew is:",target.skew())
plt.hist(target,color='orange')
plt.show()


# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[ ]:


corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])


# In[ ]:


train.OverallQual.unique()


# In[ ]:


quality_pivot = train.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.median)


# In[ ]:


quality_pivot


# In[ ]:


quality_pivot.plot(kind='bar',color='orange')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


plt.scatter(x=train['GrLivArea'],y=target,color='brown')
plt.ylabel('Sale Price')
plt.xlabel('Above grade (Ground) living area square feet')
plt.show()


# In[ ]:


plt.scatter(x=train['GarageArea'],y=target,color='brown')
plt.ylabel('Sale Price')
plt.xlabel('GarageArea')
plt.show()


# In[ ]:


train=train[train['GarageArea']<1200]


# In[ ]:


plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice),color='brown')
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('GarageArea')
plt.show()


# In[ ]:


train.isnull().sum()


# In[ ]:


nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=["Null count"]
nulls.index.name='Feature'
nulls


# In[ ]:


print("Unique values are:",train.MiscFeature.unique())


# In[ ]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# In[ ]:


print("Original: \n")
print(train.Street.value_counts(),"\n")


# In[ ]:


train['enc_street']=pd.get_dummies(train.Street, drop_first=True)
test['enc_street']=pd.get_dummies(test.Street, drop_first=True)


# In[ ]:


print('Encoded: \n')
print(train.enc_street.value_counts())


# In[ ]:


condition_pivot=train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind='bar',color='orange')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


def encode(x):
    return 1 if x == 'Partial' else 0
train['enc_condition']=train.SaleCondition.apply(encode)
test['enc_condition']=test.SaleCondition.apply(encode)


# In[ ]:


condition_pivot=train.pivot_table(index='enc_condition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind='bar',color='orange')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[ ]:


sum(data.isnull().sum()!=0)


# In[ ]:


y=np.log(train.SalePrice)
X=data.drop(['SalePrice','Id'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.33)


# In[ ]:


from sklearn import linear_model
lr=linear_model.LinearRegression()


# In[ ]:


model =lr.fit(X_train,y_train)


# In[ ]:


print('R^2 is: \n',model.score(X_test,y_test))


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
print('RMSE is: \n', mean_squared_error(y_test,predictions))


# In[ ]:


actual_values = y_test
plt.scatter(predictions,actual_values,alpha=.7,color='brown')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:


for i in range (-2,3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model=rm.fit(X_train,y_train)
    preds_ridge =ridge_model.predict(X_test)
    
    plt.scatter(preds_ridge,actual_values,alpha=.75, color='purple')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test,y_test),mean_squared_error(y_test,preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


# In[ ]:


submission = pd.DataFrame()
submission['Id']=test.Id


# In[ ]:


feats= test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()


# In[ ]:


predictions=model.predict(feats)


# In[ ]:


final_predictions = np.exp(predictions)


# In[ ]:


print("Original predictions are: \n",predictions[:5], "\n")
print("Final predictions are: \n", final_predictions[:5])


# In[ ]:


submission['SalePrice']=final_predictions
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




