#!/usr/bin/env python
# coding: utf-8

# In[34]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.ensemble import RandomForestRegressor

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import KFold
import seaborn as sns
from scipy import stats 
from scipy.stats import norm, skew ,zscore#for some statistics
import matplotlib.pyplot as plt  # Matlab-style plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[35]:


'''
PATH1 = os.path.join(os.getcwd(), os.path.join('data', 'trainHouses.csv'))
PATH2= os.path.join(os.getcwd(), os.path.join('data', 'testHouses.csv'))

train = pd.read_csv(PATH1, delimiter=',')
test = pd.read_csv(PATH2, delimiter=',')

train.head()
'''
train=pd.read_csv("../input/train.csv")
test=pd.read_csv('../input/test.csv')


# In[36]:


train.shape,test.shape


# In[ ]:







# In[37]:


train.info()


# In[ ]:





# In[38]:


test.columns^train.columns


# In[39]:





# In[40]:


numerc_fet=train.select_dtypes(include=np.number)


# In[41]:


corr=numerc_fet.corr()


# In[42]:


corr['SalePrice'].sort_values(ascending=False)[:9]


# In[43]:


#Handling outliers
plt.scatter(Xtrain['GrLivArea'], ytrain)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# In[46]:


train=train[train['GrLivArea']<3200]


# In[ ]:





# In[49]:


ytrain=train['SalePrice']
Xtrain=train.drop('SalePrice',axis=1)


# In[50]:


Xtrain.shape


# In[51]:


test.columns^Xtrain.columns  #for a check


# In[52]:


ytrain.shape,Xtrain.shape,test.shape


# In[53]:


totalData=pd.concat([Xtrain,test])


# In[54]:


totalData.shape


# In[ ]:


missing=totalData.isnull().sum().sort_values(ascending=False)
missing=missing[missing>0]
missing


# # Dealing With Missing data

# In[55]:


#for catergorical variables, we replece missing data with None
Miss_cat=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in Miss_cat:
    totalData[col].fillna('None',inplace=True)
# for numerical variables, we replace missing value with 0
Miss_num=['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'] 
for col in Miss_num:
    totalData[col].fillna(0, inplace=True)


# In[56]:


rest_val=['MSZoning','Functional','Utilities','Exterior1st', 'SaleType','Electrical', 'Exterior2nd','KitchenQual']
for col in rest_val:
    totalData[col].fillna(totalData[col].mode()[0],inplace=True)


# In[57]:


totalData['LotFrontage']=totalData.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:





# In[58]:


totalData=totalData.drop('Id',axis=1)


# In[ ]:





# In[59]:



sns.distplot(ytrain , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(ytrain)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(ytrain, plot=plt)
plt.show()


# In[62]:


ytrain.skew()


# In[61]:


ytrain=np.log(ytrain)


# In[63]:



ytrain=pd.DataFrame(ytrain)


# In[64]:


ytrain.plot.hist()


# In[ ]:





# In[ ]:





# In[65]:


ytrain.head(2)


# In[ ]:





# In[66]:


#convert the numeric values into string becuse there are many repetition 
totalData['YrSold'] = totalData['YrSold'].astype(str)
totalData['MoSold'] = totalData['MoSold'].astype(str)
totalData['MSSubClass'] = totalData['MSSubClass'].astype(str)
totalData['OverallCond'] = totalData['OverallCond'].astype(str)


# In[67]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(totalData[c].values)) 
    totalData[c] = lbl.transform(list(totalData[c].values))


# In[68]:


# shape        
print('Shape totalData: {}'.format(totalData.shape))


# In[69]:


totalData=pd.DataFrame(totalData)
ytrain=pd.DataFrame(ytrain)


# In[70]:


numeric_feats = totalData.dtypes[totalData.dtypes != "object"].index
string_feats=totalData.dtypes[totalData.dtypes == "object"].index


# In[71]:


numeric_feats


# In[72]:


string_feats


# In[ ]:





# In[73]:


totalData.shape


# In[74]:


#totalData.plot(kind="density", figsize=(50,50))
#totalData.plot.hist()
#pd.plotting.scatter_matrix(totalData,figsize=(12,12))


# In[ ]:





# In[ ]:





# In[75]:


dumies = pd.get_dummies(totalData[string_feats])
print(dumies.shape)


# In[76]:


totalData=pd.concat([totalData,dumies],axis='columns')


# In[77]:


totalData.shape


# In[78]:


totalData=totalData.drop(string_feats,axis=1)


# In[ ]:





# In[79]:


x=len(ytrain)


# In[80]:


totalData=pd.DataFrame(totalData)


# In[ ]:





# In[81]:


train_feature=totalData.iloc[:x,:]
test_feature=totalData.iloc[x:,:]


# In[82]:


#from sklearn.preprocessing import MinMaxScaler

#sc_X = MinMaxScaler()
train_sc = train_feature
test_sc = test_feature

ytrain_sc=ytrain


# In[83]:


train_feature.shape,test_feature.shape,ytrain.shape


# In[84]:


train_sc=pd.DataFrame(train_sc)
ytrain_sc=pd.DataFrame(ytrain_sc)


# In[85]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(train_sc,ytrain_sc,test_size=0.2,random_state=42)


# # First Model is linear Regression

# In[86]:


model1= LinearRegression()
model1.fit(X_train,Y_train)


# In[87]:


ypre1=model1.predict(X_test)


# In[88]:


mean = mean_squared_error(y_pred=ypre1,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre1,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre1,y_true=Y_test)
print(mean,r2_scor,absloute)


# #  Try more Models  

# In[89]:


# Test Options and Evaluation Metrics
num_folds = 5
scoring = "neg_mean_squared_error"
# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('RFR', RandomForestRegressor()))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=0)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,    scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),   cv_results.std())
    print(msg)


# In[91]:


Y_train.skew()


# In[92]:


model4= Lasso()
model4.fit(X_train,Y_train)


# In[ ]:





# In[93]:


ypre4=model4.predict(X_test)


# In[94]:


mean = mean_squared_error(y_pred=ypre4,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre4,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre4,y_true=Y_test)
print(mean,r2_scor,absloute)


# In[95]:


submission = pd.DataFrame()
submission['Id'] = test.Id


# In[ ]:





# In[97]:


test_sc=pd.DataFrame(test_sc)
yprett=model4.predict(test_sc)


# In[98]:


final_predictions = np.exp(yprett)


# In[99]:


print ("Original predictions are: \n", yprett[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])


# In[100]:


submission['SalePrice'] = final_predictions
submission.head()


# In[ ]:


#yprett=np.log1p(yprett)


# In[101]:


submission.to_csv('submission1.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




