#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import xgboost
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


DF = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
Test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


DF.head(5)


# In[ ]:


#Plotting the distribution of sales Price
plt.figure(figsize=(20,5))
sns.distplot(DF.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density");


# In[ ]:


# Visualize the correlation between the number of missing values in different columns of dataset as a heatmap 
msno.heatmap(DF)


# In[ ]:


# correlation heatmap
plt.figure(figsize=(10,8))
cor = DF.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Plotting the correlation values with the sales price 
DF.corrwith(DF.SalePrice).plot.bar(
                                    figsize = (20, 10), title = "Correlation with class", fontsize = 15,
                                     rot = 90, grid = True)


# In[ ]:


plt.figure(figsize=[20,10])
plt.subplot(331)
sns.distplot(DF['LotFrontage'].dropna().values)
plt.subplot(332)
sns.distplot(DF['GarageYrBlt'].dropna().values)
plt.subplot(333)
sns.distplot(DF['MasVnrArea'].dropna().values)
plt.suptitle("Distribution of data before Filling NA'S")


# **Impute Missing Values**
# 
# As we can see in LotFrontage plot the distribution is approimately Noraml. So we can use either mean or mode to replace the missing values.
# In GarageYrBlt plot the distribution is skewed and median is favourable to impute the missing values in this case.
# And in final MasVnrArea distribution which is also skewed, we'll use median to impute missing values. 

# In[ ]:


DF['LotFrontage']=DF.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
DF['GarageYrBlt']=DF.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
DF['MasVnrArea']=DF.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# Now apply the same in Test dataset

# In[ ]:


Test['LotFrontage']=Test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
Test['GarageYrBlt']=Test.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
Test['MasVnrArea']=Test.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# Now we'll made some new features that sounds obvious but not so certain at first. When we actually look for a house, we keep all these features mentioned below in our mind. Some wise person went through all these details in the discussion. Let's apply them here.

# In[ ]:


DF['cond*qual'] = (DF['OverallCond'] * DF['OverallQual']) / 100.0
DF['home_age_when_sold'] = DF['YrSold'] - DF['YearBuilt']
DF['garage_age_when_sold'] = DF['YrSold'] - DF['GarageYrBlt']
DF['TotalSF'] = DF['TotalBsmtSF'] + DF['1stFlrSF'] + DF['2ndFlrSF'] 
DF['total_porch_area'] = DF['WoodDeckSF'] + DF['OpenPorchSF'] + DF['EnclosedPorch'] + DF['3SsnPorch'] + DF['ScreenPorch'] 
DF['Totalsqrfootage'] = (DF['BsmtFinSF1'] + DF['BsmtFinSF2'] + DF['1stFlrSF'] + DF['2ndFlrSF'])
DF['Total_Bathrooms'] = (DF['FullBath'] + (0.5 * DF['HalfBath']) + DF['BsmtFullBath'] + (0.5 * DF['BsmtHalfBath']))


# In[ ]:


Test['cond*qual'] = (Test['OverallCond'] * Test['OverallQual']) / 100.0
Test['home_age_when_sold'] = Test['YrSold'] - Test['YearBuilt']
Test['garage_age_when_sold'] = Test['YrSold'] - Test['GarageYrBlt']
Test['TotalSF'] = Test['TotalBsmtSF'] + Test['1stFlrSF'] + Test['2ndFlrSF'] 
Test['total_porch_area'] = Test['WoodDeckSF'] + Test['OpenPorchSF'] + Test['EnclosedPorch'] + Test['3SsnPorch'] + Test['ScreenPorch'] 
Test['Totalsqrfootage'] = (Test['BsmtFinSF1'] + Test['BsmtFinSF2'] + Test['1stFlrSF'] + Test['2ndFlrSF'])
Test['Total_Bathrooms'] = (Test['FullBath'] + (0.5 * Test['HalfBath']) + Test['BsmtFullBath'] + (0.5 * Test['BsmtHalfBath']))


# In[ ]:


Old_Cols=['OverallCond','OverallQual','YrSold','YearBuilt','YrSold','GarageYrBlt','TotalBsmtSF','1stFlrSF','2ndFlrSF','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']


# Let's delete the features used in making the new features. They're not useful anymore.

# In[ ]:


Final_cols=[]
for i in DF.columns:
    if i not in Old_Cols and i!='SalePrice':
        Final_cols.append(i)
PF=DF[Final_cols]


# In[ ]:


Final_cols=[]
for i in Test.columns:
    if i not in Old_Cols and i!='SalePrice':
        Final_cols.append(i)
TF=Test[Final_cols]


# In[ ]:


PF.columns


# ** Again let's check the correlation in our new features. **

# In[ ]:


#price range correlation
corr=DF.corr()
corr=corr.sort_values(by=["SalePrice"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15,20))
sns.barplot(x=corr.values, y =corr.index.values);
plt.title("Correlation Plot")


# In[ ]:


y = DF.SalePrice


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(PF, y, test_size=0.3)


# In[ ]:


def Change(x):
    for col in x.select_dtypes(include=['object']).columns:
               x[col] = x[col].astype('category')
    for col in x.select_dtypes(include=['category']).columns: 
               x[col] = x[col].cat.codes
    return x  


# In[ ]:


X_train = Change(X_train)
X_test = Change(X_test)


# ** Let's Build Our XGBoost Model **

# In[ ]:


model = XGBRegressor(colsample_bytree=1,
                 gamma=0.5,                 
                 learning_rate=0.005,
                 max_depth=9,
                 min_child_weight=1.5,
                 n_estimators=5000,                                                                    
                 reg_alpha=0.4,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[ ]:


model.fit(X_train, y_train)
model.score(X_test,y_test)*100


# In[ ]:


feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


SalePrice = pd.DataFrame(model.predict(Change(TF)))
Id = pd.DataFrame(TF.Id)
result = pd.concat([Id, SalePrice], axis=1)
result.columns = ['Id', 'SalePrice']


# In[ ]:


result.to_csv('submission.csv',index=False)


# If you think, you can modify this kernel to achieve a better score than you're welcome to do so. But please comment the approch in the commenting session.
