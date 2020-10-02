#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


# Let's read our data into raw_data as a dataset and examine the contents and shape 

# In[ ]:


raw_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
pd.set_option("display.max_columns",0)
raw_data


# The data consists of 1460 rows and 81 columns

# In[ ]:


raw_data.shape


# In[ ]:


numeric = raw_data.select_dtypes({'int64','float64'}).columns
numeric


# In[ ]:


categorical = raw_data.select_dtypes({'object'}).columns
categorical


# In[ ]:


#Number of numeric columns
raw_data.select_dtypes({'int64','float64'}).shape[1]


# In[ ]:


#Number of categorical columns
raw_data.select_dtypes({'object'}).shape[1]


# # Step 1 : Analysing 'SalePrice'

# Before we dive into exploring datasets and deploying ML algorithms, it would be nice to look into the SalePrice variable as it's the reason of our quest.

# In[ ]:


raw_data['SalePrice'].describe()


# The minimum price isn't negative. Thus, one less thing to worry.

# In[ ]:


sns.distplot(raw_data['SalePrice'])


# In[ ]:


raw_data['SalePrice'].skew()


# Going by the definition, positve skewness implies that more number of houses are being sold for price less than the average value.

# In[ ]:


raw_data['SalePrice'].kurt()


# Kurtosis : The measure of outliers. Leptokurtic (positive kurtosis) implies that our dataset has heavy outliers

# # Intuitive parameters affecting SalePrice

# To my knowledge and the variables present in the dataset. I classify the variables into 4 categories which one looks upon while deciding a house to buy.
# 1. SIZE/AREA - LotArea, TotalBsmntSF, Bedroom, GrLivArea
# 2. LOCATION - Neighborhood, Condition1 ( Proximity to main road )
# 3. BUILT - YearBuilt
# 4. QUALITY - OverallQual

# Best way to check the correlation between parameters is i guess, HeatMap. Let's check whether our intuition about factors affecting SalePrice is correct or does it differs in actual.

# In[ ]:


#correlation matrix
corrmat = raw_data.corr()
plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, square=True);


# Looking at the heatmap, the factors having a strong correlation with SalePrice I could see are : OverallQual, GrLivArea, GarageCars, GarageArea. We should also consider mild correlated factors which could be : YearBuilt, 1stFlrSF, TotalBsmntSF.
# Anyways, let's just take the top 10 factors depending upon the correlation with SalePrice.

# In[ ]:


k = 10
most_correlated = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(most_correlated)


# In[ ]:


factors = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']


# # Step 2 : Data Cleaning

# In[ ]:


#Checking for duplicate values:
dup_data = raw_data[raw_data.duplicated()]
dup_data.shape[0]


# Thus, we dont have any duplicate value in the data set.

# In[ ]:


total = pd.isnull(raw_data).sum().sort_values(ascending=False)
percentage = ((pd.isnull(raw_data).sum() / pd.isnull(raw_data).count()).sort_values(ascending=False))*100
no_values = pd.concat([total,percentage],axis = 1, keys = ['Total', 'Percent'])


# In[ ]:


no_values.head(19)


# It seems that the data for columns with more than 15% null values i.e. PoolQC, MiscFeatur, Alley, Fence, FireplaceQu, LotFrontage is missing a lot. Trying to replace these values by mean/median/mode would definitely mean changing the distribution of the dataset. For example if we try to replace the missing values by mean value, it would imply that we have deviated the dataset toward mean.

# In[ ]:


raw_data_without_null = raw_data.drop((no_values[no_values['Total'] > 1]).index,1)


# For other columns, it's upto you how you treat your data. If you have expertise in the concerned field, you could treat the missing values accordingly, else (my case here): None of these columns have a strong correlation with SalePrice. Thus, could remove the entire column.
# Also, for the 'Electrical' column we can just remove the row containing missing value, as only a single value is missing.

# In[ ]:


raw_data_without_null = raw_data_without_null.dropna()


# In[ ]:


pd.isnull(raw_data_without_null).sum().max()


# In[ ]:


raw_data_without_null.shape


# OUTLIER DETECTION

# I'll be using the box and whisker plot method to detect the outliers.
# 1. I used the flooring and capping techique to mark the boundary.
# 2. I used the outer boundary to remove the confirmed outliers and not the suspected outliers. If you wish you can try removing the suspected outliers as well.

# In[ ]:


sns.boxplot(raw_data_without_null['OverallQual'])
q1 = raw_data_without_null['OverallQual'].quantile(.25)
q3 = raw_data_without_null['OverallQual'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['OverallQual'] > floor) & (raw_data_without_null['OverallQual'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['GrLivArea'])
q1 = raw_data_without_null['GrLivArea'].quantile(.25)
q3 = raw_data_without_null['GrLivArea'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['GrLivArea'] > floor) & (raw_data_without_null['GrLivArea'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['GarageCars'])
q1 = raw_data_without_null['GarageCars'].quantile(.25)
q3 = raw_data_without_null['GarageCars'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['GarageCars'] > floor) & (raw_data_without_null['GarageCars'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['GarageArea'])
q1 = raw_data_without_null['GarageArea'].quantile(.25)
q3 = raw_data_without_null['GarageArea'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['GarageArea'] > floor) & (raw_data_without_null['GarageArea'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['TotalBsmtSF'])
q1 = raw_data_without_null['TotalBsmtSF'].quantile(.25)
q3 = raw_data_without_null['TotalBsmtSF'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['TotalBsmtSF'] > floor) & (raw_data_without_null['TotalBsmtSF'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['1stFlrSF'])
q1 = raw_data_without_null['1stFlrSF'].quantile(.25)
q3 = raw_data_without_null['1stFlrSF'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['1stFlrSF'] > floor) & (raw_data_without_null['1stFlrSF'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['FullBath'])
q1 = raw_data_without_null['FullBath'].quantile(.25)
q3 = raw_data_without_null['FullBath'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['FullBath'] > floor) & (raw_data_without_null['FullBath'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['TotRmsAbvGrd'])
q1 = raw_data_without_null['TotRmsAbvGrd'].quantile(.25)
q3 = raw_data_without_null['TotRmsAbvGrd'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['TotRmsAbvGrd'] > floor) & (raw_data_without_null['TotRmsAbvGrd'] < cap)]


# In[ ]:


sns.boxplot(raw_data_without_null['YearBuilt'])
q1 = raw_data_without_null['YearBuilt'].quantile(.25)
q3 = raw_data_without_null['YearBuilt'].quantile(.75)
iqr = q3 - q1
floor = q1 - 3*iqr
cap = q3 + 3*iqr
print('Floor = {}, Capping = {}'.format(floor,cap))
raw_data_without_null = raw_data_without_null[(raw_data_without_null['YearBuilt'] > floor) & (raw_data_without_null['YearBuilt'] < cap)]


# In[ ]:


raw_data_without_null.shape


# # Step 3 : Checking conditions for Linear Model

# 1. Linearity - Check using scatter plots
# 2. No Endogeneity
# 3. Homoscedasticity
# 4. No autocorrelation

# In[ ]:


data_cleaned = raw_data_without_null.copy()


# In[ ]:


data_use = data_cleaned[factors].copy()


# In[ ]:


f, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(1,9, sharey = True, figsize = (30,5))
ax1.scatter(data_use['OverallQual'],data_use['SalePrice'])
ax1.set_title('OverallQual and SalePrice')

ax2.scatter(data_use['GrLivArea'],data_use['SalePrice'])
ax2.set_title('GrLivArea and SalePrice')

ax3.scatter(data_use['GarageCars'],data_use['SalePrice'])
ax3.set_title('GarageCars and SalePrice')

ax4.scatter(data_use['GarageArea'],data_use['SalePrice'])
ax4.set_title('GarageArea and SalePrice')

ax5.scatter(data_use['TotalBsmtSF'],data_use['SalePrice'])
ax5.set_title('TotalBsmtSF and SalePrice')

ax6.scatter(data_use['1stFlrSF'],data_use['SalePrice'])
ax6.set_title('1stFlrSF and SalePrice')

ax7.scatter(data_use['FullBath'],data_use['SalePrice'])
ax7.set_title('FullBath and SalePrice')

ax8.scatter(data_use['TotRmsAbvGrd'],data_use['SalePrice'])
ax8.set_title('TotRmsAbvGrd and SalePrice')

ax9.scatter(data_use['YearBuilt'],data_use['SalePrice'])
ax9.set_title('YearBuilt and SalePrice')


# Looking from the about scatter plots some factors need to be treated for linearity before applying the regression model.
# 1. OverallQual
# 2. GarageArea
# 3. TotalBsmtSF
# 4. 1stFlrSF
# 5. YearBuilt

# As we saw above there was skewness in the SalePrice, which can be treated using the logarithmic transformation. Let's verify the skewness before and after the transformation.

# In[ ]:


data_use['logPrice'] = np.log(data_use['SalePrice'])


# In[ ]:


before = raw_data['SalePrice'].skew()
after = data_use['logPrice'].skew()
print('Skewness before : {}, Skewness after : {}'.format(before,after))


# See, by mere applying the log transformation the skewness got reduced to near 0 i.e. close to Normally Distributed Graph. Also, let's check the density plot for same.

# In[ ]:


sns.distplot(data_use['logPrice'])


# In[ ]:


f, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(1,9, sharey = True, figsize = (30,5))
ax1.scatter(data_use['OverallQual'],data_use['logPrice'])
ax1.set_title('OverallQual and logPrice')

ax2.scatter(data_use['GrLivArea'],data_use['logPrice'])
ax2.set_title('GrLivArea and logPrice')

ax3.scatter(data_use['GarageCars'],data_use['logPrice'])
ax3.set_title('GarageCars and logPrice')

ax4.scatter(data_use['GarageArea'],data_use['logPrice'])
ax4.set_title('GarageArea and logPrice')

ax5.scatter(data_use['TotalBsmtSF'],data_use['logPrice'])
ax5.set_title('TotalBsmtSF and logPrice')

ax6.scatter(data_use['1stFlrSF'],data_use['logPrice'])
ax6.set_title('1stFlrSF and logPrice')

ax7.scatter(data_use['FullBath'],data_use['logPrice'])
ax7.set_title('FullBath and logPrice')

ax8.scatter(data_use['TotRmsAbvGrd'],data_use['logPrice'])
ax8.set_title('TotRmsAbvGrd and logPrice')

ax9.scatter(data_use['YearBuilt'],data_use['logPrice'])
ax9.set_title('YearBuilt and logPrice')


# If not perfectly linear, we are still able to see some improvement in linearity after applying log transformation.

# Now coming to homoscedasticity, we already implemented the log transformation which is the best fix for heteroscedasticity. Thus, we don't need any other fix or tansformation.

# Autocorrelation needs to checked when the data is a time series. Observations here are not coming from a time series or a panel data. These are just the snapshot of current situation, where it's different for each customer.

# Our dataset is almost ready, but before we dive into modelling and predictions let's include the dummy variables for categorical data that we haven't used till now. At the initial stage we included location as a factor determinig the SalePrice of a house.  I'll add the 'Neighborhood' and 'Condition1' columns and create dummy variables.

# In[ ]:


location = data_cleaned[['Neighborhood','Condition1']].copy()
data = data_cleaned[factors]
data_with_cat = data.join(location)
data_with_cat


# In[ ]:


data_with_cat['logPrice'] = np.log(data_with_cat['SalePrice'])
data_with_cat = data_with_cat.drop(['SalePrice'],axis=1)


# In[ ]:


data_with_dummies = pd.get_dummies(data_with_cat, drop_first = True)


# In[ ]:


cols = ['logPrice','OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
        'Neighborhood_Blueste', 'Neighborhood_BrDale',
       'Neighborhood_BrkSide', 'Neighborhood_ClearCr',
       'Neighborhood_CollgCr', 'Neighborhood_Crawfor',
       'Neighborhood_Edwards', 'Neighborhood_Gilbert',
       'Neighborhood_IDOTRR', 'Neighborhood_MeadowV',
       'Neighborhood_Mitchel', 'Neighborhood_NAmes',
       'Neighborhood_NPkVill', 'Neighborhood_NWAmes',
       'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
       'Neighborhood_OldTown', 'Neighborhood_SWISU',
       'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
       'Neighborhood_Somerst', 'Neighborhood_StoneBr',
       'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Feedr',
       'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN',
       'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe',
       'Condition1_RRNn']


# In[ ]:


data_preprocessed = data_with_dummies[cols]


# In[ ]:


data_preprocessed


# In[ ]:


train_input = data_preprocessed.drop(['logPrice'],1)
train_target = data_preprocessed['logPrice']


# In[ ]:


scaler = StandardScaler()
scaler.fit(train_input)


# In[ ]:


train_scaled = scaler.transform(train_input)


# # Preparing our test data

# In[ ]:


df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_test


# In[ ]:


#Checking for duplicate values:
dup_data1 = df_test[df_test.duplicated()]
dup_data1.shape[0]


# In[ ]:


total = pd.isnull(df_test).sum().sort_values(ascending=False)
percentage = ((pd.isnull(df_test).sum() / pd.isnull(df_test).count()).sort_values(ascending=False))*100
no_values = pd.concat([total,percentage],axis = 1, keys = ['Total', 'Percent'])


# In[ ]:


no_values.head(30)


# In[ ]:


df_test1 = df_test.drop((no_values[no_values['Total'] > 1]).index,1)


# In[ ]:


df_test1.shape


# Instead of treating all the missing values, we can just treat the values which are required in the test dataset. 

# In[ ]:


factors_test = [ 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']


# In[ ]:


test_use = df_test1[factors_test].copy()


# In[ ]:


test_use


# In[ ]:


location1 = df_test1[['Neighborhood','Condition1']].copy()
data = df_test1[factors_test]
data_with_cat = data.join(location)
data_with_cat


# In[ ]:


pd.isnull(data_with_cat).sum()


# Now we could have removed these single missing values but the output submission file requires all the rows. Thus, we will treat these values with appropriate method.

# In[ ]:


data_with_cat[['GarageArea','GarageCars','TotalBsmtSF']].describe()


# In[ ]:


data_with_cat['GarageArea'].fillna(data_with_cat['GarageArea'].mean(),inplace=True)
data_with_cat['TotalBsmtSF'].fillna(data_with_cat['TotalBsmtSF'].mean(),inplace=True)
data_with_cat['GarageCars'].fillna(data_with_cat['GarageCars'].median(),inplace=True)


# In[ ]:


data_with_cat['Condition1'].fillna(data_with_cat['Condition1'].mode().values[0],inplace=True)
data_with_cat['Neighborhood'].fillna(data_with_cat['Neighborhood'].mode().values[0],inplace=True)


# In[ ]:


pd.isnull(data_with_cat).sum().max()


# In[ ]:


data_with_dummies = pd.get_dummies(data_with_cat, drop_first = True)


# In[ ]:


data_with_dummies


# In[ ]:


test_data = data_with_dummies.copy()


# In[ ]:


scaler = StandardScaler()
scaler.fit(test_data)


# In[ ]:


test_scaled = scaler.transform(test_data)


# # Random Forest Regression

# In[ ]:


rf_reg = RandomForestRegressor()


# In[ ]:


max_features = ['auto','sqrt','log2']
n_estimators = [ int(x) for x in np.linspace(start=100,stop=1200,num=12)]
oob_score = ['True','False']
min_samples_leaf = [int(x) for x in np.linspace(start=1,stop=6,num=6)]


# In[ ]:


hyper_tune = { 'max_features' : max_features,
               'n_estimators' : n_estimators,
               'oob_score' : oob_score,
               'min_samples_leaf' : min_samples_leaf
    
}


# In[ ]:


rf_search = RandomizedSearchCV(estimator = rf_reg, param_distributions = hyper_tune, n_iter = 10, cv = 5, random_state = 1)


# In[ ]:


rf_search.fit(train_scaled,train_target)


# In[ ]:


rf_search.best_params_


# In[ ]:


rf_search.best_score_


# In[ ]:


rf_search.best_estimator_


# In[ ]:


test_pred = rf_search.predict(test_scaled)


# In[ ]:


output = pd.DataFrame({'ID' : df_test.Id, 'SalePrice' : np.exp(test_pred)})


# In[ ]:


output.to_csv('submission.csv',index=False)


# In[ ]:




