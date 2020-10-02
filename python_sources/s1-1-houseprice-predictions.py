#!/usr/bin/env python
# coding: utf-8

# <h5> To understand this dataset properly, we must have a proper grasp on the details provided by the author at https://ww2.amstat.org/publications/jse/v19n3/decock.pdf .<br><br>
# Just as the author of the dataset states, we shall construct a simple model(Model1), then a complex model(Model2) all the while trying to adhere to the guidelines specified in the file.<br><br>
# Next we shall move to stacking and ensembling, as and when needed. </h5>

# <h2>Importing all the necessary libraries.</h2>

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import (StratifiedKFold, KFold, cross_validate)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns= None
pd.options.display.max_rows= None
np.set_printoptions(suppress=True)
print('All libraries imported.')


# <h4>This dataset contains 80 columns originally. As we shall see later, many of them have missing values for lack of missing features in the house itself. Hence, we should not remove missing values initially for this dataset, because that will result in loss of important information. Instead, we can encode these values as YN features: if values are missing, then they do not have that feature and vice versa.</h4>

# In[ ]:


train_df= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col= 'Id')
test_df= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col= 'Id')
df= pd.concat([train_df, test_df], axis= 0)
print(len(df.columns), ' columns:')
print(df.columns) # lots of columns present


# Finding out the total number of missing values and rows just for reference purposes:

# In[ ]:


print(df.isna().sum().sum(), ' total missing values.')
print(len(df), ' is the number of rows.')


# In[ ]:


print('Column\t\t\tDtype\t\t\tMissing\t\tMissing%')
l= len(df)
for i in df.columns:
    n= df[i].isna().sum()
    print(i, '\t\t', df[i].dtype, '\t\t', n, '\t\t', (n*100)/l)


# **The column 'Neighborhood' is an interesting feature; it is a categorical variable, each category corresponding to neighborhoods within Ames. Since we are not familiar with the neighborhoods, we cannot differentiate between them; however we know neighborhoods do influence house prices. For example, houses in affluent neighborhoods are likely to be more costly than those in poor neighborhoods. We can obtain that information using the average house prices to know which neighborhoods are costly to live in, and which are cheaper.**

# In[ ]:


df[['Neighborhood', 'SalePrice']].groupby(['Neighborhood'], as_index=True).mean().sort_values(by='SalePrice', ascending=False)


# <h2><u>Encoding the categorical variables:</u></h2>
# We simply replace the categorical variables with numerical values corresponding to their categorical values. We could use LabelEncoder() for this purpose, but we want to give lower values to lower categories & vice versa. We choose to do this manually.

# In[ ]:


df.replace({
    'BsmtCond': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'BsmtQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'CentralAir': {'N': 0, 'Y': 1},
    'ExterQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'ExterCond': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'Functional': {'Typ':3, 'Min1':2, 'Min2':2, 'Mod':2, 'Maj1':1, 'Maj2':1, 'Sev':0, 'Sal':0},
    'HeatingQC': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'HouseStyle': {'1Story':1, '1.5Fin':1, '1.5Unf':1, '2Story':2, '2.5Fin':2, '2.5Unf':2, 'SFoyer':0, 'SLvl':0},
    'KitchenQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},
    'Street': {'Pave':1, 'Grvl':0},
    'PavedDrive': {'Y':1, 'P':1, 'N':0}
}, inplace= True)


# In[ ]:


df.replace({
    'Neighborhood': {'NoRidge':3, 'NridgHt':3, 'StoneBr':3, 'Timber':2, 'Veenker':2, 'Somerst':2, 'ClearCr':2,
                     'Crawfor':2, 'CollgCr':2, 'Blmngtn':2, 'Gilbert':2, 'NWAmes':2, 'SawyerW':2, 'Mitchel':1,
                     'NAmes':1, 'NPkVill':1, 'SWISU':1, 'Blueste':1, 'Sawyer':1, 'OldTown':1, 'Edwards':1,
                     'BrkSide':1, 'BrDale':1, 'IDOTRR':1, 'MeadowV':1}
}, inplace= True)


# <h2><u>Some simple feature engineering</u></h2>
# Age of the house may be relevant. House prices generally decrease with increased age.

# In[ ]:


df['Age']= ' '
df['Age']= df['YrSold']-df['YearRemodAdd']


# If the house has alley access or not. People generally prefer houses with direct alley access.

# In[ ]:


df['AlleyAccess']= ' '
df['AlleyAccess'][df['Alley'].isna()]= 0 
df['AlleyAccess'][df['Alley'].notna()]= 1


# Uniting the number of bathrooms into a single variable:

# In[ ]:


df['Bathrooms']= ' '
df['Bathrooms']= (2*df['BsmtFullBath'])+df['BsmtHalfBath']+(2*df['FullBath'])+df['HalfBath']


# What is the condition of the house basement. Creating the rating from both 'BsmtQual' and 'BsmtCond'

# In[ ]:


df['BsmtRating']= ' '
df['BsmtRating']= df['BsmtQual']*df['BsmtCond']
df['BsmtRating'][df['BsmtRating'].isna()]= 0


# If the house has basement or not.

# In[ ]:


df['BsmtYN']= ' '
df['BsmtYN'][df['BsmtQual'].isna()]= 0
df['BsmtYN'][df['BsmtQual'].notna()]= 1
df['BsmtFinSF1'][df['BsmtQual'].isna()]= 0
df['BsmtFinSF1'][df['BsmtQual'].isna()]= 0
df['BsmtUnfSF'][df['BsmtQual'].isna()]= 0
df['TotalBsmtSF'][df['BsmtQual'].isna()]= 0


# Calculating the total square feet area of the house into a single variable. House prices increase with the increase in total square feet.

# In[ ]:


df['TotalSqFt']= ' '
df['TotalSqFt']= df['1stFlrSF']+df['2ndFlrSF']+df['LowQualFinSF']+df['TotalBsmtSF']+df['BsmtFinSF2']+df['BsmtFinSF1']+df['GrLivArea']+df['BsmtUnfSF']


# **Imputing the missing values with their proper values. If values are missing, then they should have a zero value because we have already quantified the other categories.**

# In[ ]:


df['BsmtCond'][df['BsmtYN']==0]= 0
df['BsmtCond'][df['BsmtCond'].isna()]= 0
df['BsmtQual'][df['BsmtYN']==0]= 0
df['Functional'][df['Functional'].isna()]= 0
df['GarageArea'][df['GarageArea'].isna()]= df['GarageArea'].mean()
df['KitchenQual'][df['KitchenQual'].isna()]= 2 # imputing with mode value
df['TotalSqFt'][df['TotalSqFt'].isna()]= df['TotalSqFt'].mean()
df['Bathrooms'][df['Bathrooms'].isna()]= 4 # imputing with mode value


# In[ ]:


df['Cond1']= ' '
df['Cond1'][df['Condition1']=='Norm']= 0
df['Cond1'][df['Condition1']!='Norm']= 1
df['Cond2']= ' '
df['Cond2'][df['Condition2']=='Norm']= 0
df['Cond2'][df['Condition2']!='Norm']= 1
df['Condition']= df['Cond1']+df['Cond2']


# If house has fence or not. This is important because people with pets prefer fenced houses

# In[ ]:


df['FenceYN']= ' '
df['FenceYN'][df['Fence'].isna()]= 0 
df['FenceYN'][df['Fence'].notna()]= 1


# If house has fireplaces or not.

# In[ ]:


df['FireplaceYN']= ' '
df['FireplaceYN'][df['FireplaceQu'].isna()]= 0 
df['FireplaceYN'][df['FireplaceQu'].notna()]= 1
df['Fireplaces'][df['FireplaceQu'].isna()]= 0 


# If house has garage or not. People with multiple cars will generally prefer houses wth garage. They are also more likely to be rich, being able to afford higher house prices. This is a good deciding factor.

# In[ ]:


df['GarageYN']= ' '
df['GarageYN'][df['GarageType'].isna()]= 0
df['GarageYN'][df['GarageType'].notna()]= 1
df['GarageArea'][df['GarageType'].isna()]= 0


# If house has veneer or not.

# In[ ]:


df['MasVnrYN']= ' '
df['MasVnrYN'][df['MasVnrType'].isna()]= 0
df['MasVnrYN'][df['MasVnrType'].notna()]= 1
df['MasVnrArea'][df['MasVnrType'].isna()]= 0


# If house has various amenities(specified in the author's pdf file) or not.

# In[ ]:


df['Amenities']= ' '
df['Amenities'][df['MiscFeature'].isna()]= 0
df['Amenities'][df['MiscFeature'].notna()]= 1
df['Amenities'][df['MiscFeature'].isna()]= 0


# In[ ]:


df['PavedYN']= ' '
df['PavedYN']= (2*df['Street'])+df['PavedDrive']


# Presence of pool is also likely to be a deciding factor. Houses with pools generally command higher prices, and are found in rich neighborhoods.

# In[ ]:


df['PoolYN']= ' '
df['PoolYN'][df['PoolQC'].isna()]= 0
df['PoolYN'][df['PoolQC'].notna()]= 1
df['PoolArea'][df['PoolQC'].isna()]= 0


# In[ ]:


df['PorchArea']= ' '
df['PorchArea']= df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch']


# In[ ]:


df['Rating']= ' '
df['Rating']= df['OverallQual']*df['OverallCond']


# Imputing some more missing values.

# In[ ]:


df['Utilities'][df['Utilities'].isna()]= 'AllPub' # imputing with mode value

df['Gas']= ' '
df['Water']= ' '
df['Septic']= ' '

df['Gas'][df['Utilities']=='AllPub']= 1
df['Water'][df['Utilities']=='AllPub']= 1
df['Septic'][df['Utilities']=='AllPub']= 1

df['Gas'][df['Utilities']=='NoSewr']= 1
df['Water'][df['Utilities']=='NoSewr']= 1
df['Septic'][df['Utilities']=='NoSewr']= 0

df['Gas'][df['Utilities']=='NoSeWa']= 1
df['Water'][df['Utilities']=='NoSeWa']= 0
df['Septic'][df['Utilities']=='NoSeWa']= 0

df['Gas'][df['Utilities']=='ELO']= 0
df['Water'][df['Utilities']=='ELO']= 0
df['Septic'][df['Utilities']=='ELO']= 0


# **Dropping the unnecessary columns. Please note that before dropping them, we have retained information from each of these columns in our engineered columns, so we are not really losing any information. Retaining these columns will simply be an overhead for us at this point, and contribute to multicollinearity.**

# In[ ]:


df.drop(['1stFlrSF','2ndFlrSF','Alley','BsmtFinSF1', 'BsmtFinSF2','BsmtFinType1','BsmtFinType2','BsmtFullBath','BsmtHalfBath',
         'BsmtUnfSF','Fence','FireplaceQu','FullBath','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType',
         'GarageYrBlt','GrLivArea','HalfBath','PoolQC', 'Condition1', 'Condition2', 'BedroomAbvGr', 'YrSold', 'YearBuilt', 
        'Cond1', 'Cond2', 'Electrical', 'Street', 'PavedDrive', 'SaleType', 'SaleCondition', 'OverallQual'], axis= 1, inplace= True)


# In[ ]:


df.drop(['LandContour','LandSlope', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'RoofStyle', 'RoofMatl',
        'MoSold','TotalBsmtSF', 'BldgType', 'BsmtExposure', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating',
        'KitchenAbvGr','LotFrontage', 'LotShape', 'LotConfig', 'LowQualFinSF', 'MSZoning', 'MasVnrType', 'MiscFeature',
         'OverallCond', 'YearRemodAdd', 'Utilities'], axis= 1, inplace= True)


# Separating the whole dataframe into training & testing dataframes:

# In[ ]:


train_df= df[df['SalePrice'].notna()]
test_df= df[df['SalePrice'].isna()]


# **2 data points were found to be outliers in the scatterplot; those 2 observations had total square feet area of more than 15000 sqft with prices less than 400000. Those 2 can be explained by either anomalies in the dataset, or deeply discounted sales. Either way, we remove them; removal of only 2 data points is not likely to affect our regression model adversely.**

# In[ ]:


train_df.drop(train_df[train_df['TotalSqFt']>15000].index, inplace= True)
sns.scatterplot(x= train_df['TotalSqFt'], y= train_df['SalePrice'])


# **    Earlier, we constructed a naive linear model using all the original featues present in the dataset to predict the prices. The naive model worked surprisingly well, with an r2 value of 0.80. However, we chose not to stick with it, and wanted to see if it could be increased any more. Hence all these feature engineering steps from simple intuition to construct models 1 & 2.**

# <h2><U>Model 1</u></h2>

#     The author states in his documentation that "about 80% of the variation in residential sales price can be explained by simply taking into consideration the neighborhood and total square footage (TOTAL BSMT SF+ GR LIV AREA) of the dwelling." Here we try to reproduce the same, using only those 2 featues, and end up with the following results:

# In[ ]:


lr= LinearRegression()
skf= StratifiedKFold(n_splits= 10, shuffle= True)
result= cross_validate(lr, train_df[['Neighborhood', 'TotalSqFt']], train_df['SalePrice'], cv= skf, scoring= ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'])
print('Model 0b:\tMAE: %.8f\t\tMSE: %.6f\t\t\tR2: %.6f'%(result['test_neg_mean_absolute_error'].mean(), result['test_neg_mean_squared_error'].mean(), result['test_r2'].mean()))


# **The Model1 yields a r2 value of 0.77, which is well, adequate. Even though we could not obtain the author's claimed 80%, we came close. We move on to our next model.**

# <h2><U>Model 2</u></h2>

# In[ ]:


lr= LinearRegression()
skf= StratifiedKFold(n_splits= 10, shuffle= True)
result= cross_validate(lr, train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'], cv= skf, scoring= ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'])
print('Model 0b:\tMAE: %.8f\t\tMSE: %.6f\t\t\tR2: %.6f'%(result['test_neg_mean_absolute_error'].mean(), result['test_neg_mean_squared_error'].mean(), result['test_r2'].mean()))


# **Our simple linear model Model2 yields a r2 value of 0.84, that is, we are able to explain almost 84% of the total variance in the dataset using our engineered features. This is a high enough score for our simple linear model, so we stick with it. We apply this model to predict the prices for the test set.**

# In[ ]:


train_df= train_df.astype(np.float64)
test_df= test_df.astype(np.float64)
lgbm = LGBMRegressor(objective='regression', 
       num_leaves=5, #was 3
       learning_rate=0.01, 
       n_estimators=11000, #8000
       max_bin=200, 
       bagging_fraction=0.75,
       bagging_freq=5, 
       bagging_seed=7,
       feature_fraction=0.4, # 'was 0.2'
)
lgbm.fit(train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'], eval_metric='rmse')
pred= lgbm.predict(test_df.drop(['SalePrice'], axis= 1))


# In[ ]:


lr= LinearRegression()
skf= StratifiedKFold(n_splits= 10, shuffle= True)
lr.fit(train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'])
pred= lr.predict(test_df.drop(['SalePrice'], axis= 1))
pred


# In[ ]:


sub= pd.DataFrame({
    "Id": test_df.index,
    "SalePrice": pred
})
sub.to_csv('houseprice1.csv', index= False)


# In[ ]:




