#!/usr/bin/env python
# coding: utf-8

# ## Introduction  
#   
# 
# ![image.png](attachment:image.png)
# 
# Ames Housing Datset includes 80 potential features directly related to property sales. It focuses on the quality and quantity of many physical attributes of the property. Most of the variables are exactly the type of information that a typical home buyer would want to know about a potential property(e.g. When was it built? How big is the lot? etc..). 
# 
# 
# In this version,we are mainly focus on Data Cleaning. In this case, we are dealing with imputing data. Generally, we may tend to drop columns those are having more than 50 percent of data. But, if we would carefully read the document(http://jse.amstat.org/v19n3/decock/DataDocumentation.txt) and get the proper insight of the data, we will find that it can be imputed with appropriate values. 
# 
# Our aim is to clean data but keeping the shape and size of whole dataset same.
# 
# Let's take a look.

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('max_columns',100, 'max_rows',100)
sns.set(context='notebook', style='whitegrid', palette='deep')
from sklearn.impute import KNNImputer


# In[ ]:


from IPython.display import display_html
def disp_side(*args):
    html_str='  '
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# ## Loading Data

# In[ ]:


train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
dataset = pd.concat([train,test],axis = 0,ignore_index =True,sort=False)

train.shape,test.shape,dataset.shape


# In[ ]:


dataset.head()


# ## Data Cleaning
# 
# This dataset contains lots of missing values in many columns. we have applied different strategy for imputing based on data understanding.
# 
# We have observed 34 columns which contains null values. 

# In[ ]:


nullcnt = dataset.isnull().sum().to_frame()
nulldf = nullcnt[nullcnt[0]>0].sort_values(0,ascending=False)
nulldf.drop('SalePrice',axis=0,inplace=True)
print('Number of columns containing null:',nulldf.shape[0])
print('Number of columns containing nulls in 1000s :',(nulldf[0]>1000).sum())
print('Number of columns containing nulls in 100s : ',((1000>nulldf[0])&(nulldf[0] >100)).sum())
print('Number of columns containing nulls in 10s :',((100>nulldf[0]) &(nulldf[0] >10)).sum())
print('Number of columns containing nulls less than 10 :',(nulldf[0]<10).sum())
disp_side(nulldf[:12],nulldf[12:24],nulldf[24:])


# In above list, most of the columns are related to Garage and Basement(i.e.Bsmt). We will analyse columns in each category together. Because some columns may help us to impute other columns.  

# ### Imputing data in Basement related columns

# In[ ]:


bsmcols =  [col for col in dataset.columns if 'Bsmt' in col]
dataset[bsmcols].isnull().sum()


# In above list, TotalBsmtSF(Total Basement in square foot) is key column to impute other columns as well as it only has one missing value. we can set NaNs in categorical as NAv(Not available) and numeric columns to 0.

# In[ ]:


dataset[dataset['TotalBsmtSF'] == 0][bsmcols].head()


# In[ ]:


# if 'TotalBsmtSF' is 0 or not available then apply following strategy
rows = (dataset['TotalBsmtSF'] == 0) | (dataset['TotalBsmtSF'].isnull())
dataset.loc[rows,['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']] = 'NAv'
dataset.loc[rows,['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']] = 0
dataset.loc[rows,['BsmtUnfSF','TotalBsmtSF']] = 0
dataset[bsmcols].isnull().sum()            


# In[ ]:


# Remaining nulls are in categorical or discrete columns. Let's replace it with mode
remain = ['BsmtCond','BsmtExposure','BsmtFinType2','BsmtQual']
modes = dataset[remain].mode().values.tolist()[0]
mapdict = dict(zip(remain,modes))

dataset.fillna(mapdict,inplace=True)


# In[ ]:


dataset[bsmcols].isnull().sum()


# ### Imputing data in Garage related columns:

# In[ ]:


garcols =  [col for col in dataset.columns if 'Garage' in col]
dataset[garcols].isnull().sum()


# Descriptions of columns:
# 
# 
# Garage Type (Nominal): Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# 		
# Garage Yr Blt (Discrete): Year garage was built
# 		
# Garage Finish (Ordinal)	: Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage
# 		
# Garage Cars (Discrete): Size of garage in car capacity
# 
# Garage Area (Continuous): Size of garage in square feet
# 
# Garage Qual (Ordinal): Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# Garage Cond (Ordinal): Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage

# GarageArea and GarageCars has only 1 missing value. Remaining categorical columns has missing value if these two columns are zero. It means Garage is not available fot that corresponding property. So, here is the strategy, If these two columns are zero and other columns are NaN then we put NAv(Not Available) in categorical columns and 0 in 'GarageYrBlt'

# In[ ]:


garcat = ['GarageCond','GarageFinish','GarageQual','GarageType']
rows = (dataset['GarageArea'] == 0) & (dataset['GarageCars'] == 0) & (dataset[garcat].isnull().all(axis=1))
dataset.loc[rows,garcat] = 'NAv'
dataset.loc[rows,'GarageYrBlt'] = 0
dataset[garcols].isnull().sum()


# All above nulls are present in only two rows. Presence of some quantities indicates the availablity of garage in the property.Let's analyze it carefully and impute data accordingly.

# In[ ]:


dataset[dataset[garcols].isnull().any(axis=1)][garcols]


# In[ ]:


# calculating mode for categorical columns with GarageType 'Detached' 
dataset.loc[dataset['GarageType'] == 'Detchd',['GarageCond','GarageFinish','GarageQual']].mode()


# In[ ]:


#  For index 2126 and 2576 we will replace null in following ways

# 'GarageYrBlt' -> 'YearBuilt'
dataset.loc[2126,'GarageYrBlt'] = dataset.loc[2126,'YearBuilt']
dataset.loc[2576,'GarageYrBlt'] = dataset.loc[2576,'YearBuilt']

# categorical and discrete columns -> mode calculated as above
dataset.loc[[2126,2576],['GarageCond','GarageQual']] = 'TA'
dataset.loc[[2126,2576],'GarageFinish'] = 'Unf'
dataset.loc[2576,['GarageCars']] = dataset.loc[dataset['GarageType'] == 'Detchd','GarageCars'].mode().values

# numeric col -> mean
dataset.loc[2576,['GarageArea']] = dataset.loc[dataset['GarageType'] == 'Detchd','GarageArea'].mean()


# In[ ]:


dataset[garcols].isnull().sum()


# ### Imputing data in Remaining Columns

# In[ ]:


null_count = dataset.isnull().sum()
nulldf = null_count[null_count>0]
nulldf.drop('SalePrice',axis = 0)


# Let's focus on following set of columns first
# 
# Alley (Nominal): Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
#        
# Fence (Ordinal): Fence quality
# 		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
#        
# FireplaceQu (Ordinal): Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
#        
#  Mas Vnr Type (Nominal): Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
#        
#  Misc Feature (Nominal): Miscellaneous feature not covered in other categories
# 		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
#        
#  Pool QC (Ordinal): Pool quality
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
#        
#        
# For above columns we have following strategy for filling nulls
# 1. Alley and Fence are null as they are not available. Therefore impute it with NA(Not Available)
# 2. MasVnrType, PoolQC,MiscFeature will be set to NA(Not Available) if fireplaces, PoolArea,MiscVal are 0 as it indicates no availability of that feature for that particular property

# In[ ]:


dataset.fillna({'Alley':'NAv','Fence':'NAv'},inplace = True)
dataset.loc[dataset['MasVnrArea'] == 0,'MasVnrType'] = 'NAv'
dataset.loc[dataset['Fireplaces'] == 0,'FireplaceQu'] = 'NAv'
dataset.loc[dataset['MiscVal'] == 0,'MiscFeature'] = 'NAv'
dataset.loc[dataset['PoolArea']==0,'PoolQC'] = 'NAv'


# In[ ]:


null_count = dataset.isnull().sum()
nulldf = null_count[null_count>0]
nulldf.drop('SalePrice',axis = 0)


# In[ ]:


# Lets replace nulls in remainin categorical and discrete columns with mode value. 
# These column's null count are less than 5.
remain_cols = ['Electrical','Exterior1st','Exterior2nd','Functional','KitchenQual','MiscFeature','PoolQC','SaleType','Utilities']
modes = dataset[remain_cols].mode()
mapdict = dict(zip(remain_cols,modes))
dataset.fillna(mapdict,inplace=True)


# 'MasVnrType' and 'MasVnrArea' not yet imputed completely. Let's carefully look at its description.
# 
# Mas Vnr Type (Nominal): Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
# 	
# Mas Vnr Area (Continuous): Masonry veneer area in square feet
# 
# It could be possible that Masonary Veneer may not be present for particular propery if MasVnrType is None. Based on this knowledge we would replace null with NAv for MasVnrType and with 0 for MasVnrArea 

# In[ ]:


dataset[dataset[['MasVnrArea','MasVnrType']].isnull().any(axis=1)][['MasVnrArea','MasVnrType']]


# In[ ]:


rows = dataset[['MasVnrType','MasVnrArea']].isnull().all(axis=1)
dataset.loc[rows,['MasVnrType','MasVnrArea']] = 0
# row 2610 where MasVnr is present
dataset.at[2610,'MasVnrType'] = dataset['MasVnrType'].mode()


# 
# ### Dealing with nulls in MSZoning:
# 
# MSZoning indicates the zone(commercial/residential/agriculture..etc) under which property comes. It could have relation with neighborhood.To understand this, we have generated pivot table in which columns are neighborhood values and indexes are MSzoning values. Each cell in this df indicates count(no. of properties) for corresponding zone and neighborhood. 
# 
# If we will look at following df, each neighborhood column has particular MSZoning with very high frequency. Our current misssing MSzoning contains neighborhood 'IDOTRR' and 'Mitchel' so based on below df. We would replace corresponding MSZoning with 'RM' and 'RL' respectively.

# In[ ]:


pd.crosstab(dataset['MSZoning'],dataset['Neighborhood'])


# In[ ]:


dataset[dataset['MSZoning'].isnull()][['MSZoning','Neighborhood']]


# In[ ]:


dataset.fillna({'MSZoning':'RM'},inplace=True)
dataset.at[2904,'MSZoning'] = 'RL'


# ### Dealing with nulls in LotFrontage:
# LotFrontage is numeric column contains 486 missing values. We must be extra careful to impute such large number of values. Setting up constant value may affect seriously on performance of our model. Removing rows may cause unnecessary reduction of our train data.
# 
# We have two choices: first,remove column and second, impute data with appropriate value still mainitaining the distribution shape same. We would not go with first choice as this column has considerable correlation with 'SalePrice'.Let's think of second choice. We will use KNN imputer which uses k nearest neighbors to impute missing value.

# In[ ]:


before = dataset['LotFrontage'].copy()
cormat = dataset.corr()['LotFrontage']
# cormat.drop(['SalePrice','LotFrontage'],axis = 0,inplace=True)
cormat_before = cormat[cormat> 0.3].to_frame()


cormat_before


# In[ ]:



features = dataset.select_dtypes(np.number).columns.drop('SalePrice')

imputer = KNNImputer(n_neighbors=5,weights = 'uniform')
dataset[features] = imputer.fit_transform(dataset[features])


# In[ ]:


cormat = dataset.corr()['LotFrontage']
# cormat.drop(['SalePrice','LotFrontage'],axis = 0,inplace=True)
cormat_after = cormat[cormat> 0.3].to_frame()

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
sns.distplot(before,ax=ax1)
sns.distplot(dataset['LotFrontage'],ax=ax2)

disp_side(cormat_before,cormat_after)


# We have succeded in imputing these large amount of data while maintained significant correlation matrix(corr>0.3) and the original distribution shape almost same as before.

# ### Seperating train and test data:

# In[ ]:


cleaned_train = dataset[:1460].copy()
cleaned_test = dataset[1460:].copy()
cleaned_test.drop('SalePrice',axis=1,inplace=True)


# In[ ]:


nulls = cleaned_train.isnull().sum()
nulls[nulls>0]


# In[ ]:


nulls = cleaned_test.isnull().sum()
nulls[nulls>0]


# In[ ]:


# Storing for future use
cleaned_train.to_csv('ctrain.csv',index = False)
cleaned_test.to_csv('ctest.csv',index=False)


# This cleaned data will be used further for analysis and predictive modelling. In the next section we will focus on EDA and Feature Engineering
