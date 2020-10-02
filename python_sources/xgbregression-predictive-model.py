#!/usr/bin/env python
# coding: utf-8

# ## Kaggle House Price Prediction Competition
# 
# In this notebook we tackle the Ames, Iowa housing data set. We first clean the data to remove any NaNs that are present before moving on to exploratory data analysis. The final section selects and tunes machine learning models to be deployed on the test set.
# 
# I have uploaded a stripped down version of this notebook cotaning only the NaN removal component for others to use if they wish to save themselves time cleaning the data.
# * **[Ames, Iowa NaN removal](https://www.kaggle.com/ricksoc/nan-processing-for-ames-iowa-dataset)**

# In[ ]:


# Import required packages
import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
np.random.seed(27)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']   
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.color_palette('muted')
current_palette = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import mode, kstest, levene, boxcox, f_oneway

from mlxtend.plotting import plot_confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.regressor import LinearRegression as ml_lr

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as sk_lr
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, DMatrix


# In[ ]:


#Import data and take a copy for experimenting during exploration

# Import data when working on local machine
# test = pd.read_csv('test.csv')
# train = pd.read_csv('train.csv')

# Import data for Kaggle notebook
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_id = test['Id'] # save id column for indexing final submission
house_test = test.copy()
# house_test.drop(['Id'],inplace=True,axis=1)

train_id = train['Id']
house_train = train.copy()
house_train.drop(['Id'],inplace=True,axis=1)


# Having loaded the data I will run some basic examinations and descriptive statistics to get a feel for the data before examining what NaNs are present and fixing them.

# In[ ]:


# Look at the head of the training data frame to see what an observation looks like in this data set
house_train.head()


# In[ ]:


# Inspect the numerical columns of the training and test sets to get a feel for the shape of the data
print('Descriptive Statistics for the training data set')
display(house_train.describe().T)

print('\n Descriptive Statistics for the test data set')
display(house_test.describe().T)


# Now I will identify which features in the training and test sets have NaNs present. The documentation provided describing what each feature in the data set means will be useful in understanding which columns might be expected to contain NaNs.
# * [Dataset documentation](https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt)

# In[ ]:


print('Features containing NaNs in test set = {}'.format(house_test.isna().any().sum()))
print('Features containing NaNs in training set = {}'.format(house_train.isna().any().sum()))

na_col_train = house_train.isna().any()
train_na = house_train.loc[:,na_col_train].isna().sum()

na_col_test = house_test.isna().any()
test_na = house_test.loc[:,na_col_test].isna().sum()

display(pd.DataFrame([train_na,test_na],index=['train','test']).T)


# ### NaN removal explained
# I have not set out to create an exhaustive process for removing all NaNs from any data set using these features. Rather I have looked to tackle the NaNs missing from these specific training and test sets but generalised where possible.
# 
# #### Starting Point
# 18 features in the training set contain NaNs, 33 features in the test set contain NaNs.
# 
# ##### Lot Frontage
# 
# All dwellings are houses so should have some frontage, i.e. there are no flats which would not have any street connected to the property. I will impute this with the mean frontage for the Neighbourhood of the house.
# 
# ##### Alley 
# 
# NaN for this feature is defined in data description as meaning there is no alley access -> change to "None".

# In[ ]:


# Inspect Mason Veneer Area and Type NaNs
display((house_train.loc[(house_train['MasVnrArea'].notna()) & (house_train['MasVnrType'].isna())                         ,['MasVnrArea','MasVnrType']]).isna().any().sum())
display((house_test.loc[(house_test['MasVnrArea'].notna()) & (house_test['MasVnrType'].isna())                         ,['MasVnrArea','MasVnrType']]).isna().any().sum())

display(house_test.loc[(house_test['MasVnrArea'].notna()) & (house_test['MasVnrType'].isna())])


# ##### Mason Venner Type and Area
# 
# MasVrnType and MasVrnArea have multiple instances where the Area is 0 in which case the Type NaN should be set to 'None'. There is one instance in the test set where an Area is given but no type. This is for record 1150. Inspecting this record the house is made of plywood and the most common Veneer type for Plywood houses when they have one is "BrkFace".

# In[ ]:


# Inspect single observation with NaN for total basement area
display(house_test.loc[house_test['TotalBsmtSF'].isna()])

#Investigate missing BsmtQual values
display(house_test[(house_test['BsmtQual'].isna()) & (house_test['TotalBsmtSF']>0)])
display(house_train.loc[house_train['Neighborhood']=='IDOTRR'].groupby(['MSZoning','BsmtQual']).agg({'BsmtQual':'count','MSZoning':'count'}))

#Inspecting training set observation with Bsmt2 area but not finish type
display(house_train.loc[house_train['BsmtFinType1']=='GLQ']['BsmtFinType2'].value_counts())


# ##### Basement Variables
# Test observation 660 has a NaN for TotalBsmtSF. Examining this record it does not appear that there is a basement. I will therefore set TotalBsmtSF to 0 which will allow the other basement feature processing to occur without error.
# 
# There are two values in the test set where there is a non-zero TotalBsmtSF area recorded but NA BsmtQual. Both houses
# come from the same Neighbourhood and Zoning. Looking at the training set houses with these characteristics have TA BsmtQual
# so I will impute as this.
# 
# Where there is a total basement area greater than 0 but missing basement quality and/ or condition entries these will be assumed as 'TA', which is the coding for typical quality.
# 
# Where the total basement area is zero and qualitative qualitative basement feature values are missing these wil be set to 'None', except for 'BasementExposure' which will be set to 'No' to fit with the convention for this feature.
# 
# If a basement has a valid FinType1 and a NaN for FinType to this will be set to 'Unf'.
# 
# If total basement SF = 0 and BsmtFinSF1 is NaN then BsmtFinSF1 and BsmtUnfSF will be set to 0.
# 
# Missing BsmtFinSF2 will be set to 0.
# 

# In[ ]:


# Inspect missing value from training set Electrical feature
display(house_train.loc[house_train['Electrical'].isna()])


# ##### Electrical
# The house missing this info (training id 1379) has air con which implies it must have electrical. I will impute this as the modal electrical type.
# 
# ##### FireplaceQu
# All NaNs for this feature correspond to the 'Fireplace' feature being 0 so are set to 'None'.

# In[ ]:


# Investigate missing values in Garage features
# Garage Cars and Garage Area both have a single missing value in the test set only
display(house_test.loc[(house_test['GarageCars'].isna()) | (house_test['GarageArea'].isna())])

# Single test record where year built is 2207
display(house_test.loc[house_test['GarageYrBlt']==2207])

# Check other missing garage values correspond to not having a garage
garage_cols = [col for col in test.columns if 'Garage' in col]
garage_cols.remove('GarageArea')
missing_train_garage = house_train[garage_cols].loc[(house_train['GarageArea']>0)].isna().sum()
missing_test_garage = house_test[garage_cols].loc[(house_test['GarageArea']>0)].isna().sum()
display('Training',missing_train_garage)
display('Test',missing_test_garage)


# In[ ]:


ind = house_test[garage_cols].index[(house_test['GarageArea']>0) & (house_test['GarageYrBlt'].isna())]
display(house_test.iloc[ind])


# ##### Garage Variables
# Test observation 1116 appears to have a garage but is missing almost all information about it other than it is detached. I have dealt with this as a special case rather than in the main function. To impute the missing values I have grouped houses by their zone and garage type and worked out the modal value for other garage features and input these into test observation 1116.
# 
# Test record 666 is responsible for the other anomalous data where there is a non-zero garage area but other garage features record NaNs. I will use the function described above to process record 666 also.
# 
# Test observation 1132 has a 'GarageYrBlt' value of 2207. Looking at this record it seems that this should have been 2007 so this has been explicitely corrected.
# 
# Other qualitative garage feature NaNs correspond to houses without garages so have been set to 'None'

# In[ ]:


# Missing pool values
display(house_test.loc[(house_test['PoolArea']>0) & (house_test['PoolQC'].isna())])
display(house_train['PoolQC'].value_counts())


# ##### Pool Quality
# There are some Pools which have an area but no quality value. These have been set to Gd as this is the most common quality in the training set (albeit with a small sample size). All other NaNs are 'None'.
# 
# ##### Fence
# NaNs are assumed as not having a fence so set to 'None'.
# 
# ##### Misc Feature
# NaNs set to 'None'.
# 
# ##### MSZoning
# MSZoning - all hosues must be zoned, impute as modal zone type.
# 
# ##### Utilities
# All houses should have some utility access so this has been imputed as the modal value for the feature.
# 
# ##### Exterior Covering
# I have assumed that houses in the same neighbourhood have similar styles and that all houses have some exterior covering. NaNs have been imputed as the modal type for their neighbourhood.
# 
# ##### Kitchen Quality
# Where a kitchen is present but has a NaN for quality these has been set to typical 'TA'.
# 
# ##### Functionality
# NaNs for this feature have been set to 'Typ' as this fits with the most common entry and likelihood based on feature description.
# 
# ##### Sale Type
# NaNs set to 'WD' as the most likely value.

# In[ ]:


#Process NaNs in training and test data frames

#Deal with Garage NaNs in test record 1116 of the test set
garage_cols = [col for col in test.columns if 'Garage' in col]
garage_cols.remove('GarageType')
garage_groups = house_train.groupby(['MSZoning','GarageType'])[garage_cols].agg(lambda x: mode(x)[0])

def fill_row_na(input_df,row,fill_group):
    '''function to fill in missing values for a particular dataframe row using a groupby object created outside the function'''
    df = input_df.copy() # take copy of data frame so as not to double modify
    zone = df.iloc[row,df.columns.get_loc("MSZoning")]
    gtype = df.iloc[row,df.columns.get_loc('GarageType')]
    fill_group = fill_group.loc[zone,gtype]
    for ind, item in fill_group.iteritems():
        if np.isnan(df.loc[row,ind]):
            df.loc[row,ind] = item
    return df

house_test = fill_row_na(house_test,1116,garage_groups)
house_test = fill_row_na(house_test,666,garage_groups)

#Correct GarageYrBlt = 2207 in test set
house_test.loc[1132,'GarageYrBlt'] = 2007

#Test set record 660 creates a specific problem as it records a NaN for TotalBsmtSF. Setting this to 0 will allow
#the na_processing function below to handle the other NaNs
house_test.loc[house_test['TotalBsmtSF'].isna(),'TotalBsmtSF'] = 0

#One test observation has a veneer area but no type, set this to BrkFace as it best fits the other
house_test.loc[(house_test['Neighborhood']=='Mitchel') & (house_test['MasVnrArea']>0),'MasVnrType'] = 'BrkFace'

def na_processing(input_df,training):
    '''Function for processing remaining NaNs in training and test data sets. Values are either imputed, or set to 0 or None'''
    
    df = input_df.copy() # take copy of dataframe to avoid modifying the original other than with function call
    
    #Lot Frontage
    lot_frontage_fill = training.groupby('Neighborhood').agg({'LotFrontage':'mean'})
    df = df.set_index('Neighborhood')
    df['LotFrontage'].fillna(lot_frontage_fill['LotFrontage'],inplace=True)
    df = df.reset_index()
    
    #Alley
    df['Alley'].fillna('None',inplace=True)
    
    #Masonary Veneer Area and Typr
    df['MasVnrArea'].fillna(0,inplace=True)
    df['MasVnrType'].fillna('None',inplace=True)
    
    #Basement Variables
    df.loc[(df['TotalBsmtSF']>0) & (df['BsmtQual'].isna()) ,'BsmtQual'] = 'TA'
    
    df.loc[(df['TotalBsmtSF']>0) & (df['BsmtCond'].isna()),'BsmtCond'] = 'TA'
    
    df.loc[df['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']] =     df.loc[df['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('None')
    
    df.loc[(df['TotalBsmtSF']>0) & (df['BsmtExposure'].isna()),'BsmtExposure'] = 'No'
    
    df.loc[(df['BsmtFinType1'].notna()) & (df['BsmtFinType2'].isna()),'BsmtFinType2'] = 'Unf'
    
    df.loc[(df['TotalBsmtSF']==0) & (df['BsmtFinSF1'].isna()),['BsmtFinSF1','BsmtUnfSF']] = 0
    
    df['BsmtFinSF2'].fillna(0,inplace=True)
    
    df.loc[:,['BsmtFullBath','BsmtHalfBath']] = df.loc[:,['BsmtFullBath','BsmtHalfBath']].fillna(0)

    #Electrical
    df['Electrical'].fillna(training['Electrical'].mode()[0],inplace=True)
    
    #Fireplace Quality
    df['FireplaceQu'].fillna('None',inplace=True)
    
    #Garage Variables
    garage_cols = [col for col in df.columns if 'Garage' in col]
    garage_cols.remove('GarageCars')
    garage_cols.remove('GarageArea')
    df.loc[:,garage_cols] = df.loc[:,garage_cols].fillna('None')
    
    #Pool Quality
    df.loc[(df['PoolArea']>0) & (df['PoolQC'].isna()),'PoolQC'] = 'Gd'
    df['PoolQC'].fillna('None',inplace=True)
    
    #Fence
    df['Fence'].fillna('None',inplace=True)
    
    #Misc Feature
    df['MiscFeature'].fillna('None',inplace=True)
    
    #MS Zoning
    df['MSZoning'].fillna(training['MSZoning'].mode()[0],inplace=True)

    #Utilities
    df['Utilities'].fillna(training['Utilities'].mode()[0],inplace=True)
    
    #Exterior Covering
    exterior_fill = training.groupby('Neighborhood').agg({'Exterior1st': lambda x: mode(x)[0],                                                        'Exterior2nd': lambda x: mode(x)[0]})
    
    df = df.set_index('Neighborhood')
    df['Exterior1st'].fillna(exterior_fill['Exterior1st'],inplace=True)
    df['Exterior2nd'].fillna(exterior_fill['Exterior2nd'],inplace=True)
    df = df.reset_index()
    
    #Kitchen Quality
    df.loc[(df['KitchenAbvGr']>0) & (df['KitchenQual'].isna()),'KitchenQual'] = 'TA'
    
    #Functionality
    df['Functional'].fillna('Typ',inplace=True)
    
    #Sale Type
    df['SaleType'].fillna('WD',inplace=True)
       
    return df

test_processed = na_processing(house_test,house_train)
train_processed = na_processing(house_train,house_train)


print('Features containing NaNs in test set = {}'.format(test_processed.isna().any().sum()))
print('Features containing NaNs in training set = {}'.format(train_processed.isna().any().sum()))


# In[ ]:


# Save data frames with NaNs removed
test_clean = pd.concat([test_id,test_processed],axis=1)
train_clean = pd.concat([train_id,train_processed],axis=1)

test_clean.to_csv('test_clean.csv')
train_clean.to_csv('train_clean.csv')


# In[ ]:


# Clean up data artefacts created during NaN cleaning
del [na_col_train, na_col_test, train_na, test_na, missing_train_garage, missing_test_garage, garage_groups     , house_train, house_test]


# ### EDA
# Having cleaned the data of NaNs I will now dig deeper to understand which features are likley to be of use in machine learning models (feature selection) and whether there are combinations of, or derivations from, the features which will yield more predictive information, or more efficient predictive information, on the target feature (feature engineering).
# 
# From now on I will observe the discipline of only looking at the training set. I will further split the cleaned training data into a true training set and a validation set to be used for hyperparamter turning.

# In[ ]:


# Create training and validation sets.
house_train = train_clean
house_test = test_clean

X_train, X_valid, y_train, y_valid = train_test_split(house_train.drop('SalePrice',axis=1),                                            house_train['SalePrice'], test_size=0.2, random_state=27)

X_train.drop('Id',inplace=True,axis=1) #drop Id columns as they are not needed
X_valid.drop('Id',inplace=True,axis=1)

display(X_train.shape,X_valid.shape) # Check shape of training and validation data sets

# Use rough eyeball test to check that the validation target set is representative of the training set  
plt.hist(y_train, bins=50, alpha=0.5, label='training')
plt.hist(y_valid, bins=50, alpha=0.5, label='validation')
plt.legend(loc='upper right')
plt.xlabel('Sale Price')
plt.title('Sale Price Distribution in Training and Validation Sets')
plt.show()

# Recombine features and target for EDA and model tuning purposes
df_train = pd.concat([X_train,y_train],axis=1)
df_valid = pd.concat([X_valid,y_valid],axis=1)


# The distribution of the training and validation sets looks similar and looks to have a right skew which should be investigated further. First I will look at how well correlated the numerical features of the data set are with the sale price. The categorical features will require some further processing before they can be checked in the same way.

# In[ ]:


# Look at whether the Sale Price should be log transformed
fig, ax = plt.subplots(1,3,figsize=(20,8))
df_train.hist(column='SalePrice',bins=20,ax=ax[0],color=colours[0])

print('Sale Price Skew = {:.2f}'.format(df_train['SalePrice'].skew()))
print('Sale Price Kurtosis = {:.2f}'.format(df_train['SalePrice'].kurtosis()))

sale_price_log = np.log(df_train['SalePrice'])
ax[1].hist(sale_price_log,bins=20,color=colours[1])
ax[1].set_title('Log Sale Price')
print('Log Skew = {:.2f}'.format(sale_price_log.skew()))
print('Log Kurtosis = {:.2f}'.format(sale_price_log.kurtosis()))

_ = sns.boxplot(y='SalePrice',data=df_train,ax=ax[2],color=colours[2]).set_title('Sale Price')


# The Sale Price target is normally distributed though in its base form is right-tail skewed. Taking the log of Sale Price corrects this so it may help the model to predict log Sale Price and then take the exponential to create the final predictions.
# 
# There are two clear outliers which should probably be removed from the training set before modelling.

# In[ ]:


# Look at how taking the log of the sale price affects feature correlation. Also examine skewness and kurtosis of the features.
df_train['Log_SalePrice'] = np.log(df_train['SalePrice'])
log_correlation = df_train.corr()['Log_SalePrice']
correlation = df_train.corr()['SalePrice'].sort_values(ascending=False)
kurt = df_train.kurtosis()
skew = df_train.skew()

log_train_correlation = pd.concat([correlation,log_correlation,kurt,skew],axis=1)
log_train_correlation.rename(columns={'Log_SalePrice':'Log_Price_Correlation','SalePrice':'Price_Correlation',                                     0:'Kurtosis',1:'Skewness'},inplace=True)
log_train_correlation.drop('Log_SalePrice',inplace=True)

display(log_train_correlation.head(10))


# Taking the log of the Sale Price improves the correlation factor of most nuerical variables, including nine of the top 10, without changing their order. This implies that using the log of the Sale Price may improve model accuracy, particularly in simpler models. I will continue to do base EDA using the Sale Price as this is the real-world value but may use its log in model buidling.

# In[ ]:


# Create dictionary of data types by column in training set
type_dict = {str(k) : list(v) for k,v in X_train.groupby(df_train.dtypes,axis=1)}
for k,v, in type_dict.items():
    print(k, len(v))
display(df_train.loc[:,type_dict['float64']].head(10))

# The features classed as float will not lose an important level of detail by being recast as int64
df_train = df_train.astype({"LotFrontage": np.int64, "MasVnrArea": np.int64})
df_valid = df_valid.astype({"LotFrontage": np.int64, "MasVnrArea": np.int64})
house_test = house_test.astype({"LotFrontage": np.int64, "MasVnrArea": np.int64})


# ### Categorical Feature
# 
# With two distinct groups of features I will first look at the object/ categorical set. While some models, e.g. decision trees, would be able to use these fields in their basic form in order to open up a wider set of models these features will need some processing. Dummy encoding is the most likely option, though it may be possible to map some ordinal fields to numeric values if there is confidence that the resulting intervals would be valid.

# In[ ]:


# Examine distribution of all categorical features
cat_features = pd.melt(df_train.loc[:,type_dict['object']], value_vars=type_dict['object'])
fig = sns.FacetGrid(cat_features, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
fig = fig.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in fig.axes.flat]
fig.fig.tight_layout()
plt.show()


# Dummy encode each categorical feature in turn then measure the correlation with the sale price. This will tell us which features under a univariate analysis are useful predictors.

# In[ ]:


# Look at feature category correlations with Log_SalePrice
df_features = df_train.loc[:,type_dict['object']]
max_correlation = []

for col in df_features.columns:
    # set up dummy variables for each categorical feature column in sequence
    dummies = pd.get_dummies(df_train[col])
    dummies = pd.concat([df_train['Log_SalePrice'],dummies],axis=1)
    # find the maximum correlation of any categoriy in the feature
    max_corr = dummies.corr()['Log_SalePrice'][1:].abs().max()
    max_correlation.append(max_corr)
    

feature_correlations = pd.DataFrame({'Features':df_features.columns,'Correlation':max_correlation})
feature_correlations = feature_correlations.loc[feature_correlations['Correlation']>0.3]        .sort_values(['Correlation'],ascending=False)

feature_correlations


# #### ANOVA
# As well as how well correlated an individual categorical feature is we can also test to see how much of the variance in Sale Price each categorical feature explains. Combining these two insights will point to which features are most likely to be useful and how to process them.

# In[ ]:


anova = {'feature':[], 'f':[], 'p':[]}
for feature in df_train.select_dtypes(include=['object']).columns: #better way of selecting features by dtype
    category_prices = []
    for category in df_train[feature].unique():
        category_prices.append(df_train[df_train[feature] == category]['SalePrice'].values)
    f, p = f_oneway(*category_prices)
    anova['feature'].append(feature)
    anova['f'].append(f)
    anova['p'].append(p)
anova = pd.DataFrame(anova)
anova.sort_values('p', inplace=True)

# Plot
plt.figure(figsize=(14,6))
sns.barplot(anova['feature'], np.log(1/anova['p']))
plt.xticks(rotation=90)
plt.show()

_ = sns.barplot(feature_correlations['Correlation'],feature_correlations['Features'], orient='h')


# Neighborhood, ExterQual, KitchenQual, Foundation, HeatingQC and MasVnrType all have a significant p-value and a single categoriy with a relatively high correlation to sale price.

# #### Exterior Quality and Kitchen Quality
# The Exterior Quality feature has four possible categories and the highest single correlated observation type of any categorical feature.

# In[ ]:


# Examine correlations of each observation type by dummy enociding
ext_dummies = pd.get_dummies(df_train['ExterQual'])
ext_dummies = pd.concat([df_train['Log_SalePrice'],ext_dummies],axis=1)
ext_corr = ext_dummies.corr()['Log_SalePrice'][1:]

display(ext_corr, df_train['ExterQual'].value_counts())


# The categories in ExterQual are not evenly soaced in terms or correlation so shouldn't just be interval encoded. TA and Gd dominate and interestingly Ex has a lower positive correlation than Gd. I will try the approach of grouping the positively and negatively correlation types together.

# In[ ]:


# Group ExterQuals into only two categories and recheck correlation.
df_train['ExterQualRed'] = df_train['ExterQual'].replace({'Ex':1,'Gd':1,'Fa':0,'TA':0})

ext_qual = df_train[['Log_SalePrice','ExterQualRed']]
ext_corr = ext_qual.corr()['Log_SalePrice'][1:]

display(ext_corr)


# In[ ]:


# Repeat analysis above for KitchenQual
ext_dummies = pd.get_dummies(df_train['KitchenQual'])
ext_dummies = pd.concat([df_train['Log_SalePrice'],ext_dummies],axis=1)
ext_corr = ext_dummies.corr()['Log_SalePrice'][1:]

display(ext_corr, df_train['KitchenQual'].value_counts())

# Group ExterQuals into only two categories and recheck correlation.
df_train['KitchenQualRed'] = df_train['KitchenQual'].replace({'Ex':1,'Gd':1,'Fa':0,'TA':0})

ext_qual = df_train[['Log_SalePrice','ExterQualRed','KitchenQualRed']]
ext_corr = ext_qual.corr()

display(ext_corr)


# ExterQual and KitchenQual can both be usefully processed into a binary feature with good correlation, though with some risk of adversely influencing some low frequency cases. 
# 
# However, they are highly correlated with each other so only one of them should be included. I will choose ExterQual as it has the higher correlation and lower p-value.

# #### Foundation
# Foundation has six categories though the value counts are heavily dominated by two of them. It should hopefully be able to again reduce which of the categories need to be considered.

# In[ ]:


# Examine correlations of each observation type by dummy enociding
ext_dummies = pd.get_dummies(df_train['Foundation'])
ext_dummies = pd.concat([df_train['Log_SalePrice'],ext_dummies],axis=1)
ext_corr = ext_dummies.corr()['Log_SalePrice'][1:]

display(ext_corr, df_train['Foundation'].value_counts())


# The CBlock foundation type has a stronger correlation than any other category so I will binary encode this feature with 1 for CBlock else 0.

# #### FireplaceQu, MasVrnType, Neighborhood

# In[ ]:


plt.figure(figsize=(20,12))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

_ = sns.swarmplot(x='FireplaceQu',y='SalePrice',data=df_train,ax=ax1)
_ = sns.swarmplot(x='MasVnrType',y='SalePrice',data=df_train,ax=ax2)
_ = sns.swarmplot(x='Neighborhood',y='SalePrice',data=df_train,ax=ax3)


# The primary difference in FireplaceQu and MasVnrType seems to be around whether or not the value is recorded as 'None' so I will create a new feature to capture this.
# 
# For neighbourhood there is too much going on and given it's low p-value I will just create a full dummy encoding.
# 
# I will now look at these features for multicolineraity.

# #### Garage Type and Garage Finish

# In[ ]:


# Use dummy encoding to look at the correlation of each category in garage type with the log of sale price
gartype_dummies = pd.get_dummies(df_train['GarageType'])
gartype_dummies = pd.concat([df_train['Log_SalePrice'],gartype_dummies],axis=1)
gartype_corr = gartype_dummies.corr()['Log_SalePrice'][1:]

# Display correlations sorted from most negative to most positive
display(gartype_corr.sort_values(),gartype_corr.sort_values().index)

# Create an ordinal encoding of GarageType to see how well the correlates with sale price
GarTypes = gartype_corr.sort_values().index
GarTypeOrd = [num for num in range(len(GarTypes))]
GarType_dict = dict(zip(GarTypes,GarTypeOrd))
df_train['GarageType_Ordinal'] = df_train['GarageType'].map(GarType_dict)


# In[ ]:


garfin_dummies = pd.get_dummies(df_train['GarageFinish'])
garfin_dummies = pd.concat([df_train['Log_SalePrice'],garfin_dummies],axis=1)
garfin_corr = garfin_dummies.corr()['Log_SalePrice'][1:]

# Display correlations sorted from most negative to most positive
display(garfin_corr.sort_values(),garfin_corr.sort_values().index)

# Create an ordinal encoding of GarageType to see how well the correlates with sale price
GarFins = garfin_corr.sort_values().index
GarFinOrd = [num for num in range(len(GarFins))]
GarFin_dict = dict(zip(GarFins,GarFinOrd))
df_train['GarageFinish_Ordinal'] = df_train['GarageFinish'].map(GarFin_dict)

GarageOrdinal_corr = df_train[['Log_SalePrice','GarageFinish_Ordinal','GarageType_Ordinal']].corr()

display(GarageOrdinal_corr)


# The ordinal encodings of both the garage finish and type have useful predictive power but are also correlated. Therefore I will keep the garage finish feature as it has a slightly higher correlation with the log of the sale price.

# In[ ]:


#Check new binary features for multicolinearity
df_train['PConc'] = np.where(df_train['Foundation']=='PConc',1,0)
df_train['HasFireplace'] = np.where(df_train['FireplaceQu']=='None',0,1)
df_train['MasVnrRed'] = np.where(df_train['MasVnrType']=='None',0,1)

ext_qual = df_train[['Log_SalePrice','ExterQualRed','KitchenQualRed','PConc','HasFireplace','MasVnrRed']]
ext_corr = ext_qual.corr()
display(ext_corr[1:])


# In[ ]:


#Clean up added feature. Final processing will be done on main feature column feature
df_train.drop(['ExterQualRed','KitchenQualRed','PConc','HasFireplace','MasVnrRed'],axis=1,inplace=True)


# ExterQual, KitchenQual and having a PConc foundation are more highly correlated with each other than with the Sale Price. I will therefore only include one of these in the final model. I will choose ExterQual as it has the lowest p-value and highest correlation.

# ### Categorical Feature Processing
# Having identified the most promising categorical features to use as predictors these now need to be processed into a useable ML format in the trainig, test and validation dataframes.

# In[ ]:


def create_cat_features(df):
    df['ExterQual'] = df['ExterQual'].replace({'Ex':1,'Gd':1,'Fa':0,'TA':0}).astype(np.int64)
    df['HasFireplace'] = np.where(df['FireplaceQu']=='None',0,1).astype(np.int64)
    df['MasVnr'] = np.where(df['MasVnrType']=='None',0,1).astype(np.int64)
    
    df_dummies = pd.get_dummies(df['Neighborhood'],prefix='Neighbourhood')
    df = pd.concat([df,df_dummies],axis=1)
    
    garfin_dummies = pd.get_dummies(df_train['GarageFinish'])
    garfin_dummies = pd.concat([df_train['Log_SalePrice'],garfin_dummies],axis=1)
    garfin_corr = garfin_dummies.corr()['Log_SalePrice'][1:]

    # Create an ordinal encoding of GarageType to see how well the correlates with sale price
    GarFins = garfin_corr.sort_values().index
    GarFinOrd = [num for num in range(len(GarFins))]
    GarFin_dict = dict(zip(GarFins,GarFinOrd))
    df['GarageFinish_Ordinal'] = df['GarageFinish'].map(GarFin_dict)
    
    return df

df_train = create_cat_features(df_train)
df_valid = create_cat_features(df_valid)
house_test = create_cat_features(house_test)


# ### Numerical Features
# 

# In[ ]:


# Examine distribution of all numerical features
num_features = pd.melt(df_train.select_dtypes(include='int64'), value_vars=type_dict['int64'])
fig = sns.FacetGrid(num_features, col='variable', col_wrap=4, sharex=False, sharey=False)
fig = fig.map(plt.hist,'value',bins=30)


# In[ ]:


# Inspect numerical columns for correlation to sale price
# Remove neighborhood columns as they will make the plot harder to parse
corr_cols = [col for col in df_train.columns if 'Neighbourhood' not in col]

correlation = df_train[corr_cols].corr()['SalePrice'].sort_values(ascending=False)

plt.figure(figsize=(8,10))
sns.barplot(correlation[2:], correlation.index[2:], orient='h')
plt.show()


# #### Overall Quality
# The feature most highly correlated with sale price is the Overall Quality. Analysing this in conjunction with the Overall Condition might be instructive.

# In[ ]:


# Examine Overall Quality and Condition
fig, ax = plt.subplots(1,3,figsize=(20,8))
sns.boxplot(x='OverallQual',y='SalePrice',data=df_train,ax=ax[0])
sns.boxplot(x='OverallCond',y='SalePrice',data=df_train,ax=ax[1])
sns.scatterplot(x='OverallQual',y='OverallCond',data=df_train,ax=ax[2])

QualCon = log_train_correlation.loc[['OverallQual','OverallCond']]
display(QualCon)

Qual_var = df_train.groupby('OverallQual').agg({'SalePrice':'var'}).rename(columns={'SalePrice':'Sale Price Variance'})
display(Qual_var.T)

print('Levene test of OverallQual and Sale Price = {}'.format(levene(df_train['OverallQual'],df_train['SalePrice'])))
print('Levene test of OverallQual and Log Sale Price = {}'.format(levene(df_train['OverallQual'],df_train['Log_SalePrice'])))


# There is clearly heteroscedasticity present is the house price as the price increases. Further transformation of the feature might reduce this but at the cost of model interpretability and ease of converting the final model output back into a $ figure. I will therefore proceed with using the log of the Sale Price and accept the heteroscedasticity.
# 
# Overall Condition does not have a high correlation with Sale Price and what correlation there is in the inverse of Overall Quality so combining these will not produce a useful feature.
# 
# I will remove the outliers with a Sale Price in excess of $700,000 from the training set. In a real life situation it would be important to understand how important to the client it was to factor in very high sale prices.

# In[ ]:


# Remove outlier sale prices
df_train.drop(df_train.loc[df_train['SalePrice']>700000].index,inplace=True)
_ = df_train.hist(column='SalePrice',bins=20,color=colours[2],alpha=0.7)


# #### GrLivArea
# The feature with the highest correlation to sale price after overall quality is 'GrLivArea'. Other features which may work well with this are 'LotArea' and 'TotalBsmtSF' to give an overall size feature and 'TotRmsAbvGrd', 'FullBath' and 'HalfBath' which can be combined to give the total number of (non) basement rooms in the house.

# In[ ]:


# Inspect Living Area related features
display(log_train_correlation.loc[['GrLivArea','TotalBsmtSF','LotArea','TotRmsAbvGrd','FullBath','HalfBath']].       sort_values('Log_Price_Correlation',ascending=False))

# Box Plots of continuous variables
fig, ax = plt.subplots(3,1,figsize=(10,5))
sns.boxplot(y='GrLivArea',data=df_train,ax=ax[0],orient='h',color=current_palette[0]).set_ylabel('GrLivArea')
sns.boxplot(y='TotalBsmtSF',data=df_train,ax=ax[1],orient='h',color=current_palette[1]).set_ylabel('TotalBsmtSF')
sns.boxplot(y='LotArea',data=df_train,ax=ax[2],orient='h',color=current_palette[2]).set_ylabel('LotArea')


# On initial inspection Lot Area has a much lower correlation with the Sale Price than the other area features so I will not proceed with mixing it in for a new feature. There also look to be clear outliers on Living Area above 4000 sqft and a single large basement above 600 sqft which I will remove from the training set.

# In[ ]:


# Drop outliers from GrLivArea - this also accounts for the single large basement
df_train.drop(df_train.loc[df_train['GrLivArea']>4000].index,inplace=True)
# Drop large lot area outlier
df_train.drop(df_train.loc[df_train['LotArea']>100000].index,inplace=True)

# Box Plots of continuous variables following outlier removal
fig, ax = plt.subplots(3,1,figsize=(10,5))
_ = sns.boxplot(y='GrLivArea',data=df_train,ax=ax[0],orient='h',color=current_palette[0]).set_ylabel('GrLivArea')
_ = sns.boxplot(y='TotalBsmtSF',data=df_train,ax=ax[1],orient='h',color=current_palette[1]).set_ylabel('TotalBsmtSF')
_ = sns.boxplot(y='LotArea',data=df_train,ax=ax[2],orient='h',color=current_palette[2]).set_ylabel('LotArea')


# In[ ]:


# See how correlation with Sale Price has changed by removing outliers
old_correlation = log_train_correlation.loc[['GrLivArea','TotalBsmtSF','LotArea','TotRmsAbvGrd','FullBath','HalfBath']                                            ,'Log_Price_Correlation']
new_correlation = df_train[['GrLivArea','TotalBsmtSF','LotArea','TotRmsAbvGrd','FullBath','HalfBath','Log_SalePrice']].                                        corr()['Log_SalePrice']
new_correlation.drop('Log_SalePrice',inplace=True)

living_area_correlation = pd.concat([new_correlation,old_correlation],axis=1)
living_area_correlation.columns = ['Outliers Removed','Outliers Present']
display(living_area_correlation)


# I will now engineer the features above to create new features which will hopefully have more predictive power.

# In[ ]:


# Engineer new features from area and room count features
Area_df = df_train[['Log_SalePrice','GrLivArea','TotalBsmtSF','TotRmsAbvGrd','FullBath','HalfBath']]

Area_df['TotalArea'] = Area_df['GrLivArea'] + Area_df['TotalBsmtSF']
Area_df['TotalRooms'] = Area_df['TotRmsAbvGrd'] + Area_df['FullBath'] + Area_df['HalfBath']
Area_df['AvgRmSize'] = Area_df['GrLivArea']/Area_df['TotRmsAbvGrd']
Area_df[['Log_SalePrice','GrLivArea','TotalBsmtSF','TotalArea','TotalRooms','AvgRmSize']].corr().iloc[1:]


# All three new features have a useful correlation with the sale price but they are also highly correlated with each other. I will therefore only keep the TotalArea feature as this has the highest correlation factor.

# #### Garage Cars & Garage Area
# These two features seem likely to be highly correlated but it may be possible to combine them in such a way as to yield a more useful single feature.

# In[ ]:


garage_df = df_train[['Log_SalePrice','GarageCars','GarageArea']]
garage_df['AreaPerCar'] = garage_df['GarageArea']/garage_df['GarageCars']
garage_corr = garage_df.corr()
garage_corr.iloc[1:]


# As expected only one of these features should be included in a regression model so I will select Garage Cars.

# #### Year Built and Year Remodelled
# Both of these features have high correlations to Sale Price. I will investigate whether the last year which work was carried out is more predictive than wither individually.

# In[ ]:


work_df = df_train[['Log_SalePrice','YearBuilt','YearRemodAdd']]
work_df['LastMod'] = work_df[['YearBuilt','YearRemodAdd']].max(axis=1)
work_df.corr().iloc[1:]


# The year built looks to have the most impact so this will be preferred for modelling.

# #### Lot Frontage
# The Lot Frontage will be a function of the overall lot area but there might be some value attached to the layout of the house with respect to what percentage the Frontage makes up.

# In[ ]:


lot_df = df_train[['Log_SalePrice','LotArea']]
lot_df = pd.concat([lot_df,Area_df['TotalArea']],axis=1)
lot_df['Lot_Proportion'] = lot_df['LotArea']/lot_df['TotalArea']
lot_df.corr().iloc[1:]


# #### Kitchens Above Grade
# This shows up has having one of the stronger negative correlations so I want to investigate it further

# In[ ]:


plt.figure(figsize = (8,6))
ax1 = sns.boxplot(x = 'KitchenAbvGr', y = 'SalePrice', data = df_train)
display(df_train['KitchenAbvGr'].value_counts())

two_kit = df_train.loc[df_train['KitchenAbvGr']==2,['LotArea','YearBuilt','TotRmsAbvGrd','GarageCars']]
one_kit = df_train.loc[df_train['KitchenAbvGr']==1,['LotArea','YearBuilt','TotRmsAbvGrd','GarageCars']]
display(one_kit.describe(),two_kit.describe())


# There does seem to be a genuine correlation with having two kitchens and a lower sale price. Whether this is causal or not it may help the model so I will code a feature which is 1 is two kitchens and 0 otherwise.

# #### Porch
# There are a number of porch related columns with a degree of positive correlation. It may be possible to engineer a single feature to replace multiple.

# In[ ]:


porch_cols = [col for col in df_train.columns if 'Porch' in col]
porch_df = df_train.loc[:,porch_cols]
porch_df = pd.concat([df_train['Log_SalePrice'],porch_df],axis=1)

# Create feature for whether or not a house has a porch
porch_df['HasPorch'] = np.where(df_train[porch_cols].sum(axis=1)>0,1,0)

# Create feature which is an ordinal encoding of porch type based on univariate correlation
def porch_func(row):
    if row['EnclosedPorch']>0:
        return 0
    elif row['3SsnPorch']>0:
        return 1
    elif row['ScreenPorch']>0:
        return 2
    else:
        return 3

porch_df['PorchQual'] = porch_df.apply(porch_func,axis=1)

porch_corr = porch_df.corr()
porch_corr.iloc[1:]


# The simplification to whether or not a house has a porch has a better correlation than any individual porch feature or an ordinal feature.

# ### Numerical Feature Processing
# Now I need to create my engineered numerical features in all three dataframes

# In[ ]:


def create_num_features(df):
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['TwoKitchens'] = np.where(df['KitchenAbvGr']==2,0,1).astype(np.int64)
    porch_cols = [col for col in df.columns if 'Porch' in col]
    df['HasPorch'] = np.where(df[porch_cols].sum(axis=1)>0,1,0)
    
    return df

df_train = create_num_features(df_train)
df_train['Log_SalePrice'] = np.log(df_train['SalePrice'])

df_valid = create_num_features(df_valid)
df_valid['Log_SalePrice'] = np.log(df_valid['SalePrice'])

house_test = create_num_features(house_test)


# In[ ]:


# Look at heatmap or the correlation between non-neighbourhood columns

corr_cols = [col for col in df_train.columns if 'Neighbourhood' not in col]
corr = df_train[corr_cols].corr()
corr = corr.reindex(index =(['SalePrice','Log_SalePrice'] + list([col for col in corr.columns if 'SalePrice' not in col]))                ,columns=(['SalePrice','Log_SalePrice'] + list([col for col in corr.columns if 'SalePrice' not in col])))

fig = plt.figure(figsize=(16,15))
ax = fig.add_subplot(111)
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, 
           xticklabels=corr.columns.values,
           yticklabels=corr.index.values,
           cmap = cmap)
ax.xaxis.tick_top()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()


# As expecting from the aboce analysis TotalArea is highly correlated with a number of other features in the original data set as well as those which I engineered. Based on univariate correlation and colinearity I will use an initial subset of features:
# 
# OverallQual, HasFireplace, MasVrn, OverallQual, TotalArea, GarageCars, YearBuilt, TwoKitchens, HasPorch and the binary encoded Neighbourhood columns
# 
# #### Create selected feature subset

# In[ ]:


predictors = ['ExterQual','HasFireplace','MasVnr','OverallQual','TotalArea','GarageCars','YearBuilt','TwoKitchens'                ,'HasPorch','GarageFinish_Ordinal'] + [col for col in df_train.columns if 'Neighbourhood' in col]

target_col = ['Log_SalePrice']


# #### Standardise Features
# The selected features have differing scales and given that I will be trialing linear regression algorithms I will need to transform the features with larger scales.
# 
# First inspect the data again to identify the features which need scaling.

# In[ ]:


df_train[['ExterQual','HasFireplace','MasVnr','OverallQual','TotalArea','GarageCars','YearBuilt','TwoKitchens'                ,'HasPorch','GarageFinish_Ordinal']].describe()


# Looking at the data OverallQual, GarageCars and GarageFinish_Ordinal can all be sensible scaled to be between 0 and 1 while TotalArea and YearBuilt are probably better standardized.

# In[ ]:


scaler = StandardScaler()
scaler.fit(df_train[['TotalArea','YearBuilt']])

def scale_data(df):
    df['OverallQual'] = df['OverallQual']/10
    df['GarageCars'] = df['GarageCars']/4
    df['GarageFinish_Ordinal'] = df['GarageFinish_Ordinal']/3
    df[['TotalArea','YearBuilt']] = scaler.transform(df[['TotalArea','YearBuilt']])
    
    return df
    


# In[ ]:


df_train = scale_data(df_train)
df_valid = scale_data(df_valid)
house_test = scale_data(house_test)


# In[ ]:


df_train[['ExterQual','HasFireplace','MasVnr','OverallQual','TotalArea','GarageCars','YearBuilt','TwoKitchens'                ,'HasPorch','GarageFinish_Ordinal']].describe()


# #### Look At Best Features
# I have already selected a subset of features based on my EDA but I will now use more analytical methods to determine the most useful features in case I later want to restrict the model further.

# In[ ]:


# SelectKBest
selector = SelectKBest(f_regression, k=10)
selector.fit(df_train[predictors], df_train[target_col])

df_new = selector.transform(df_train[predictors])
print(df_new.shape)

# Look at selected columns
df_train[predictors].columns[selector.get_support(indices=True)].tolist()


# #### Test Initial Models
# A number of regression model are available. I will test the base version of these on the training data set to see which performs the best

# In[ ]:


models = [('sk_LinearRegression',sk_lr()),('Ridge',Ridge()),('Lasso',Lasso()),          ('DecisionTree',DecisionTreeRegressor()),('RandomForest',RandomForestRegressor())          ,('GradientBoost',GradientBoostingRegressor())]

model_comp = pd.DataFrame(columns = ['Model','CV_mean','CV_std'])

for ind, (name, model) in enumerate(models):
    cv_results = cross_val_score(model, df_train[predictors],df_train[target_col],cv=5,scoring='neg_root_mean_squared_error',                                verbose = 1)
    model_comp.loc[ind,'Model'] = name
    model_comp.loc[ind,'CV_mean'] = cv_results.mean()
    model_comp.loc[ind,'CV_std'] = cv_results.std()

model_comp.sort_values(['CV_mean'],ascending=False)


# The top two performing models in base configuration are GradientBoosting and Ridge Regression. As these take two fundementally different approaches I will optimise both on the validation set and choose the best performing as my final model.
# 
# #### Model Tuning

# In[ ]:


# # Ridge Regression
# ridge = Ridge()
# params = {'alpha':[0.05,0.1,0.2,0.4,0.7,1]}
# gs_ridge = GridSearchCV(ridge,param_grid=params,cv=5,scoring='neg_mean_squared_error',verbose=1)
# gs_ridge.fit(df_valid[predictors],df_valid[target_col])
# print(gs_ridge.score(df_valid[predictors],df_valid[target_col]))
# print(gs_ridge.best_params_)


# In[ ]:


# # GradientBoosting
# gbr = GradientBoostingRegressor()
# params = {'learning_rate':[0.01,0.05,0.1,0.2],'n_estimators':[50,75,100,125,150],'subsample':[0.5,0.8,1],\
#          'max_depth':[1,3,5]}
# gs_gbr = GridSearchCV(gbr,param_grid = params,cv=5,scoring='neg_mean_squared_error',verbose=1)
# gs_gbr.fit(df_valid[predictors],df_valid[target_col])
# print(gs_gbr.score(df_valid[predictors],df_valid[target_col]))
# print(gs_gbr.best_params_)


# GradientBoosting is marginally better but I will produce predictions for the test set using the best parameters found for both models and the combined training and validation sets for training. The final predictions will need converting back into actual dollars instead of log(dollars)

# In[ ]:


# Join training and validation data sets together for maximum training data
training = pd.concat([df_train[predictors],df_valid[predictors]])
target = pd.concat([df_train[target_col],df_valid[target_col]])


# In[ ]:


# # Ridge Regression
# tuned_ridge = Ridge(**gs_ridge.best_params_)
# tuned_ridge.fit(training, target)
# ridge_output = tuned_ridge.predict(house_test[predictors])

# # Convert back to real $
# dollar_ridge_output = np.exp(ridge_output)

# # Flatten array to become df
# dollar_ridge_output = np.ndarray.flatten(dollar_ridge_output)

# ridge_submission = pd.DataFrame({'Id':test_id,'SalePrice':dollar_ridge_output})


# In[ ]:


# # GradientBoost Regression
# tuned_gbr = GradientBoostingRegressor(**gs_gbr.best_params_)
# tuned_gbr.fit(training, target)
# gbr_output = tuned_gbr.predict(house_test[predictors])

# # Convert back to real $
# dollar_gbr_output = np.exp(gbr_output)

# # Flatten array to become df
# dollar_gbr_output = np.ndarray.flatten(dollar_gbr_output)

# gbr_submission = pd.DataFrame({'Id':test_id,'SalePrice':dollar_gbr_output})


# In[ ]:


# # Produce submission files
# ridge_submission.to_csv('ridge_submission.csv',index=False)
# gbr_submission.to_csv('gbr_submission.csv',index=False)


# In[ ]:


# XGBoost
xgbr = XGBRegressor()
xgbr_param_grid = {'objective':['reg:squarederror'],
                   'learning_rate': [0.1],
                    'n_estimators': [200],
                    'subsample': [ 0.7, 0.75],
                   'max_depth': [3,4],
                  'gamma': [0, 0.01],
                  'lambda':[0.15, 0.2, 0.25]}
# xgbr_param_grid = {'objective':['reg:squarederror'],
#                     'learning_rate': [0.1,0.5],
#                     'n_estimators': [100],}

tuned_xgbr = GridSearchCV(estimator = xgbr, param_grid = xgbr_param_grid, cv=3, verbose=1, scoring='neg_mean_squared_error')
tuned_xgbr.fit(training,target)
print('Best Score = {:.3f}'.format(tuned_xgbr.best_score_))
print('Best Parameters: ',tuned_xgbr.best_params_)


# In[ ]:


xgbr_final = XGBRegressor(**tuned_xgbr.best_params_)
xgbr_final.fit(training,target)
xgbr_output = xgbr_final.predict(house_test[predictors])

# Convert back to real $
dollar_xgbr_output = np.exp(xgbr_output)

# Flatten array to become df
dollar_xgbr_output = np.ndarray.flatten(dollar_xgbr_output)

xgbr_submission = pd.DataFrame({'Id':test_id,'SalePrice':dollar_xgbr_output})


# In[ ]:


xgbr_submission.to_csv('xgbr_submission.csv',index=False)

