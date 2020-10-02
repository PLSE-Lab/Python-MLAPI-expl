#!/usr/bin/env python
# coding: utf-8

# <h1>House Prices Project - Part 1: Feature Engineering and Data Transformation</h1>
# 
# In this notebook, I present my data preparation and my analysis of the House Prices Project dataset.
# 
# Here you'll find feature engineering techniques and some visualizations that help us have a good idea of how this dataset is structured.
# 
# I create additional variables keeping in mind that I don't have a pre-selected regression model that I intend to use. So some variables may be more or less useful depending on the regression model adopted in the future. The model and variable selection techniques I'll present in my next notebook on the House Prices Project. 

# <h2>Loading libraries and datasets</h2>

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from scipy.stats import pearsonr
from scipy.stats import mode

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
plt.rcParams["figure.figsize"] = (15,7) # plot size


# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# <h2>Checking the datasets</h2>

# First things first. Let's see the datasets dimensions.

# In[ ]:


print('Train shape: ' + str(train.shape) + '.')
print('Test shape: ' + str(test.shape) + '.')


# Since we already have more than 80 columns in the training set, I'll adjust the display to show up to 100 rows and columns. This can help us better visualize some data that will be generated.

# In[ ]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# Let's check for missing values.

# In[ ]:


print('Missing values in the train set: ' + str(train.isnull().sum().sum()) + '.')
print('Missing values in the test set: ' + str(test.isnull().sum().sum()) + '.')


# It will be necessary to work on this missing values. Since there are several missing values in the train and test sets, is more efficient to join both datasets and work on the missing values than to do it separately.

# In[ ]:


train['dataset'] = 'train'   # identify this as the train dataset
test['dataset'] = 'test'     # identify this as the train dataset
dataset = train.append(test, sort = False, ignore_index = True) # merge both datasets
del train, test              # free some memory.


# In[ ]:


dataset.shape


# In[ ]:


dataset.dataset.value_counts()


# Checking all the columns in the dataset and getting some statistics.

# In[ ]:


dataset.columns


# In[ ]:


stats = dataset.describe().T
for i in range(len(dataset.columns)):
    stats.loc[dataset.columns[i], 'mode'], stats.loc[dataset.columns[i], 'mode_count'] = mode(dataset[dataset.columns[i]])
    stats.loc[dataset.columns[i], 'unique_values'] = dataset[dataset.columns[i]].value_counts().size
    stats.loc[dataset.columns[i], 'NaN'] = dataset[dataset.columns[i]].isnull().sum()
    if np.isnan(stats.loc[dataset.columns[i], 'count']): 
        stats.loc[dataset.columns[i], 'count'] = dataset.shape[0] - stats.loc[dataset.columns[i], 'NaN']
stats = stats[['count', 'NaN', 'unique_values', 'mode', 'mode_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats.index.name = 'variable'
stats.reset_index(inplace = True)
stats


# <h2>Feature Engineering</h2>

# <h3>Dealing with NaN Values</h3>
# 
# Let's treat all these missing data. First of all, lets check how many observations in each variable are missing values.

# In[ ]:


variables = list(stats[stats['NaN'] > 0].sort_values(by = ['NaN'], ascending = False).variable)
sns.barplot(x = 'variable', y='NaN', data = stats[stats['NaN'] > 0], order = variables)
plt.xticks(rotation=45)
stats[stats['NaN'] > 0].sort_values(by = ['NaN'], ascending = False)[['variable', 'NaN']]


# Having detailed information about which variables have missing values and how many they are, we can treat these cases and replace the *NaN* values for other values that may be more adequate. Some things that I would like to highlight:
# 
# - One thing to notice is that the SalePrice variable has 1459 *NaN* values (the same number of rows in the test set). This is so because these are the values that we have to predict in this competition, so we are not dealing with those missing values now, they are our final goal;
# - Checking the data_description.txt file we can see that most of these missing values actually indicates that the house doesn't have that feature. i.e. Missing values in the variable FireplaceQu indicates that the house doesn't have a fireplace. With this in mind I'll replace the missing values with a *NA* when in case of a categorical variable or I'll replace it with a *0* otherwise.

# <h3>Direct transformation of NaN values into NA or into 0</h3>
# 
# For this reason, the following variables had their *NaN* values transformed:
# 
# - Alley, 
# - BsmtCond, 
# - BsmtExposure, 
# - BsmtFinSF1, 
# - BsmtFinSF2,
# - BsmtFinType1, 
# - BsmtFinType2, 
# - BsmtFullBath,
# - BsmtHalfBath,   
# - BsmtQual, 
# - BsmtUnfSF 
# - Fence, 
# - FireplaceQu, 
# - GarageCond, 
# - GarageFinish, 
# - GarageQual, 
# - GarageType,
# - MiscFeature, 
# - TotalBsmtSF.

# In[ ]:


dataset['MiscFeature'].fillna('NA', inplace = True)
dataset['Alley'].fillna('NA', inplace = True)
dataset['Fence'].fillna('NA', inplace = True)
dataset['FireplaceQu'].fillna('NA', inplace = True)
dataset['GarageFinish'].fillna('NA', inplace = True)
dataset['GarageQual'].fillna('NA', inplace = True)
dataset['GarageCond'].fillna('NA', inplace = True)
dataset['GarageType'].fillna('NA', inplace = True)
dataset['BsmtExposure'].fillna('NA', inplace = True)
dataset['BsmtCond'].fillna('NA', inplace = True)
dataset['BsmtQual'].fillna('NA', inplace = True)
dataset['BsmtFinType1'].fillna('NA', inplace = True)
dataset['BsmtFinType2'].fillna('NA', inplace = True)
dataset['BsmtFullBath'].fillna(0.0, inplace = True)
dataset['BsmtHalfBath'].fillna(0.0, inplace = True)
dataset['BsmtFinSF1'].fillna(0.0, inplace = True)
dataset['BsmtFinSF2'].fillna(0.0, inplace = True)
dataset['BsmtUnfSF'].fillna(0.0, inplace = True)
dataset['TotalBsmtSF'].fillna(0.0, inplace = True)


# <h4>The following variables required some kind of additional evaluation before I could transform the missing values.</h4>
# 
# <h3>PoolQC</h3>
# 
# We can see in the stats dataset that PoolQC has 2909 missing values, but PoolArea has only 2906 zero values. So the 3 observations mismatched are real missing values. I'll check how is the crosstabulation between these two variables before decide what to do. 

# In[ ]:


dataset.PoolQC.value_counts()


# In[ ]:


pd.crosstab(dataset.PoolArea, dataset.PoolQC)


# In[ ]:


dataset[(pd.isna(dataset['PoolQC'])) & (dataset['PoolArea'] > 0)].PoolArea.value_counts()


# Checking the variables we can see that the range between each classification in PoolQC doesn't quite match the range of these missing values. Checking the description file we see that there is another category that is not present in this classification: 'TA' meaning 'Average/Typical'. We have no rule of thumb here. It seems reasonable to me to assume that the missing labes are 'TA' and for this reason I'm coding these three values as 'TA', but another acceptable approach would be to take the median of the PoolArea variable of each category in PoolQC and assign the missing observations to the category in PoolQC that is closer to its median value in PoolArea. In the end, the most important thing here is to don't mislabel these three cases as *NA*.

# In[ ]:


indexes = dataset[(pd.isna(dataset['PoolQC'])) & (dataset['PoolArea'] > 0)].index
dataset.loc[indexes, 'PoolQC'] = 'TA'
dataset['PoolQC'].fillna('NA', inplace = True)


# <h3>LotFrontage</h3>
# 
# LotFrontage is going to need some manipulation since it is a numerical variable with several *NaN* values. Luckily it is related to other variables with characteristics of the lot. Let's check:
# - LotArea;
# - LotShape;
# - LotConfig.

# In[ ]:


#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1 = plt.subplot(212)
ax2 = plt.subplot(221)
ax3 = plt.subplot(222)
#plt.subplots_adjust(hspace = 0.5)

sns.scatterplot(y = 'LotFrontage', x = 'LotArea', data = dataset, ax = ax1, palette = 'rainbow')
sns.boxplot(y = 'LotFrontage', x = 'LotShape', data = dataset, ax = ax2, palette = 'rainbow')
sns.boxplot(y = 'LotFrontage', x = 'LotConfig', data = dataset, ax = ax3, palette = 'rainbow')


# Looking at the variables we see that LotArea seems to be closer related to LotFrontage than the other variables. Yet, this relation doens't seem to be linear. I'll check it's correlation with this variable as it is, with its square root and with its fourth roth to see which of these transformations are more related to LotFrontage.

# In[ ]:


pearsonr(dataset.LotFrontage.dropna(), dataset[pd.notna(dataset['LotFrontage'])].LotArea)


# In[ ]:


pearsonr(dataset.LotFrontage.dropna(), np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/2))


# In[ ]:


pearsonr(dataset.LotFrontage.dropna(), np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))


# The fourth root of Lot Area is closer related to LotFrontage and for this reason I'll use it to fill in the missing values in LotFrontage.
# 
# Below I present the distribution of the fourth root of LotArea:

# In[ ]:


ax = sns.distplot(np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))
ax.set(xlabel = 'Fourth root of LotArea')


# The missing values will be fit on the regression line presented next in the scatterplot:

# In[ ]:


ax = sns.regplot(y=dataset.LotFrontage.dropna(), x=np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))
ax.set(xlabel = 'Fourth root of LotArea')


# Ok. So I'll use a robust regression model to predict the values in the missing observations.

# In[ ]:


X = np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4)
X = sm.add_constant(X)
model = sm.RLM(dataset.LotFrontage.dropna(), X)
results = model.fit()


# In[ ]:


index = dataset[pd.isna(dataset['LotFrontage'])].index
X_test = np.power(dataset.loc[index, 'LotArea'], 1/4)
X_test = sm.add_constant(X_test)
dataset.loc[index, 'LotFrontage'] = results.predict(X_test)


# In[ ]:


ax = sns.scatterplot(y=dataset.LotFrontage, x=np.power(dataset.LotArea, 1/4))
ax.set(xlabel = 'Fourth root of LotArea')


# That's it.

# <h3>GarageYrBlt</h3>
# 
# Since this is a numeric variable, if I just fill in its *NaN* values with a *zero* I can end up inserting a serious bias in the variable. It seems more reasonable to find another variable that is correlated with GarageYrBlt and see how I can manipulate both so I can fill in these gaps without harming my future models. For this reason, I'm checking its correlation with the YearBuilt variable.

# In[ ]:


pearsonr(dataset.GarageYrBlt.dropna(), dataset[pd.notna(dataset['GarageYrBlt'])].YearBuilt)


# Since these variables have a strong correlation, lets plot them together:

# In[ ]:


sns.regplot(y = dataset.GarageYrBlt.dropna(), x = dataset[pd.notna(dataset['GarageYrBlt'])].YearBuilt)


# Great! We can visualize the strong relation in the data and yet we see that there is a mislabelled observation in GarageYrBlt (the one with GarageYrBlt > 2200). To avoid that the mislabelled observation in GarageYrBlt insert a bias in the model, I'm going to replace it with a *NaN* value and then I'm going to create a linear model to predict the values in all the *NaN* observations in the GarageYrBlt variable.

# In[ ]:


index = dataset[dataset['GarageYrBlt'] > 2200].index
dataset.loc[index, 'GarageYrBlt'] = np.nan


# In[ ]:


# Fits the Regression Model.
X = dataset[pd.notna(dataset['GarageYrBlt'])]['YearBuilt']
X = sm.add_constant(X)
model = sm.OLS(dataset.GarageYrBlt.dropna(), X)
results = model.fit()


# In[ ]:


# Fill in the NaN values.
index = dataset[pd.isna(dataset['GarageYrBlt'])].index
X_test = dataset.loc[index, 'YearBuilt']
X_test = sm.add_constant(X_test)
X_test
dataset.loc[index, 'GarageYrBlt'] = round(results.predict(X_test),0).astype(int)


# The regression line in the previous plot suggests that in the more recent years GarageYrBlt might have a smaller value than YearBuilt. I'll check it:

# In[ ]:


dataset[(dataset['GarageYrBlt'] < dataset['YearBuilt'])][['GarageYrBlt', 'YearBuilt']]


# Ok. Is easy to see when the model filled the missing values. These observations, in recent years, are the ones where GarageYrBlt is equal to YearBuilt minus 4. In these cases, I'll make GarageYrBlt equal to YearBuilt. I'm calling 'recent years' anything that came after 2000 (counting the year 2000).   

# In[ ]:


dataset['GarageYrBlt'] = np.where((dataset['GarageYrBlt'] >= 2000) & (dataset['GarageYrBlt'] == dataset['YearBuilt'] - 4), dataset['YearBuilt'], dataset['GarageYrBlt'])


# <h3>MasVnrType and MasVnrArea</h3>
# 
# There is one more observation in the MasVnrType variable counting as *NaN* then there is in the MasVnrArea variable. So that observation is very likely to be mislabelled. To fix it, I'll check what are the means of the MasVnrArea variable when grouped by the categories in MasVnrType and I'll choose the category with the median in MasVnrArea closest to the value in the observation with mislabelled data.

# In[ ]:


dataset[(pd.notna(dataset['MasVnrArea'])) & (pd.isna(dataset['MasVnrType']))][['MasVnrArea', 'MasVnrType']]


# In[ ]:


dataset.groupby('MasVnrType', as_index = False)['MasVnrArea'].median()


# In[ ]:


index = dataset[(pd.notna(dataset['MasVnrArea'])) & (pd.isna(dataset['MasVnrType']))].index
dataset.loc[index, 'MasVnrType'] = 'Stone'


# Now that we have the same number of *NaN* in both variables, we can set them equal to *NA* and zero.

# In[ ]:


dataset['MasVnrType'].fillna('NA', inplace = True)
dataset['MasVnrArea'].fillna(0, inplace = True)


# <h3>MSZoning</h3>
# 
# According to the description file, there should be no *NaN* in this variable. To fix this, I'll compare the values in this variable with the values in MSSubClass and in LotArea (since the lot area may be influenced by the zoning classification of the sale). I'll choose the MSZoning value according to the category in MSSubClass of the observations with *NaN* values in the variable MSZoning and according to the LotArea closer to the median of LotArea of the observations grouped by MSSubClass.

# In[ ]:


# LotArea and MSSubClass of the observations with NaN in the MSZoning variable.
dataset[pd.isna(dataset['MSZoning'])][['MSSubClass', 'LotArea']]


# In[ ]:


# median LotArea grouped by MSZoning and MSSubClass.
temp = dataset.groupby(['MSSubClass', 'MSZoning'], as_index=False)['LotArea'].median()
temp[temp['MSSubClass'].isin([20, 30, 70])]


# In[ ]:


# Makes the substitutions.
indexes = dataset[(pd.isna(dataset['MSZoning'])) & (dataset['MSSubClass'] == 30)].index
dataset.loc[indexes, 'MSZoning'] = 'C (all)'
indexes = dataset[pd.isna(dataset['MSZoning'])].index
dataset.loc[indexes, 'MSZoning'] = 'RL'


# In[ ]:


dataset['MSZoning'].value_counts()


# <h3>Utilities</h3>
# 
# Let's check the distribution of this variable.

# In[ ]:


dataset['Utilities'].value_counts()


# Ok. So it's no brainer in which category the missing values should be classified in.

# In[ ]:


dataset['Utilities'].fillna('AllPub', inplace = True)


# <h3>Functional</h3>
# 
# Let's check the distribution of this variable.

# In[ ]:


dataset['Functional'].value_counts()


# Ok. I guess it's reasonable to classify the missing values as 'Typ'.

# In[ ]:


dataset['Functional'].fillna('Typ', inplace = True)


# <h3>GarageArea</h3>
# 
# Let's check this variable.

# In[ ]:


dataset['GarageArea'].value_counts()


# In[ ]:


dataset[pd.isna(dataset['GarageArea'])]


# In[ ]:


dataset[dataset['GarageType'] == 'Detchd'].GarageArea.describe()


# I'll set this *NaN* observation equal to the median value of the variable GarageArea when the GarageType is equal to 'Detchd'.

# In[ ]:


dataset['GarageArea'].fillna(399, inplace = True)


# <h3>GarageCars</h3>

# In[ ]:


dataset['GarageCars'].value_counts()


# In[ ]:


dataset[pd.isna(dataset['GarageCars'])]


# In[ ]:


temp = dataset.groupby(['GarageType', 'GarageCars'], as_index=False)['GarageArea'].median()
temp[temp['GarageType'] == 'Detchd']


# It seems reasonable to assume that the GarageArea is equal to 1 or 2. I'll be pragmatic here and choose the one with the median Area closer to 399.

# In[ ]:


dataset['GarageCars'].fillna(1, inplace = True)


# <h3>Exterior1st and Exterior2nd</h3>

# In[ ]:


dataset[pd.isna(dataset['Exterior2nd'])]


# Both missing values of both variables are in the same line. I'll check some crosstabulations:

# In[ ]:


pd.crosstab(dataset['Exterior1st'], dataset['ExterCond'])


# In[ ]:


pd.crosstab(dataset['Exterior2nd'], dataset['ExterCond'])


# The numbers don't change very much from one table to the other. This suggests that there must be many cases in which both variables have the same values. Let's see if this is true: 

# In[ ]:


len(dataset[dataset['Exterior1st'] == dataset['Exterior2nd']])


# Indeed, in most of the cases both variables have the same value. Since 'VinylSd' is the the most common value for both variables, I'm setting the missing value in both variables equal to 'VinylSd'.

# In[ ]:


dataset['Exterior1st'].fillna('VinylSd', inplace = True)
dataset['Exterior2nd'].fillna('VinylSd', inplace = True)


# <h3>KitchenQual</h3>

# In[ ]:


dataset[pd.isna(dataset['KitchenQual'])]


# In[ ]:


dataset[dataset['KitchenAbvGr'] ==  1].KitchenQual.value_counts()


# In[ ]:


dataset['KitchenQual'].fillna('TA', inplace = True)


# <h3>Electrical</h3>

# In[ ]:


dataset['Electrical'].value_counts()


# In[ ]:


dataset['Electrical'].fillna('SBrkr', inplace = True)


# <h3>SaleType</h3>

# In[ ]:


dataset[pd.isna(dataset['SaleType'])]


# In[ ]:


dataset[dataset['SaleCondition'] == 'Normal'].SaleType.value_counts()


# In[ ]:


dataset['SaleType'].fillna('WD', inplace = True)


# <h3>SalePrice - Variable Transformation</h3>
# 
# This variable statistics in the stats table strongly suggest that this variable is skewed to the left. Being this the case, it is recommended to log-transform SalePrice so that its distribution become more like a normal distribution, helping our dependent variable meet some assumptions made in inferential statistics.
# 
# Let's check SalePrice distribution as it is:

# In[ ]:


sns.distplot(dataset.SalePrice.dropna())


# Lets check its distribution after log transformation:

# In[ ]:


sns.distplot(np.log(dataset.SalePrice.dropna()), hist=True)


# Comparing both distributions we can see that the log transformed variable seems closer to a normal distribution than the original data and for this reason I'm going to work with the log transformed variable in my regression models.

# In[ ]:


index = dataset[pd.notna(dataset['SalePrice'])].index
dataset.loc[index, 'SalePriceLog'] = np.log(dataset.loc[index, 'SalePrice'])


# <h2>Data Transformations</h2>
# 
# The distribution of some variables suggest us that some transformations may be adequate to the regression models, depending on the model we choose to work with. Whith this in mind I'll update the stats dataset and I'll use it to help me decide which variables should be transformed, or created.

# In[ ]:


stats = dataset.describe().T
for i in range(len(dataset.columns)):
    stats.loc[dataset.columns[i], 'mode'], stats.loc[dataset.columns[i], 'mode_count'] = mode(dataset[dataset.columns[i]])
    stats.loc[dataset.columns[i], 'unique_values'] = dataset[dataset.columns[i]].value_counts().size
    stats.loc[dataset.columns[i], 'NaN'] = dataset[dataset.columns[i]].isnull().sum()
    if np.isnan(stats.loc[dataset.columns[i], 'count']): 
        stats.loc[dataset.columns[i], 'count'] = dataset.shape[0] - stats.loc[dataset.columns[i], 'NaN']
stats = stats[['count', 'NaN', 'unique_values', 'mode', 'mode_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats.index.name = 'variable'
stats.reset_index(inplace = True)
stats


# Some observations based on the table presented previously:
# - The variables *MoSold, MSSubClass, OverallCond* and *OverallQual* may work better in a regression model if coded as **categorical variables**. For this reason I'll change them to be treated as categorical;
# - Some variables with no values equal to zero could perform better in a regression model if **log transformed** since they are skewed and a transformation could help prevent problems of multicolinearity: 
#     - MSSubClass; 
#     - LotFrontage; 
#     - LotArea; 
#     - 1stFlrSF; 
#     - GrLivArea.
# - Some other variables can be used to generate new **dummy variables** indicating the presence/absence of certain features:
#     - 2ndFlrSF;
#     - 3SsnPorch;
#     - Alley;
#     - EnclosedPorch;
#     - Fence;
#     - FireplaceQu;
#     - GarageQual;
#     - LowQualFinSF;
#     - MasVnrType;
#     - MiscFeature;
#     - MiscVal;
#     - PoolQC;
#     - OpenPorchSF;
#     - ScreenPorch
#     - TotalBsmtSF;
#     - WoodDeckSF.

# First I'll convert the above mentioned variables into string types.

# In[ ]:


dataset['MoSold'] = dataset['MoSold'].astype(str)
dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)
dataset['OverallCond'] = dataset['OverallCond'].astype(str)
dataset['OverallQual'] = dataset['OverallQual'].astype(str)


# Now I make the log transformation of the following variables: *LotFrontage, LotArea, 1stFlrSF* and *GrLivArea*.

# In[ ]:


dataset['LotFrontageLog'] = np.log(dataset['LotFrontage'])
dataset['LotAreaLog'] = np.log(dataset['LotArea'])
dataset['1stFlrSFLog'] = np.log(dataset['1stFlrSF'])
dataset['GrLivAreaLog'] = np.log(dataset['GrLivArea'])


# In[ ]:


ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

sns.distplot(dataset['LotFrontageLog'], ax = ax1)
sns.distplot(dataset['LotAreaLog'], ax = ax2)
sns.distplot(dataset['1stFlrSFLog'], ax = ax3)
sns.distplot(dataset['GrLivAreaLog'], ax = ax4)


# Finally, I create dummy variables to indicate the presence/absence of some features in the houses.

# In[ ]:


dataset['2ndFlrDummy'] = np.where(dataset['2ndFlrSF'] > 0, 1, 0)
dataset['3SsnPorchDummy'] = np.where(dataset['3SsnPorch'] > 0, 1, 0)
dataset['AlleyDummy'] = np.where(dataset['Alley'] != 'NA', 1, 0)
dataset['EnclosedPorchDummy'] = np.where(dataset['EnclosedPorch'] > 0, 1, 0)
dataset['FireplaceDummy'] = np.where(dataset['FireplaceQu'] != 'NA', 1, 0)
dataset['LowQualFinDummy'] = np.where(dataset['LowQualFinSF'] > 0, 1, 0)
dataset['OpenPorchDummy'] = np.where(dataset['OpenPorchSF'] > 0, 1, 0)
dataset['PoolDummy'] = np.where(dataset['PoolQC'] != 'NA', 1, 0)
dataset['ScreenPorchDummy'] = np.where(dataset['ScreenPorch'] > 0, 1, 0)
dataset['PorchDummy'] = np.where(dataset['3SsnPorchDummy'] + dataset['EnclosedPorchDummy'] + dataset['OpenPorchDummy'] + dataset['ScreenPorchDummy'] > 0, 1, 0)
dataset['BsmtDummy'] = np.where(dataset['TotalBsmtSF'] > 0, 1, 0)
dataset['DeckDummy'] = np.where(dataset['WoodDeckSF'] > 0, 1, 0)


# <h2>Final look at the data</h2>
# 
# This is a final look at the dataset before implementing the regression models.
# 
# Here I try to have an idea of how each variable interact with the dependent variable of my future models: SalePriceLog.

# <h3>Correlation Matrix</h3>
# 
# I'll start by checking for some correlations to have an idea of which variables are more likely to contribute to a regression model and which aren't. 

# In[ ]:


sns.heatmap(dataset.corr(), cmap="Blues", linewidths = .2)


# Since there are many variables in the dataset, I think an easier way of checking for correlations with the dependent variable its to just check the column SalePriceLog in the correlation matrix.

# In[ ]:


dataset.corr()['SalePrice'].sort_values(ascending = False)


# <h3>Visualizations</h3>
# 
# Some variables didn't appear in the previous correlation analysis because they are categorical.
# To have an ideia of how they interact with the dependent variable I'll plot scatterplots of the numerical variables and scatterplots of the categorical variables. The Y axis is always SalePriceLog (the same visualizations can be generated to SalePrice by only replacing SalePriceLog by it in the code below).

# In[ ]:


variables = list(dataset.columns)[1:80] + list(dataset.columns)[83:]

while len(variables) >= 8:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    plt.subplots_adjust(hspace = 0.5)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    for i in range(9):
        if type(dataset[variables[i]][0]) in [np.int64, np.float64]:
            sns.scatterplot(y = 'SalePriceLog', x = variables[i], data = dataset, ax = ax[i])
        else:
            sns.boxplot(y = 'SalePriceLog', x = variables[i], data = dataset, palette = 'rainbow', ax = ax[i])
    variables = variables[9:]    

fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(15, 4.5))
plt.subplots_adjust(hspace = 0.5)
sns.boxplot(y = 'SalePriceLog', x = variables[0], data = dataset, ax = ax1, palette = 'rainbow')
sns.boxplot(y = 'SalePriceLog', x = variables[1], data = dataset, ax = ax2, palette = 'rainbow')
sns.boxplot(y = 'SalePriceLog', x = variables[2], data = dataset, ax = ax3, palette = 'rainbow')
sns.boxplot(y = 'SalePriceLog', x = variables[3], data = dataset, ax = ax4, palette = 'rainbow')
sns.boxplot(y = 'SalePriceLog', x = variables[4], data = dataset, ax = ax5, palette = 'rainbow')


# The previous correlations and visualizations suggests that some variables that would be interresting to have in a regression model are: 
# - 1stFlrSFLog;
# - BsmtCond;
# - BsmtDummy;
# - BsmtExposure;
# - BsmtFinSF1;
# - BsmtQual;
# - CentralAir;
# - ExterQual;
# - Fireplaces;
# - FireplaceQu;
# - FullBath;
# - GarageArea;
# - GarageCars;
# - GarageFinish;
# - GarageQual;
# - GarageYrBlt;
# - GrLivAreaLog;
# - HeatingQC;
# - KitchenQual;
# - LotAreaLog;
# - LotFrontage;
# - MasVnrArea;
# - OpenPorchDummy;
# - OverallQual;
# - TotalBsmtSF;
# - TotRmsAbvGrd;
# - YearBuilt;
# - YearRemodAdd.

# <h2>Train and Test Set</h2>
# 
# Since there is no more modifications that I would like to make to the dataset, it's time to separate it into train and test set again.

# In[ ]:


train = dataset[dataset['dataset'] == 'train'].copy()
train['dataset'] = None
test = dataset[dataset['dataset'] == 'test'].copy()
test['dataset'] = None


# In[ ]:


print('training set shape: ' + str(train.shape))
print('test set shape: ' + str(test.shape))


# In[ ]:


train.to_csv('train_mod.csv', index = False)
test.to_csv('test_mod.csv', index = False)


# This is it. The next step is the regression analysis, but since this notebook is already very long, I'll leave it to my next notebook.
# 
# Please, feel free to make comments and suggestions. I'm open to new ways of improving this notebook. :)
