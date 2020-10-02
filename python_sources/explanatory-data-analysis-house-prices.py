#!/usr/bin/env python
# coding: utf-8

# # *Explanatory Data Analysis | House Prices*  
# **David Rivas, Ph.D.**  

# # Abstract  
# We analyse 'SalePrice' by itself and with the most correlated variables.  
# We deal with missing data and outliers.  
# We test some of the fundamental statistical assumptions.   
# Finally, we transform categorial variables into dummy variables.  
# 
# This work is based on the references and will be further extended in the future. In particular, more exploratory data analysis and feature engineering will be performed.  Also, various regression models will be applied to this data for evaluation purposes. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df_train.columns


# In[ ]:


df_train.head(10)


# # Dependent variable analysis 

# In[ ]:


df_train['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(df_train['SalePrice']);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# # Selection of significant variables based on qualitative/subjective analysis  
# The **bold face variables below** were selected based on qualitative thinking and visual inspections of SalePrice plots versus the variables (ironically, Neighborhood (location) was not found to be significant; but this is perhaps due to using scatter plots instead of boxplots (which are more suitable for categorical variable visualization) in the visual inspections). This selection will be confirmed numerically through a correlation analysis in the next section.   
#   
# **SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.**  
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling  
# **OverallQual: Overall material and finish quality**  
# * OverallCond: Overall condition rating  
# **YearBuilt: Original construction date**  
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area  
# **TotalBsmtSF: Total square feet of basement area**  
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)  
# **GrLivArea: Above grade (ground) living area square feet**  
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# ###  Relationship with numerical variables

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# The above graph shows a linear relationship.

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# The above graph shows that for houses with a basement there is a linear(almost exponential) relationship, and also another branch for the no basement case.

# ###  Relationship with categorical features

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# The above graph shows a rough linear (somewhat exponential) relationship.

# In[ ]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# The above graph shows that buyers are more prone to spend more money in new houses than in old relics (however, this is not a strong trend).

# #  Selection of significant variables based on an numerical/correlations/visual analysis

# ### Correlation matrix (heatmap style)

# In[ ]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# The above correlations matrix shows:  
# * Very Strong correlations (yellow colored squares) between  'TotalBsmtSF' and '1stFlrSF' variables; and the'GarageX' variables (in each case variables give almost the same information implying multicollinearity).  
# * Notable 'SalePrice' correlations with 'GrLivArea', 'TotalBsmtSF', 'OverallQual' (confirming our previous subjective analysis above) and others that we examine below.  

# ### 'SalePrice' correlation matrix (zoomed heatmap style)

# Based on the correlation matrix above, we select the variables that are the most correlated with SalePrice to examine them further:

# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# The above matrix of the 10 (=k) variables most correlated with 'SalePrice'shows that:  
# 
# * 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.   
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, since the number of cars that fit into the garage is limited by the garage area, 'GarageCars' is highly correlated with 'GarageArea' and thus only one of these variables needs to remain in the analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).  
# * 'TotalBsmtSF' and '1stFloor' are also correlated with 'SalePrice', and with each other for obvious reasons. We can keep 'TotalBsmtSF'.
# * 'FullBath'is correlated with 'SalePrice'.  
# * 'TotRmsAbvGrd' is highly correlated with 'GrLivArea'.  
# * 'YearBuilt' is slightly correlated with 'SalePrice'.  

# ### Scatter plots of the most correlated variables with'SalePrice'

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# The above plot has a lot of information about the relationships between variables. For instance, notice:  
# 
# 'TotalBsmtSF' and 'GrLiveArea' - basement areas are usually less than the above ground living area  
# 
# 'SalePrice' and 'YearBuilt' - the 'dots cloud' appears to be have a somewhat exponential function, prices have been increasing faster more recently

# # Missing Data

# Important questions about missing data:  
# 
# How prevalent is the missing data? Missing data can imply a reduction of the sample size. This can prevent us from proceeding with the analysis.  
# Is missing data random or does it have a pattern? We need to ensure that the missing data process is not biased and hidding an inconvenient truth.   
# 

# In[ ]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# (above we're practically dividing the total number of nulls in a column per the total number of rows in that column)

# For this data,we'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.), specially since none of these variables seem to be "statistically" significant (according to the analysis in the previous section), and are strong candidates for outliers.  
# 
# 'GarageX' variables have the same number of missing data, and thus probably refer to the same set of observations. Since the most important information regarding garages is already expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, we'll delete the mentioned 'GarageX' variables. The same logic applies to 'BsmtX' variables.  
# 
# Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, we will not lose information if we delete 'MasVnrArea' and 'MasVnrType'.  
# 
# Finally, we have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.  
# 
# In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.  

# In[ ]:


#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...


# # Outliers

# Outliers is a complex subjet because it may or may not contain important information.  
# ### Univariate analysis  
# The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data, that is, we'll convert data values to have a mean of 0 and a standard deviation of 1.

# In[ ]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# According to the above standarization, we have a long tail only on the positive axis, thus with possibly 2 outliers near 7 (which for now we won't consider them to be outliers).  
# ### Bivariate analysis  
# 

# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In the above plot (which was already previously plotted):
# The two far RHS observations are likely to be outliers, we'll thus delete them.  
# The two highest observations seem to be aligned with the trend (and in fact, correspond to the two outliers near 7 discused in the previous paragraph) so we'll keep them.

# In[ ]:


#identifying the indices of the outliers
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]


# In[ ]:


#deleting outliers
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# We can feel tempted to eliminate some observations (e.g. TotalBsmtSF > 3000) but for now let us just not delete any of them.

# # Variables tested against the multivariate analysis assumptions  
# The four statistical assumptions that must be tested in order for a multivariate analysis to be valid are [Multivariate Data Analysis](https://www.amazon.com/Multivariate-Data-Analysis-Joseph-Hair/dp/9332536503/ref=as_sl_pc_tf_til?tag=pmarcelino-20&linkCode=w00&linkId=5e9109fa2213fef911dae80731a07a17&creativeASIN=9332536503):  
# 
# **Normality** - The data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that with big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedasticity - opposite to homoscedasticity) which is the main reason why we are doing this analysis.  
# 
# **Homoscedasticity** - It refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.  
# 
# **Linearity**- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.  
# 
# **Absence of correlated errors** - Correlated errors occur when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.  

# ### In the search for normality
# The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:
# 
# **Histogram** - Kurtosis and skewness.  
# **Normal Probability Plot** - Data distribution should closely follow the diagonal line that represents the normal distribution.

# ### SalePrice original plots

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# 'SalePrice' is not normal. It shows positive kurtosis (i.e. peakedness), positive skewness and does not follow the diagonal line.  
# 
# ### SalePrice plots after a transformation  
# Transformation to get a normal distribution:  in case of positive skewness, log transformations usually works well.  

# In[ ]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# ### GrLivArea original plots

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# Skewness is seen so the same type of transformation as above should work.  
# ### GrLivArea plots after a transformation  
# 

# In[ ]:


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# As shown above, the transformation results in normalization.  
# ### TotalBsmtSF original plots

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In the above histogram we have skewness but in the probability plot we have a significant number of observations with zero value (houses without basement), which do not allow us to do log transformations.  
# ### TotalBsmtSF plots after a transformation  
# In order to do a log transformation here, we'll create a variable that can give the effect of having or not having a basement (binary variable). Then, we'll do a log transformation of all the non-zero observations, thus ignoring those with zero values. This should enable us to transform the data, without losing the effect of having or not having a basement (there is a chance that this is not the right treatment but as shown in the plots below the results look good). 

# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# ### Homoscedasticity Test  
# The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).  

# In[ ]:


#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# The above post log transformation plot exhibits homoscedasticity. By contrast, prior to the log transformation, in the bivariate scatter plots in the previous sections above these variables had a conic shape.  
# **Notice the power of normality! By just ensuring normality in some variables, we solved the homoscedasticity problem.**

# In[ ]:


#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# Similarly, post transformation, in the above plot, 'SalePrice' exhibit equal levels of variance across the range of 'TotalBsmtSF', thus showing homoscedasticity.

# # Converting categorical variables into dummy variables

# In[ ]:


#converting categorical variables into dummy variables
df_train = pd.get_dummies(df_train)


# # References used  
# 
# [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) by Pedro Marcelino  
# 
# [Data analysis and feature extraction with Python](https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python) by Pedro Marcelino 
