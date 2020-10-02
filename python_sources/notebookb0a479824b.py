#!/usr/bin/env python
# coding: utf-8

# ##### UDACITY MACHINE LEARNING NANODEGREE - CAPSTONE PROJECT
# LEON PAUL
# TITLE : HOUSING PREICTIONS ON THE AMES HOUSING DATASET : ADVANCED REGRESSION MODELS.

# I would like to acknowldege the following kernels that were used  as a form of guidance while writing this notebook
# 1. https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 2. https://www.kaggle.com/humananalog/xgboost-lasso
# 3. https://www.kaggle.com/dfitzgerald3/randomforestregressor
# 4. https://www.kaggle.com/eliotbarr/stacking-starter
# 5. https://www.kaggle.com/yassineghouzam/eda-introduction-to-ensemble-regression
# 6. https://www.kaggle.com/dansbecker/learning-to-use-xgboost
# 7. https://www.kaggle.com/humananalog/xgboost-lasso/code
# 8. https://www.kaggle.com/zyedpls/regularized-linear-models
# 

# IMPORTING REQUIRED LIBRARIES:

# In[ ]:


import pandas as pd
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from random import seed
import datetime
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
print('Library Import done...')


# IMPORTING THE DATASETS:

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("the training dataset has " +str(df_train.shape[0])+" rows and "+str(df_train.shape[1])+" columns")
print("the testing dataset has " +str(df_test.shape[0])+" rows and "+str(df_test.shape[1])+" columns")


# EXPLORING THE DATA SETS:
# Getting a feel for the features available.

# In[ ]:


df_train.head(10)


# In[ ]:


target_variable = df_train['SalePrice']
target_variable.shape[0]


# Exploring the Target Variable: SalePrice

# In[ ]:


target_variable.describe()


# In[ ]:


sns.distplot(target_variable)
#plt.savefig('SalePrice_skewed.png', dpi=300, bbox_inches='tight')


# It is visible from the above distribution that the SalePrice is skewed to the left. It would be sensible to transform this to better fit a linear model. Let's take a look at the skewness and kurtosis of the SalePrice.

# In[ ]:


print("Skewness: %f" % target_variable.skew())
print("Kurtosis: %f" % target_variable.kurt())


# In[ ]:


sns.distplot(np.log(target_variable))
plt.savefig('SalePrice_norm.png', dpi=300, bbox_inches='tight')


# In[ ]:


print("Skewness: %f" % np.log(target_variable.skew()))
print("Kurtosis: %f" % np.log(target_variable.kurt()))


# This is much better.

# Let's take a look at the columns:

# In[ ]:


df_train.columns


# Exploring the features:
# At a first glance, I tried to use my intuition and domain knowldege to determine what features would heavily influence the SalePrice of a house. Based upon my judgement, the following features would be important:
#  1. GrLivArea - The square foot area of General Living spaces in the house.
#  2. 1stFlrSF - The square foot area of the 1st floor of the house
#  3. 2ndFlrSF - The square foot area of the 2nd floor of the house
#  4. OverallQual - The Overall quality of the house
#  5. FullBath - The number of full bathrooms in the house.
#  6. YearBuilt - The Year in which the house was built
#  7. GarageArea - The square foot area of the garage.
#  8. Neighborhood - The quality of the neighborhood.
# 
# The rest of the features, while also important, would not be as influential as the above features in deciding the SalePrice.
# 
# Let's use a couple of visualizations to explore these predictors. To start of lets combine the training and testing datasets.

# In[ ]:


df_full = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)
df_full.shape


# Extracting the numerical and categorical features into two separate dataframes

# In[ ]:


df_num = df_full.select_dtypes(include = ['int64','float64'])
df_car = df_full.select_dtypes(include = ['object'])
print(df_full.shape[1])
print(df_num.shape[1])
print(df_car.shape[1])


# ScatterPlots for the numeric features against the target Variable will provide an understanding of their distributions

# In[ ]:


df_num.columns


# In[ ]:


df_train.plot.scatter(x = 'GrLivArea', y = 'SalePrice')
plt.savefig('GrLivArea.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')
plt.savefig('TotalBsmtSF.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'GarageCars', y = 'SalePrice')
plt.savefig('GarageCars.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'OverallQual', y = 'SalePrice')
plt.savefig('OverallQual.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'YearBuilt', y = 'SalePrice')
plt.savefig('YearBuilt.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'FullBath', y = 'SalePrice')
plt.savefig('FullBath.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'PoolArea', y = 'SalePrice')
plt.savefig('PoolArea.png', dpi=300, bbox_inches='tight')


# From these plots I can deduce that the GrLivArea, TotalBsmtSF, GarageCars, OverallQual, YearBuilt and FullBath have a pretty strong linear/exponential relationship with the response variable. In addition, I'd like to explore how some of the other variables mentioned earlier relate with these main features.

# In[ ]:


df_train.plot.scatter(x = '1stFlrSF', y = 'GrLivArea')
#plt.savefig('Scatter_GarageCars.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = '2ndFlrSF', y = 'GrLivArea')
#plt.savefig('Scatter_1.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = 'GarageCars', y = 'GarageArea')
#plt.savefig('Scatter_1.png', dpi=300, bbox_inches='tight')


# From these plots, it is visible that the 1stFlrSF and 2ndFlrSF correlate quite strongly with the GrLivArea. Hence, they can probably be dropped and the GrLivArea feature should be sufficient to provide the necessary information gain. Similarly the scatter plot between the GarageArea and GarageCars features indicates that they share a strong correlation and that one of them can be dropped without reducing the information gain.
# 
# Hence, it seems that there might be multiple features with high correlations that could be dropped to reduce the dimensionality of the dataset. A correlation Heatmap and scatter plots between the highest correlated variables will help identify these features.

# In[ ]:


cor_matrix = df_train.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(cor_matrix, vmax=0.8, square=True, annot=False, linewidth = 5)
plt.savefig('Heatmap_1.png', dpi=300, bbox_inches='tight')


# From ths heatmap there are multiple features that seem to be highly correlated. This could lead to mutlicollinearity which is something I'd want to avoid. So a closer scrutiny of the heatmap will allow us to remove these redundant features.

# In[ ]:


k = 15 #Looking for the top 15 features highly correlated with SalePrice
cols = cor_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(20, 15))
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.savefig('Heatmap_2.png', dpi=300, bbox_inches='tight')
plt.show()


# Immediately we can see some features that stand out as important and some which can be dropped. For example, OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF and 1stFlrSF all have correlation scores higher than 0.6 with the response feature meriting their compulsory inclusion amongst the numerical features. At the same time the GarageCars and GarageAreas features are also highly correlated at 0.88 and so one of them can be dropped without too much information loss. Similarly, the TotRmsAbvGrd and GrLivArea are also quite highly correlated with a score of 0.83 and one of them can be dropped. Since, the GrLivArea feature is more highly correlated with the SalePrice, the TotRmsAbvGrd feature can be dropped. By examining the heatmap in this manner, more redundant numerical features can be dropped.

# Next the Categorical Features can be explored to observe their relation with the SalePrice response.

# In[ ]:


df_car.columns


# Amongst these features, some that I would consider as quite important in influencing the SalePrice are BsmtQual, CentralAir, ExterCond, GarageCond/GarageQual, KitchenQual, Neighborhood and SaleCondition. We will use box plots to check their relationship with SalePrice and observe their distributions.

# In[ ]:


var = 'BsmtQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('BsmtQual.png', dpi=300, bbox_inches='tight')


# This is as expected, the general trend of SalePrice increases with an increase in Basement Quality. There aren't any anomalies.

# In[ ]:


var = 'CentralAir'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('CentralAir.png', dpi=300, bbox_inches='tight')


# It seems like the presence or absence of Central Air doesn't affect the SalePrice by much in 80-85% of the houses. There are certain outliers which show that Central Air does affect the SalePrice of some of the houses.

# In[ ]:


var = 'ExterCond'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('ExterCond.png', dpi=300, bbox_inches='tight')


# Here, there are reasonable trends that show that the SalePrice increases with the Exterior Condition of the house. However, there some anomalies where houses with average Exterior Conditions still have quite high SalePrices.

# In[ ]:


var = 'GarageQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('GarageQual.png', dpi=300, bbox_inches='tight')


# In[ ]:


var = 'GarageCond'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('GarageCond.png', dpi=300, bbox_inches='tight')


# Based upon the above boxplots, I can deduce that while there are some houses with average Quality/Condition of the basements that have rather high SalePrices, there isn't too much of variance in the SalePrice values irrespective of the GarageQuality.

# In[ ]:


var = 'KitchenQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('KitchenQual.png', dpi=300, bbox_inches='tight')


# Once again the KitchenQual boxplots show a reasonable trend and nothing out of the ordinary.

# In[ ]:


var = 'Neighborhood'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('Neighborhood.png', dpi=300, bbox_inches='tight')


# This is a slightly more complicated feature to interpret independently. Most of the neighborhoods seem to have almost similar mean SalePrice values, with some of them showing some variance over the SalePrice. I would ratify that the neighborhood definitely does affect the SalePrice, in combination with features like OverallQual, GrLivArea, GarageArea and TotalBsmtSF, based upon my intuition and domain knowledge.

# In[ ]:


var = 'FullBath'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('FullBath.png', dpi=300, bbox_inches='tight')


# The effect of the number of FullBathrooms remains almost the same when the number of bathrooms is between 0-2 but rises quite a bit when the bathrooms increase to 3.

# These plots have given me a good amount of understanding into some of the important features and how they affect the SalePrice response variable. This will be helpful in constructing my Benchmark model and subsequent models. The next step is to explore the missing values in the dataset.

# I will write a chunk which compites the sum of missing values and the percentage of them as well. 

# In[ ]:


#Imputing missing LotFrontage values
vec_lf = df_full.loc[df_full['LotFrontage'].isnull()].index.tolist()
df_full.ix[vec_lf,'LotFrontage'] = 70
df_full['LotFrontage'].isnull().sum()


# In[ ]:


total = df_full.isnull().sum().sort_values(ascending=False)
percent = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] >= 1]


# At first glance it's easy to see that some of the features have more than 90% of their values missing. I have decided to arbitarily select 20% missing as a threshold for missing values and any feature that has more than 20% of missng features will be dropped. This still leaves quite a few features to impute but I decided on 20% since I do not want to lose too much of information. The rest of the features with fewer missing values can be accounted for by checking the importance the feature and dropping it or simply deleting the rows with missing variables. It is possible that these are not missing features but instead indicate an absence of these features for the respetive houses. In that case, since more than 90% of these hosues do not have these features, the ones that do would more likely act as outliers fot the rest of the data. Hence, dropping them would make more sense.
# 
# The PoolQC, MiscFeature, Alley, Fence and FirePlaceQU features are automatically dropped since they all have more than 20% of missing values. As far as the rest are concerned I will explore them a bit more before deciding how to deal with them.

# The Garage features involving the finish qulaity, Year built etc. don't seem to have too much of an influence on the SalePrice as has been seen earlier while exploring the categorical variables. The GarageCars feature seemed like the only feature that really mattered. So I will drop all the other Garage features.

# In[ ]:


var = 'BsmtExposure'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'BsmtFinType1'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'BsmtFinType2'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'BsmtCond'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'BsmtQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# It is clear that none of these Basement features explain a lot of variance for the SalePrice. SO I can drop them as well.

# In[ ]:


var = 'MasVnrArea'
df_train.plot.scatter(x = var, y = 'SalePrice')


# In[ ]:


var = 'MasVnrType'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(28, 15))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# From the above plots it is sufficient to say that the Mason Area feature shares a weak relationship with SalePrice while the Mason type feature again explains little or none of the SalePrice variance. Hence, they both can be dropped.

# The last missing value feature is Electrical. Since this is just one missing row, I will simply drop the row from the dataset.

# NOTE: I am removing features for the combined train and test datasets since I am working on the assumption that both test and train datasets should be drawn from the same sample. Hence, removal of features should be done on both the train and test data. Further I am not involving the target variable for the removal in any manner so as to avoid any chance of overfitting. after dropping the columns I will split the datasets into the original train and test sets again and drop rows with single missing values from the train set only.

# In[ ]:


df_full['LotFrontage'].shape


# In[ ]:


total1 = df_full.isnull().sum().sort_values(ascending=False)
percent1 = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_full['LotFrontage'].isnull().sum()


# In[ ]:


missing_data1[missing_data1['Total'] > 1]


# In[ ]:


df_full = df_full.drop((missing_data[missing_data['Total'] > 5]).index,1)


# In[ ]:


df_full['LotFrontage'].shape


# In[ ]:


df_full.isnull().sum()


# In[ ]:


total1 = df_full.isnull().sum().sort_values(ascending=False)
percent1 = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_train = df_full[:df_train.shape[0]]
df_train['SalePrice'] = target_variable
df_train.head(10)


# In[ ]:


df_test = df_full[df_train.shape[0]:]
df_test.shape
df_test.head(10)


# In[ ]:


#df_train['LotFrontage']
#df_test['LotFrontage']


# In[ ]:


total1 = df_train.isnull().sum().sort_values(ascending=False)
percent1 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


# In[ ]:


total1 = df_train.isnull().sum().sort_values(ascending=False)
percent1 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_train.shape


# In[ ]:


############################################################################################################################


# In[ ]:


df_test.shape


# In[ ]:


total2 = df_test.isnull().sum().sort_values(ascending=False)
percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# I will explore the summary statistics for these columns and impute using either median values or mean values

# In[ ]:


df_full['MSZoning'].describe()


# In[ ]:


vec = df_test.loc[df_test['MSZoning'].isnull()].index.tolist()
df_test.ix[vec,'MSZoning']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'MSZoning'] = 'RL'
df_test.ix[vec,'MSZoning']


# In[ ]:


df_full['Functional'].describe()


# In[ ]:


vec = df_test.loc[df_test['Functional'].isnull()].index.tolist()
df_test.ix[vec,'Functional']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'Functional'] = 'Typ'
df_test.ix[vec,'Functional']


# In[ ]:


df_full['Utilities'].describe()


# In[ ]:


vec = df_test.loc[df_test['Utilities'].isnull()].index.tolist()
df_test.ix[vec,'Utilities']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'Utilities'] = 'AllPub'
df_test.ix[vec,'Utilities']


# In[ ]:


df_full['BsmtHalfBath'].describe()


# In[ ]:


vec = df_test.loc[df_test['BsmtHalfBath'].isnull()].index.tolist()
df_test.ix[vec,'BsmtHalfBath']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'BsmtHalfBath'] = 0
df_test.ix[vec,'BsmtHalfBath']


# In[ ]:


df_test.shape


# In[ ]:


df_full['BsmtFullBath'].describe()


# In[ ]:


vec = df_test.loc[df_test['BsmtFullBath'].isnull()].index.tolist()
df_test.ix[vec,'BsmtFullBath']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'BsmtFullBath'] = 0
df_test.ix[vec,'BsmtFullBath']


# In[ ]:


df_test.shape


# In[ ]:


df_full['BsmtFinSF2'].describe()


# In[ ]:


vec = df_test.loc[df_test['BsmtFinSF2'].isnull()].index.tolist()
df_test.ix[vec,'BsmtFinSF2']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'BsmtFinSF2'] = 0
df_test.ix[vec,'BsmtFinSF2']


# In[ ]:


df_test.shape


# In[ ]:


total2 = df_test.isnull().sum().sort_values(ascending=False)
percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_full['SaleType'].describe()


# In[ ]:


vec = df_test.loc[df_test['SaleType'].isnull()].index.tolist()
df_test.ix[vec,'SaleType']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'SaleType'] = 'WD'
df_test.ix[vec,'SaleType']


# In[ ]:


df_test.shape


# In[ ]:


df_full['Exterior1st'].describe()


# In[ ]:


vec = df_test.loc[df_test['Exterior1st'].isnull()].index.tolist()
df_test.ix[vec,'Exterior1st']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'Exterior1st'] = 'VinylSd'
df_test.ix[vec,'Exterior1st']


# In[ ]:


df_test.shape


# In[ ]:


df_full['KitchenQual'].describe()


# In[ ]:


vec = df_test.loc[df_test['KitchenQual'].isnull()].index.tolist()
df_test.ix[vec,'KitchenQual']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'KitchenQual'] = 'TA'
df_test.ix[vec,'KitchenQual']


# In[ ]:


df_test.shape


# In[ ]:


df_full['Exterior2nd'].describe()


# In[ ]:


vec = df_test.loc[df_test['Exterior2nd'].isnull()].index.tolist()
df_test.ix[vec,'Exterior2nd']
#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]


# In[ ]:


df_test.ix[vec,'Exterior2nd'] = 'VinylSd'
df_test.ix[vec,'Exterior2nd']


# In[ ]:


total2 = df_test.isnull().sum().sort_values(ascending=False)
percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_test.shape


# In[ ]:


df_test.ix[df_test.loc[df_test['GarageArea'].isnull()].index.tolist(),'GarageArea'] = 0
df_test.ix[df_test.loc[df_test['TotalBsmtSF'].isnull()].index.tolist(),'TotalBsmtSF'] = 0
df_test.ix[df_test.loc[df_test['GarageCars'].isnull()].index.tolist(),'GarageCars'] = 0
df_test.ix[df_test.loc[df_test['BsmtFinSF1'].isnull()].index.tolist(),'BsmtFinSF1'] = 0
df_test.ix[df_test.loc[df_test['BsmtUnfSF'].isnull()].index.tolist(),'BsmtUnfSF'] = 0


# In[ ]:


df_test.shape


# In[ ]:


total2 = df_test.isnull().sum().sort_values(ascending=False)
percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data1[missing_data1['Total'] > 0]


# In[ ]:


df_test.shape


# So all of the missing values in the training and testing datasets have been handled.

# ###########################################################################################################################

# In[ ]:


#df_full['LotFrontage']


# Since all the missing values from the training set have been removed I will now move onto checking any outliers.

# In[ ]:



for colname, col in df_train.select_dtypes(include = ['int64','float64']).iteritems():
    plt.scatter(df_train[colname], df_train['SalePrice'])
    plt.ylabel('Sale Price')
    plt.xlabel(colname)
    plt.show()


# From the above data, there are only one or two outliers in the entire dataset for the numeric features. i don;t think these are worth removing. SO now we can move onto fitting the benchmark model.

# For my benchmark I will be running a simple Linear regression on a selection of features that I consider important based on domain knowledge.

# In[ ]:


df_lin_full = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)
df_lin_full.shape


# In[ ]:


imp_feats = ['Id','GrLivArea','OverallQual','GarageCars','TotalBsmtSF','1stFlrSF','FullBath']
df_lin_full = df_lin_full[imp_feats]
df_lin_full.shape


# In[ ]:


df_lin_train = df_lin_full[:df_train.shape[0]]
df_lin_train.head(5)


# In[ ]:


df_lin_test = df_lin_full[df_train.shape[0]:]
df_lin_test.head(5)
df_lin_test.isnull().sum()


# Now that the data has been preprocessed the next step will be to run a benchmark model.

# ########################################################################################################################################################################################################################################################

# I will use the following rmse_cv() function that will use K-Fold Cross Validation to score the performance of a model on the training dataset. This will give me a way to score my own models and gauge their performance since the scoring on the test data will be done by Kaggle.

# In[ ]:


#Creating a scoring function to check mdoel performance using k-Fold Sampling of the training dataset
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

print('Done')


# ############################################################################################################################

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
# Log transforming the SalePrice variable since it is left skewed
X = df_lin_train.drop('Id',1)
y = np.log(df_train['SalePrice'])

lm.fit(X, y)


# In[ ]:


benchmark_model = pd.DataFrame(zip(X.columns,lm.coef_), columns = ['features','Model Coefficients'])
benchmark_model


# In[ ]:


#sns.pairplot(X, x_vars=['GrLivArea'], y_vars=y, size=7, aspect=0.7, kind='reg')


# In[ ]:


benchmark_pred = lm.predict(df_lin_test.drop('Id',1))
len(benchmark_pred)


# We will now write this submission to a csv and submit to the Kaggle page to get our logarithmic RMSE score

# In[ ]:


benchmark_sub = pd.DataFrame(np.exp(benchmark_pred), columns = ['SalePrice'])
benchmark_sub['Id'] = df_lin_test['Id']
#One of the Id values has become null. I will ahev to fill it back in
benchmark_sub.loc[benchmark_sub['Id'].isnull()].index.tolist()
benchmark_sub.ix[1379,'Id'] = 2840
benchmark_sub.ix[1375:1380,]

benchmark_sub['Id'] = benchmark_sub['Id'].astype('int64')
benchmark_sub.head(10)


# In[ ]:


#writing to a csv file
benchmark_sub.to_csv('benchmark_submission.csv')


# This submission was uploaded to the Kaggle competition page to score it and get the benchmark RMSE. The image is posted below:

# In[ ]:


from IPython.display import Image
Image(filename='Benchmark_Sub.jpg')


# We can see that our benchmark RMSE is 0.17649. The relative position is 1800 out of 2307 teams as of 2017-11-13 18:30.

# ########################################################################################################################################################################################################################################################

# Now that the benchmark has been set, the next step is to work on creating a final optimized model. For this step I have shortlisted 3 approaches:
# 1. RandomForest with GridSearch Cross Validation as a stand alone result.
# 2. Gradient Bossted Trees with GridSearch Cross Validation as a stand alone result.
# 3. Lasso Regression to check if regularization improves the performance of the humble Linear regression.
# 4. Averaging the results from the above two models together OR using a weighted average of the results of the two models.

# ############################################################################################################################

# RandomForest Model with GridSearch Cross Validation

# In[ ]:


df_fl = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)
df_fl = pd.get_dummies(df_fl)
df_fl.head(10)


# In[ ]:


df_train = df_fl[:df_train.shape[0]]
df_train['SalePrice'] = target_variable
df_test = df_fl[df_train.shape[0]:]


# In[ ]:


#pd.get_dummies(df_train).head(10)
df_train.head(10)
#df_train.shape
#df_test['LotFrontage']


# In[ ]:


pd.get_dummies(df_test).head(10)
df_test.head(10)
#df_test.shape


# ############################################################################################################################

# The first approach will be using RandomForests (RF). RFs work by creating N number of 'fully grown' decision trees/estimators that have a Low Bias and High Variance and then averaging their results. These trees/estimators are uncorrelated so that the variance of the model is reduced i.e. it reduces error by minimizing variance. But the bias of the model will be bounded by the bias of the individual trees in the ensemble. As, a result this is a Low Variance and High Bias model. Normally the initial bias of the model is kept low by estimating with large trees that are unpruned.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from random import seed

seed(124578)
enc = OneHotEncoder()
rfg = RandomForestRegressor()
tuned_parameters = {'n_estimators': [100, 150, 200, 250, 300, 400, 500, 700, 800, 1000], 'max_depth': [1, 2, 3], 'min_samples_split': [1.0, 2, 3]}

clf = GridSearchCV(rfg, tuned_parameters, cv=10, n_jobs=-1, verbose=1)


X_train = df_train.drop('SalePrice',1)
y_train = np.log(df_train['SalePrice'])
clf.fit(X_train, y_train )


# In[ ]:


rf_best = clf.best_estimator_
#plt.hist(rf_best.feature_importances_, label = df_test.columns)
rf_best


# In[ ]:


#Writing model to pickle file
filename = 'randomforest_model.sav'
pickle.dump(rf_best, open(filename, 'wb'))


# In[ ]:


#Loading model from pickle file
rf_model_load = pickle.load(open(filename, 'rb'))
rf_model_load


# In[ ]:


rf_best.fit(X_train, y_train)
print('Model Fitting done....')


# In[ ]:


rf_rmse = rmsle_cv(rf_best)
rf_rmse.mean()


# In[ ]:


rf_opt_pred = rf_best.predict(df_test)
print('Prediction with optimium model on test set done...')


# In[ ]:


rf_sub = pd.DataFrame(np.exp(rf_opt_pred), columns = ['SalePrice'])
rf_sub['Id'] = df_test['Id']
rf_sub.head(10)


# In[ ]:


#writing this result to a csv and making another submission
rf_sub.to_csv('rf_submission.csv')


# This submission gives a score about 0.2 which indicates that the model might be overfitting due to errors in my suggested parameter grid.

# ############################################################################################################################

# I decided to use the make_scorer to determine the best n_estimators to reduce the RMSE of the model.

# In[ ]:


from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cross_validation import cross_val_score

rf_scorer = make_scorer(mean_squared_error, False)
n_est = [150,250,500,750,1000]

for i in n_est:
    rf_clf = RandomForestRegressor(n_estimators=i, n_jobs=-1)
    rf_cv_score = np.sqrt(-cross_val_score(estimator=rf_clf, X=X_train, y=y_train, cv=15, scoring = rf_scorer))

    plt.figure(figsize=(10,5))
    plt.bar(range(len(rf_cv_score)), rf_cv_score)
    plt.title('Cross Validation Score for '+ str(i) + " estimators")
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')

    plt.plot(range(len(rf_cv_score) + 1), [rf_cv_score.mean()] * (len(rf_cv_score) + 1))
    plt.tight_layout()


# Based on these plots an n_estimator of around 400 to 600 would be sufficient. The average RMSE values are approximately 0.14.

# In[ ]:


#rf_best = clf.best_estimator_
rf_best = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)


# In[ ]:


#Writing model to pickle file
filename = 'randomforest_model.sav'
pickle.dump(rf_best, open(filename, 'wb'))


# In[ ]:


#Loading model from pickle file
rf_model_load = pickle.load(open(filename, 'rb'))
rf_model_load


# In[ ]:


rmsle_cv(rf_best).mean()


# In[ ]:


rf_best.fit(X_train, y_train)
print('Model Fitting done....')


# In[ ]:


rf_opt_pred = rf_best.predict(df_test)
print('Prediction with optimium model on test set done...')


# In[ ]:


rf_sub = pd.DataFrame(np.exp(rf_opt_pred), columns = ['SalePrice'])
rf_sub['Id'] = df_test['Id']
rf_sub.head(10)


# In[ ]:


#writing this result to a csv and making another submission
rf_sub.to_csv('rf_submission.csv')


# This is submitted to Kaggle to generate the RMSE score. The score was 0.14544. This was a 21.4% improvement over the 0.17649 benchmark and raises my rank from 1800 to 1395 as of 2017-11-13 19:34. That is a jump of 408 up the ranks.

# In[ ]:


from IPython.display import Image
Image(filename='RF_N500_Sub.jpg')


# ############################################################################################################################

# The next approach will be using the XGBoost model. Unlike Random Forests, Boosting works differently by using a bag of 'Weak Learners' i.e. trees with shallow depths or perhaps even simple decision stumps that have high bias and low variance. The aim is to reduce the error by minimizing the bias of the boosted model. Additionally by aggregating the results from the weak learners, Boosting is able to reduce variance to some extent as well.

# In[ ]:


##########################################################################################################################


# ########################################################################################################################

# In[ ]:


from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold

seed(235689)
#seed(235690)
xg_model = XGBRegressor()

#GridSearchCV
n_estimators = [50, 100, 150, 200, 250, 300, 500, 750, 1000]
max_depth = [2, 3, 4, 5, 6]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate = learning_rate)

k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
grid_search = GridSearchCV(xg_model, param_grid, scoring="neg_mean_squared_log_error", n_jobs=-1, cv=20, verbose=1)
print("GridSearch done ...")


# In[ ]:


X_train = df_train.drop('SalePrice',1)
y_train = np.log(df_train['SalePrice'])
final_result = grid_search.fit(X_train, y_train)


# In[ ]:


print("Best: %f using %s" % (final_result.best_score_, final_result.best_params_))


# In[ ]:


final_result.grid_scores_


# In[ ]:


final_xgb_model = final_result.best_estimator_
final_xgb_model


# In[ ]:


print('Beginning Model Fitting....')
final_xgb_model.fit(X_train, y_train)
print('Model fitting completed...')


# In[ ]:


#Writing model to pickle file
filename = 'xgboost_model.sav'
pickle.dump(final_xgb_model, open(filename, 'wb'))


# In[ ]:


#Loading model from pickle file
xg_model_load = pickle.load(open('xgboost_model.sav', 'rb'))
xg_model_load


# In[ ]:


rmsle_cv(final_xgb_model).mean()


# In[ ]:


xgb_pred = final_xgb_model.predict(df_test)
print('Prediction with optimium model on test set done...')


# In[ ]:


xgb_sub = pd.DataFrame(np.exp(xgb_pred), columns = ['SalePrice'])
xgb_sub['Id'] = df_test['Id']
xgb_sub.head(10)


# In[ ]:


#writing this result to a csv and making another submission
xgb_sub.to_csv('xgb_submission.csv')


# This XGB model was Cross Validated using k-fold Cross Validation and the best parameters were estimated. The best model was fitted to the training data and then used to predict on the test dataset. The RMSE for this model was 0.14029, a 25.8% increase over our benchmark score of 0.17649. In term's of relative performance, over the 1800 rank achieved by the benchmark, the XGBoost model has achieved the highest rank so far of 1318 as of 2017-11-14 15:57.

# In[ ]:


from IPython.display import Image
Image(filename='XGB_Sub.jpg')


# In[ ]:


datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


# ############################################################################################################################

# I will now run a Lasso Regression on the data

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV

seed(785623)

lasso_model = Lasso()

#GridSearchCV
alpha = [0.0001, 0.0002, 0.0003, 0.0004,0.0005,0.0006,0.007, 0.0008,0.0009,0.001, 0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.2, 0.3,0.4]
param_grid_ls = param_grid = dict(alpha=alpha)

grid_search_ls = GridSearchCV(lasso_model, param_grid_ls, scoring="neg_mean_squared_error", n_jobs=-1, cv=15, verbose=1)
print("GridSearch done ...")


# In[ ]:


X_train = df_train.drop('SalePrice',1)
y_train = np.log(df_train['SalePrice'])
final_result_ls = grid_search_ls.fit(X_train, y_train)


# In[ ]:


final_lasso_model = final_result_ls.best_estimator_
final_lasso_model


# In[ ]:


final_result_ls.grid_scores_


# In[ ]:


#Writing model to pickle file
filename = 'lasso_model.sav'
pickle.dump(final_lasso_model, open(filename, 'wb'))


# In[ ]:


#Loading model from pickle file
ls_model_load = pickle.load(open('lasso_model.sav', 'rb'))
ls_model_load


# In[ ]:


rmsle_cv(final_lasso_model).mean()


# In[ ]:


print('Beginning Model Fitting....')
final_lasso_model.fit(X_train, y_train)
print('Model fitting completed...')


# In[ ]:


lasso_pred = final_lasso_model.predict(df_test)
print('Prediction with optimium model on test set done...')


# In[ ]:


lasso_sub = pd.DataFrame(np.exp(lasso_pred), columns = ['SalePrice'])
lasso_sub['Id'] = df_test['Id']
lasso_sub.head(10)


# In[ ]:


#writing this result to a csv and making another submission
lasso_sub.to_csv('lasso_submission.csv')


# This Lasso model was Cross Validated using k-fold Cross Validation and the best parameters were estimated. The best model was fitted to the training data and then used to predict on the test dataset. The RMSE for this model was 0.12671, approximately a 40% increase over our benchmark score of 0.17649. In term's of relative performance, over the 1800 rank achieved by the benchmark, the XGBoost model has achieved the highest rank so far of 902, up from XGB's 1318, as of 2017-11-14 16:45.

# In[ ]:


from IPython.display import Image
Image(filename='Lasso_Sub.jpg')


# ############################################################################################################################

# Hence, so far I have trained a series of models and tested their Logarithmic RMSE scores using the Kaggle Submission portal. In conclusion, I will use a stacked regression model by taking a weighted average of the two best models, the XGBoost and the Lasso Regression, and checking it's performance against the individual models' Logarithmic RMSE scores. The model performance summary has been shown below in the figure:

# In[ ]:


from IPython.display import Image
Image(filename='Final_Models.jpg')


# ############################################################################################################################

# Summarizing all the models and their performances. I will create two dictionaries of the models. One with their Cross Validated RMSLE and one with their Test score RMSLE values from Kaggle. A visulaiztion will help compare them, as well as trace my progress.

# In[ ]:


#Loading saved models from pickle files
rf_model_load = pickle.load(open('randomforest_model.sav', 'rb'))
xg_model_load = pickle.load(open('xgboost_model.sav', 'rb'))
ls_model_load = pickle.load(open('lasso_model.sav', 'rb'))


# In[ ]:


ls_model_load


# In[ ]:


X_train = df_train.drop('SalePrice',1)
y_train = np.log(df_train['SalePrice'])
X_train.head(10)


# In[ ]:


raw_data = {'Models': ['Benchmark Linear regression', 'Random Forest Regressor', 'XGBoost Regressor', 'Lasso Regularized Linear Regression'],
        'train_rmsle_score': [0,rmsle_cv(rf_model_load).mean(),rmsle_cv(xg_model_load).mean(),rmsle_cv(ls_model_load).mean()],
        'test_rmsle_score': [0.17649,0.14544,0.14029,0.12671]}
df = pd.DataFrame(raw_data, columns = ['Models', 'train_rmsle_score', 'test_rmsle_score'])
df.shape


# In[ ]:


1-df['train_rmsle_score']


# In[ ]:


# Setting the positions and width for the bars
pos = list(range(len(df['train_rmsle_score']))) 
width = 0.4 

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with train_rmsle_score data,
# in position pos,
plt.bar(pos, 
        #using df['pre_score'] data,
        df['train_rmsle_score'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=df['Models'][0]) 

# Create a bar with test_rmsle_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df['test_rmsle_score'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in first_name
        label=df['Models'][1]) 

# Set the y axis label
ax.set_ylabel('Score')

# Set the chart's title
ax.set_title('Model RMSLE Scores for Training and Test sets.')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['Models'], rotation=45, ha='right')

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(df['train_rmsle_score'] + df['test_rmsle_score'])])

# Adding the legend and showing the plot
plt.legend(['Training Score', 'Testing Score'], loc='upper left')
plt.grid()
plt.show()


# In[ ]:


accuracies = pd.DataFrame(df['Models'])
accuracies['Train_accuracy'] = 1 - df['train_rmsle_score']
accuracies['Test_accuracy'] = 1 - df['test_rmsle_score']
accuracies.ix[0,'Train_accuracy'] = 0
accuracies


# In[ ]:


# Setting the positions and width for the bars
pos = list(range(len(accuracies['Train_accuracy']))) 
width = 0.4 

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with train_rmsle_score data,
# in position pos,
plt.bar(pos, 
        #using accuracies['Train_accuracy'] data,
        accuracies['Train_accuracy'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=accuracies['Models'][0]) 

# Create a bar with test_rmsle_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using accuracies['Test_accuracy'] data,
        accuracies['Test_accuracy'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in first_name
        label=accuracies['Models'][1]) 

# Set the y axis label
ax.set_ylabel('Accuracy Scores')

# Set the chart's title
ax.set_title('Model Accuracy Scores for Training and Test sets.')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(accuracies['Models'], rotation=45, ha='right')

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(accuracies['Train_accuracy'] + accuracies['Test_accuracy'])])

# Adding the legend and showing the plot
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
plt.grid()
plt.show()


# I can see from these plots that the overall RMSLE scores have progressively reduced with time which is good news. On the whole, the XGBoost has performed better on the training data compared to the test set, which was what I expected. Surprsingly the Lasso regression model has performed slightly better on the Test set compared to the Training set. It seems like a weighted average of these two may or may not improve my score.

# In[ ]:


#fitting the three models to the training data
rf_model_load.fit(X_train, y_train)
print('Random Forest model fitted')
xg_model_load.fit(X_train, y_train)
print('XGBoost model fitted')
ls_model_load.fit(X_train, y_train)
print('Lasso model fitted')


# In[ ]:


#Generating predictions on the test data for the three models loaded from pickle data files.
rf_pred_wa = rf_model_load.predict(df_test)
xg_pred_wa = xg_model_load.predict(df_test)
ls_pred_wa = ls_model_load.predict(df_test)
rf_pred_wa


# In[ ]:


#Generating predictions on the test data for the three models loaded from pickle data files.
#rf_pred_wat = np.exp(rf_model_load.predict(X_train))
#xg_pred_wat = np.exp(xg_model_load.predict(X_train))
#ls_pred_wat = np.exp(ls_model_load.predict(X_train))

rf_pred_wat = rf_model_load.predict(X_train)
xg_pred_wat = xg_model_load.predict(X_train)
ls_pred_wat = ls_model_load.predict(X_train)


# In[ ]:


#Defining a rmse calculator to check training accuracy
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


x1 = rf_pred_wa
x2 = xg_pred_wa
x3 = ls_pred_wa
#y_actual = np.exp(y_train)

#df_stack = pd.DataFrame(x1,x2,x3,y_actual, columns=['RandomForest','XGBoost','LassoRegression','ActualValues'])
df_stack_tst = pd.DataFrame(x1,columns=['RandomForest'])
df_stack_tst['XGBoost'] = x2
df_stack_tst['LassoRegression'] = x3
#df_stack_tst['ActualValues'] = y_train
df_stack_tst.head(5)


# In[ ]:


y1 = rf_pred_wat
y2 = xg_pred_wat
y3 = ls_pred_wat
#y_actual = np.exp(y_train)

#df_stack = pd.DataFrame(x1,x2,x3,y_actual, columns=['RandomForest','XGBoost','LassoRegression','ActualValues'])
df_stack_trn = pd.DataFrame(x1,columns=['RandomForest'])
df_stack_trn['XGBoost'] = y2
df_stack_trn['LassoRegression'] = y3
df_stack_trn['ActualValues'] = y_train
df_stack_trn.head(5)


# In[ ]:


results = pd.DataFrame(columns=['XGBoost Weight','Lasso Weight','Training RMSLE'])
results['XGBoost Weight'] = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
results['Lasso Weight'] = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
results


# In[ ]:


wghts = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
index = [0,1,2,3,4,5,6,7,8,9]

dfr = []
x = 0
for i in wghts:
    stacked1 =  (df_stack_trn['XGBoost']*i + df_stack_trn['LassoRegression']*(1.0-i))
    r = rmsle(y_train,stacked1)
    dfr.append(r)
    #results.ix[index,'Lasso Weight'] = (1.0-i)
    #results.ix[index,'Training RMSLE'] = r
    
#stacked1.head(5)
dfr


# In[ ]:


results['Training RMSLE'] = dfr
results


# From this table I can see that the XGBoost model performs better on the training set. The RMSLE score reduces as the weight on the XGBoost model increases. From the earlier bar plots we had seen that the XGBoost model had a higher RMSLE score for the training set than the test set. This might be due to some level of overfitting since the XGBoost model was cross validated on the training set. The Lasso model does have a lower RMSLE error for the test set compared to the Training set. Hence I feel like it should have a slightly higher weightage than 0.0. I will test the RMSLE scores for the test data from Kaggle for the following stacked models:
# 
# Model 1 :: XGBoost(0.9) + Lasso(0.1)
# Model 2 :: XGBoost(0.7) + Lasso(0.3)
# Model 3 :: XGBoost(0.6) + Lasso(0.4)
# Model 4 :: XGBoost(0.2) + Lasso(0.8)

# In[ ]:


i = 0.9
model_1 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_1 = pd.DataFrame(np.exp(model_1), columns = ['SalePrice'])
stack_pred_1['Id'] = df_test['Id']
stack_pred_1.to_csv('Stacked_Model_1.csv')
stack_pred_1.head(5)


# In[ ]:


i = 0.7
model_2 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_2 = pd.DataFrame(np.exp(model_2), columns = ['SalePrice'])
stack_pred_2['Id'] = df_test['Id']
stack_pred_2.to_csv('Stacked_Model_2.csv')
stack_pred_2.head(5)


# In[ ]:


i = 0.6
model_3 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_3 = pd.DataFrame(np.exp(model_3), columns = ['SalePrice'])
stack_pred_3['Id'] = df_test['Id']
stack_pred_3.to_csv('Stacked_Model_3.csv')
stack_pred_3.head(5)


# In[ ]:


i = 0.2
model_4 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_4 = pd.DataFrame(np.exp(model_4), columns = ['SalePrice'])
stack_pred_4['Id'] = df_test['Id']
stack_pred_4.to_csv('Stacked_Model_4.csv')
stack_pred_4.head(5)


# Next I uploaded each of these four models to Kaggle to get the test data RMSLE scores. Honestly I did not expect to see a drastic improvement, but i still wanted to explore how stacking/combining the models would affect the overall performance.

# The output of all 4 stacked models are shown below:

# In[ ]:


from IPython.display import Image
Image(filename='All_Stacked.jpg')


# The output for the 4th Stacked model is hown below. It socred the best at 0.12526, a 40.5% improvement over the benchmark score of 0.17649. This model advanced me in terms of relative scoring by 76 places from the previous best performance of 0.12671 achieved by the Lasso Regression.

# In[ ]:


from IPython.display import Image
Image(filename='Stackd_Model_4.jpg')


# The 4th stacked model was the best amongst them all. Just to be sure I will check one more Stacked model with the following configuration:
# Model 5 :: XGBoost(0.1) + Lasso(0.9)

# In[ ]:


i = 0.1
model_5 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_5 = pd.DataFrame(np.exp(model_5), columns = ['SalePrice'])
stack_pred_5['Id'] = df_test['Id']
stack_pred_5.to_csv('Stacked_Model_5.csv')
stack_pred_5.head(5)


# In[ ]:


from IPython.display import Image
Image(filename='Stackd_Model_5.jpg')


# This did not improve the score so it makes more sense to explore models which are some more in between the Lasso and XGBoost mdoels.The next model will be an averaged out:
# Model 6 :: XGBoost(0.5) + Lasso(0.5)

# In[ ]:


i = 0.5
model_6 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_6 = pd.DataFrame(np.exp(model_6), columns = ['SalePrice'])
stack_pred_6['Id'] = df_test['Id']
stack_pred_6.to_csv('Stacked_Model_6.csv')
stack_pred_6.head(5)


# This did not improve the score. Stacked Model 6 had a score of 0.12714.
# I will now check two more configurations:
# Model 7 :: XGBoost(0.3) + Lasso(0.7)
# Model 8 :: XGBoost(0.4) + Lasso(0.6)

# In[ ]:


i = 0.3
model_7 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_7 = pd.DataFrame(np.exp(model_7), columns = ['SalePrice'])
stack_pred_7['Id'] = df_test['Id']
stack_pred_7.to_csv('Stacked_Model_7.csv')
stack_pred_7.head(5)


# In[ ]:


i = 0.4
model_8 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)
#odel_1 = np.exp(model_1)
stack_pred_8 = pd.DataFrame(np.exp(model_8), columns = ['SalePrice'])
stack_pred_8['Id'] = df_test['Id']
stack_pred_8.to_csv('Stacked_Model_8.csv')
stack_pred_8.head(5)


# In[ ]:


#Model 7 :: XGBoost(0.3) + Lasso(0.7)
from IPython.display import Image
Image(filename='Stackd_Model_7.jpg')


# In[ ]:


#Model 8 :: XGBoost(0.4) + Lasso(0.6)
from IPython.display import Image
Image(filename='Stackd_Model_8.jpg')


# As a final step will create some visualizations to summarize the test data RMSLE scores obtained from Kaggle. I will also use a plot of predictions on the training data superimposed on the training response label using the best performer which was  the Stacked Model no.4

# In[ ]:


i = 0.2
stacked_pred_trn = df_stack_trn['XGBoost']*i + df_stack_trn['LassoRegression']*(1.0-i)

pred_comp1 = pd.DataFrame(np.exp(stacked_pred_trn), columns = ['Stacked Predicted_SalePrice'])
pred_comp1['Actual_SalePrice'] = target_variable
pred_comp1['Id'] = df_test['Id']
pred_comp1.head(5)


# In[ ]:



stacked_pred_trn_ls = ls_model_load.predict(X_train)

pred_comp2 = pd.DataFrame(np.exp(stacked_pred_trn_ls), columns = ['Lasso Predicted_SalePrice'])
pred_comp2['Actual_SalePrice'] = target_variable
pred_comp2['Id'] = df_test['Id']
pred_comp2.head(5)


# In[ ]:



stacked_pred_trn_lm = lm.predict(df_lin_test.drop('Id',1))

pred_comp3 = pd.DataFrame(np.exp(stacked_pred_trn_lm), columns = ['Benchmark Predicted_SalePrice'])
pred_comp3['Actual_SalePrice'] = target_variable
pred_comp3['Id'] = df_test['Id']
pred_comp3.head(5)


# In[ ]:


#Plotting these results to check how the model fot's the training data
x = pred_comp1['Actual_SalePrice']
y = pred_comp1['Stacked Predicted_SalePrice']
z = pred_comp2['Lasso Predicted_SalePrice']
w = pred_comp3['Benchmark Predicted_SalePrice']
ids = pred_comp1.shape[0]/2
fig,ax1 = plt.subplots(figsize=(25,35))

ax1 = fig.add_subplot(311)
ax1.scatter(x[:ids], y[:ids], s=10, c='b', marker="s", label='Actual SalePrice')
ax1.scatter(x[ids:],y[ids:], s=10, c='r', marker="o", label='Stacked Predicted SalePrice')
plt.legend(loc='upper left');

ax1 = fig.add_subplot(312)
ax1.scatter(x[:ids], z[:ids], s=10, c='b', marker="s", label='Actual SalePrice')
ax1.scatter(x[ids:],z[ids:], s=10, c='r', marker="o", label='Lasso Predicted SalePrice')

ax1 = fig.add_subplot(313)
ax1.scatter(x[:ids], z[:ids], s=10, c='b', marker="s", label='Actual SalePrice')
ax1.scatter(x[ids:],w[ids:], s=10, c='r', marker="o", label='Benchmark Predicted SalePrice')

plt.legend(loc='upper left');

plt.show()


# I can see from these plots that both the stacked and Lasso mooels had good fits to the training data while the Benchmark model had a very low degree of fit. These observations were reinforced by the RMSLE scores on the test data. I will also use a bar chart to depict the progression of the test set RMSLE scores obtained from Kaggle for all the models I trained.

# In[ ]:


final_df_models = ['Benchmark Model','Random Forest Try 1','Random Forest Try 2','Gradient Boosting XGBoost','Lasso Regression','Stacked Model 1 :: XGBoost(0.9) + Lasso(0.1)','Stacked Model 2 :: XGBoost(0.7) + Lasso(0.3)','Stacked Model 3 :: XGBoost(0.6) + Lasso(0.4)','Stacked Model 4 :: XGBoost(0.2) + Lasso(0.8)','Stacked Model 5 :: XGBoost(0.1) + Lasso(0.9)','Stacked Model 6 :: XGBoost(0.5) + Lasso(0.5)','Stacked Model 7 :: XGBoost(0.3) + Lasso(0.7)','Stacked Model 8 :: XGBoost(0.4) + Lasso(0.6)']
final_df_scores = [0.17649,0.20190,0.14544,0.14029,0.12671,0.13677,0.13101,0.12883,0.12526,0.12571,0.12714,0.12535,0.12598]

final_sum_df = pd.DataFrame(np.column_stack([final_df_models, final_df_scores]),columns = ['Models','Test data RMSLE from Kaggle'])

final_sum_df['Test data RMSLE from Kaggle'] = final_sum_df['Test data RMSLE from Kaggle'].astype('float64')
final_sum_df


# In[ ]:


pos = list(range(len(final_sum_df['Models']))) 
width = 0.2 

# Plotting the bars
fig, ax = plt.subplots(figsize=(20,20))

# Create a bar with train_rmsle_score data,
# in position pos,
plt.bar(pos, 
        #using accuracies['Train_accuracy'] data,
        final_sum_df['Test data RMSLE from Kaggle'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=final_sum_df['Models'][0]) 



# Set the y axis label
ax.set_ylabel('Accuracy Scores')

# Set the chart's title
ax.set_title('Model Accuracy Scores for Test sets.')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(final_sum_df['Models'], rotation=45, ha='right')

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 0.05+max(final_sum_df['Test data RMSLE from Kaggle'])])

# Adding the legend and showing the plot
plt.legend(['Training Accuracy'], loc='upper left')
plt.grid()
plt.show()


# In[ ]:


lasso_coef = ls_model_load.coef_


# In[ ]:


lasso_cols = X_train.columns
lasso_cols.shape


# In[ ]:


lasso_df = pd.DataFrame(np.column_stack([lasso_cols,lasso_coef]), columns = ['Feature','Lasso Coefficient'])
lasso_df.sort_values(by ='Lasso Coefficient', ascending = False).head(25)


# In[ ]:


final_df_models1 = ['Benchmark Model','Random Forest Try 1','Random Forest Try 2','Gradient Boosting XGBoost','Lasso Regression','Stacked Model 1 :: XGBoost(0.9) + Lasso(0.1)','Stacked Model 2 :: XGBoost(0.7) + Lasso(0.3)','Stacked Model 3 :: XGBoost(0.6) + Lasso(0.4)','Stacked Model 4 :: XGBoost(0.2) + Lasso(0.8)','Stacked Model 5 :: XGBoost(0.1) + Lasso(0.9)','Stacked Model 6 :: XGBoost(0.5) + Lasso(0.5)','Stacked Model 7 :: XGBoost(0.3) + Lasso(0.7)','Stacked Model 8 :: XGBoost(0.4) + Lasso(0.6)']
final_df_scores1 = [0.17649,0.20190,0.14544,0.14029,0.12671,0.13677,0.13101,0.12883,0.12526,0.12571,0.12714,0.12535,0.12598]
final_df_ranks1 = [1800,1800,1395,1318,902,902,902,902,839,839,839,839,839]

final_sum_df1 = pd.DataFrame(np.column_stack([final_df_models1, final_df_scores1, final_df_ranks1]),columns = ['Models','Test data RMSLE from Kaggle','Leadership Board Ranking'])

final_sum_df1['Test data RMSLE from Kaggle'] = final_sum_df1['Test data RMSLE from Kaggle'].astype('float64')
final_sum_df1['Leadership Board Ranking'] = final_sum_df1['Leadership Board Ranking'].astype('int64')

final_sum_df1.sort_values(by = 'Leadership Board Ranking', ascending = False)
final_sum_df1


# In[ ]:


fig, ax1= plt.subplots(figsize=(20,20))
ax2 = ax1.twinx()  # set up the 2nd axis

ax1.bar(pos, 
        #using accuracies['Train_accuracy'] data,
        final_sum_df1['Test data RMSLE from Kaggle'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.4, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=final_sum_df1['Models'][0]) 

ax2.plot(final_df_ranks1)
ax1.axes.set_xticklabels(final_sum_df1['Models'], rotation = 45, ha = 'right')
ax1.xaxis.set_visible(True)

ax1.set_ylabel('RMSLE Scores')
ax2.set_ylabel('Leaderboard Rankings Scores')
# Set the chart's title
ax1.set_title('Model Test RMSLE Scores and Kaggle Leaderboard ranking progression.')

plt.xlim(min(pos)-width, max(pos)+width*4)
ax1.set_xticks([p + 1.5 * width for p in pos])
plt.legend(['Training Accuracy', 'Leaderboard Ranking'], loc='upper left')

#ax2.set_xticklabels(final_sum_df1['Models'], rotation=45, ha='right')


# ###########################################################################################################################
