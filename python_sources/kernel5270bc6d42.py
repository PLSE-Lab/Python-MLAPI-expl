#!/usr/bin/env python
# coding: utf-8

# # The Housing price of Ames, Iowa, USA
# This is my very first problem solving in the Regression space. Before this, it was all theory; and the very first realization was that, even in a friendly competition like this one in Kaggle, one has a lot to struggle and learn.
# 
# My humble admission is that I have learn a lot from other people's work.<br> 
# However, I have also added a few perspective of mine. I hope some new-comer might be benefitted from my work that I am publishing.
# 
# Please feel free to use my notebook as a baseline and make more improvements on the same. Push your suggestions, comments to me as it is also my starting journey in the world of Data Science.<br>
# Also, in case, you like some part of my work, please do upvote. Thank you in advance !
# 
# ## The Goal
# * Each row in the dataset describes the characteristics of a house.
# * The characteristics is indicated/captured by 80 features e.g Neighborhood, Utilities, Landslope, Bedrooms, Kitchens, heating etc. (More fully available in the Data Description text file ) 
# * Our goal is to predict the SalePrice, given these features.
# * Our models are evaluated on the Root-Mean-Squared-Error (RMSE) between the log of the SalePrice predicted by our model, and the log of the actual SalePrice.
# 
# ## Salient features of this work:
# 1. I used Visualization techniques in the Exploratory Data Analysis (EDA) stage to present the data in a concise manner so that a person can capture a good view of the data with very little scrolling. ***This give a good perspective or bird-eye view of the data***
# 2. In the Feature Engineering stage, I used a lot of generic techniques to fill the missing values. **Very little 'Hard-coding' has been used**. This will ensure that the notebood can be used on a different set of data whose data elements might be the same but the characteristics of the data is very different.
# 3. Lastly, in the model-fitting stage, I have used a wighted average of the models that are used. Since, the belended model provided a better RMSE score, I used that in teh final prediction of the price.

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# Creating  a new Worksheet after learing few new techniques from other notebooks that have been shared within the community. I have also worked on few improvisation over the learning from other notebook. 

# ### **First import the necessary libraries **

# In[ ]:


# Import section

import pandas as pd
import numpy as np

import seaborn as sns
import types
import pandas as pd
from botocore.client import Config
#import ibm_boto3

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Misc
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from datetime import datetime

pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# Read the Training dataset into a dataframe

# In[ ]:


trainDf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


trainDf.head(10)


# Next read the Test dataset into another Dataframe

# In[ ]:


testDf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


testDf.head()


# # 1. Exploratory Data Analysis (EDA)

# ## 1.1 Target Attribute (SalePrice) Observation

# In[ ]:


# First let us look at the Target and plot it visually
import matplotlib.pyplot as plt

sns.set_style ("white")
sns.set_color_codes (palette = 'deep')
f, ax = plt.subplots (figsize=(8, 7))

sns.distplot (trainDf['SalePrice'], color='b')
ax.xaxis.grid =False
ax.set (ylabel='Frequency' )
ax.set (ylabel='SalePrice' )
ax.set (title = 'SalePrice Distribution')
sns.despine(trim=True, left=True)
plt.show()


# In[ ]:


# Skewness and Kurtosis

print ("Skewness of Data : %.2f" % trainDf['SalePrice'].skew())
print ("Kurtosis of Data : %.2f" % trainDf['SalePrice'].kurt())


# **Observation** : The data distribution is skewed towards Right. In other words, the tail is towards the right

# ## 1.2 Feature Observation

# Let us observe some of the features in the training dataset

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.2f' %x)
trainDf.describe()


# In[ ]:


trainDf.describe(include = ['object'], exclude = ['int', 'float'])


# **Observation**:<br>
# The attributes **Street**, **Utilities**, **Condition2**, **RoofMatl**, **Heating** have very ***little variation***. Most of the data is in one category. Suspicion is that they might cause undue influence on the model.

# ### Let us now observe how the SalePrice is related/influenced to some of the Features.<br>
# Let us start with two Features namely OverallQual and OverallCond

# In[ ]:


fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(10, 8))
fig.tight_layout(pad=6.0)
sns.boxplot(x='OverallQual', y='SalePrice', data=trainDf, orient='v', ax=axes[0])
sns.boxplot(x='OverallCond', y='SalePrice', data=trainDf, orient='v', ax=axes[1])


# **Observation** :The feature OverallQual has less of overlaps but OverallCond have more overlap between the categories
# Also, in OverallCond, for the value 5, we have huge outliers

# Let us observe the Numeric features. Some Numeric can be continuous and some will be Categorical

# In[ ]:


NumericColumns = trainDf.select_dtypes([np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]).columns
NumericColumns
FeaturePlot = NumericColumns.drop([ 'SalePrice'])


# In[ ]:


ColumnDisplay = 2
fig, axex = plt.subplots( ncols=ColumnDisplay, nrows=0, figsize=(12,120) )

plt.subplots_adjust (top=2, right=2)
sns.color_palette ('husl', 8)
for i, feature in enumerate (list(trainDf[FeaturePlot]), 1) :
    plt.subplot (len(list(FeaturePlot)), ColumnDisplay, i )
    sns.scatterplot (x=feature, y='SalePrice', data=trainDf, hue='SalePrice', palette='Blues')


# **Key Observations**
# The belowFeatures appear to be Categorical and not Continuous
# 
# <li>MSSubClass</li>
# <li>OverallQual</li>
# <li>OverallCond</li>
# <li>LowQualFinSF</li>
# <li>BsmtFullBath</li>
# <li>BsmtHalfBath</li>
# <li>FullBath</li>
# <li>HalfBath</li>
# <li>BedroomAbvGr</li>
# <li>KitchenAbvGr</li>
# <li>TotalRmsAbvGnd</li>
# <li>Fireplaces</li>
# <li>GarageCars</li>
# <li>MoSold</li>
# <li>YrSold</li>

# In[ ]:


CategoricColumns = trainDf.select_dtypes([np.object]).columns
#CategoricColumns


# Let us visualize the Categorical Features especially that are of type Object/String

# In[ ]:


DisplayColumns=2
fig, axes = plt.subplots (ncols=DisplayColumns, nrows=0, figsize=(12, 120))
plt.subplots_adjust (top=2, right=2)
sns.color_palette('RdGy', 10)

for i, feature in enumerate (list(trainDf[CategoricColumns]), 1) :
    plt.subplot (len(list(CategoricColumns)), ColumnDisplay, i )
    sns.boxplot (x=feature, y='SalePrice', data=trainDf, orient='v')


# **Observations**
# 
# <li>LotShape - Has a lot of overlap across two categories</li>
# <li>LandContour - Has a lot of overlap across 4 categories</li>
# <li>LotConfig - Has a lot of overlap</li>
# <li>LandSlope - Has a lot of overlap</li>
# <li>BsmtFinType1 and BsmtFinType2 have lot of overlaps</li>
# <li>Functional has overlaps across 7 categories</li>

# # 2. Feature Engineering

# As part of the Feature Engineering, we need to carry out steps to make the data ready for feeding into the Algorithm.
# This will involve the Observations made in the EDA step. Feature Engineering will involve :
# 
# Removing Features that do not seem to add value, rather might make the Algorithm pick up the noise
# <li>Removing Outliers if required</li>
# <li>Finding out NaN (Null) values and see how to update them without distorting the sets</li>
# <li>Fixing Skewness of the data</li>

# **Let us re-look at the Target once more i.e the SalePrice. We noticed that it had a tail towards the right or in other words it had a positive skew of 1.88.**<br>
# Most models do not perform well if the data-distribution is not normal.
# So we apply log(1+x) transformation on SalePrice

# In[ ]:


trainDf['TranSalePrice'] = np.log1p(trainDf['SalePrice'])


# In[ ]:


trainDf[['Id', 'SalePrice', 'TranSalePrice']].head()


# **Now let us plot the two bar-charts side-by-side to visualize the transformation**

# In[ ]:


fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(8, 6))

fig.tight_layout(pad=4.0)

#Set the generic properties of Seaborn
sns.set_style("white")
sns.set_color_codes (palette = 'deep')
sns.despine(trim=True, left=True)

# The first distribution plot is for the original SalePrice data
sns.distplot(trainDf['SalePrice'], color="b", ax=axes[0]);
#ax.grid(False)
axes[0].set(ylabel="Frequency")
axes[0].set(xlabel="SalePrice")
#axes[0].xticks(rotation=90)
axes[0].set(title="SalePrice distribution-Original")

# The Second distribution plot is for the original SalePrice data
sns.distplot(trainDf['TranSalePrice'], fit=norm, color="g", ax=axes[1]);
(mu, sigma) = norm.fit(trainDf['TranSalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#ax.xaxis.grid(False)
axes[1].set(ylabel="Frequency")
axes[1].set(xlabel="SalePrice")
axes[1].set(title="SalePrice distribution-Transform")

plt.show()


# **As a Next step, let us merge the Test data with the Train data.**
# We can later seperate them out.This will keep the two dataset in sync.
# Also the operations can be done in one iteration.

# ## 2.1 Remove Outliers

# In[ ]:


# Remove outliers

trainDf.drop(trainDf[(trainDf['GrLivArea']>4000) & (trainDf['SalePrice']<300000)].index, inplace=True)
trainDf.reset_index(drop=True, inplace=True)


# **Now let us combine the two datasets into a single dataset**

# In[ ]:


trainEnd = trainDf.shape[0] #retain the count for segregation in future
CombineDf = pd.concat ([trainDf, testDf], sort=True).reset_index(drop=True)
#trainEnd


# In[ ]:


CombineDf.head()


# In[ ]:


CombineDf.drop(['Id', 'SalePrice', 'TranSalePrice'], axis=1, inplace=True)


# ## 2.1 Dealing with missing Values (NaN)

# In[ ]:


# Let us check the extent of values that are NaN (missing Values)

# Create a subroutine to list down the NaN % in a tabular form. This subroutine will be invoked multiple times.
def ListEmptiness (df) :
    CombineNaN = (df.isnull().sum()/df.shape[0]) * 100 # Get the % of the Attributes that have Null value
    CombineNaN = CombineNaN[CombineNaN !=0].sort_values(ascending=False)
    nanData = pd.DataFrame({'Nan Ratio': CombineNaN})
    return nanData


# In[ ]:


Emptyness = ListEmptiness (CombineDf)
#Emptyness


# In[ ]:


#Visualize the Missing Attributes
f,ax = plt.subplots (figsize=(10,8))
sns.barplot (y='Nan Ratio', x=Emptyness.index, data=Emptyness)
plt.xticks(rotation=90);
plt.ylabel('Percentage of Missing data in the Feature')
plt.xlabel('Features')
plt.title('Missng data by Feature');


# ### We have to now take the features on a case-by-case basis and handle them.<br>
# 

# 1. PoolQC

# In[ ]:


CombineDf['PoolQC'].value_counts(dropna=False).to_frame()


# A huge majority **(2907 / 2917)** of the housed do not have the value captured for PoolQC. We could have dropped this Feature altogether since we have another Feature called PoolCond
# However, we are going to make NaN = "None" for the feature PoolQC. This is as per the Data Description File.
# What matters is whether a Pool exist or not

# 2. PoolArea

# In[ ]:


CombineDf['PoolArea'].describe()


# Based on the above two observation related to Pool (Swimming Pool), it is recommended that the PoolQC be made "none" for the missing values

# In[ ]:


CombineDf['PoolQC'] = CombineDf['PoolQC'].fillna("None")


# 2. MiscFeature

# In[ ]:


CombineDf['MiscFeature'].value_counts(dropna=False).to_frame()


# A huge majority **(2812 / 2917)** of the housed do not have the value captured for MiscFeature. We are going to make NaN = "None" for the feature MiscFeature. This is as per the Data Description File.
# **This is also a potential Feature that can be dropped**

# In[ ]:


CombineDf['MiscFeature'] = CombineDf['MiscFeature'].fillna("None")


# 3. Alley- No data is construed to be "None"

# In[ ]:


CombineDf['Alley'] = CombineDf['Alley'].fillna("None")


# 4. Fence - No data is construed to be "None"

# In[ ]:


CombineDf['Fence'] = CombineDf['Fence'].fillna ("None")


# 5. FireplaceQu - Fill the NaNs with "None" as per the data description

# In[ ]:


CombineDf['FireplaceQu'] = CombineDf['FireplaceQu'].fillna ("None")


# 6.LotFrontage -<br>
# LotFrantage is a Continuous Feature. As per Data Description, it is the Linear feet of street connected to property
# It will be tempting to assign the mean value of the dataset to this feature with the missing values.
# ***However, a better way is to find the Median of each Neighborhood and assign the same to the missing rows based on Neighborhood.***
# The assumption being that the LotFrontage will be similar within a Neighborhood.
# 

# In[ ]:


CombineDf['LotFrontage'] = CombineDf.groupby('Neighborhood')['LotFrontage'].transform (lambda x:x.fillna(x.median()))


# 7. GarageType<br>
# 8. GarageFinish<br>
# 9. GarageQual<br>
# 10. GarageCond<br>
# For the four categorical features above (all related to Garage), NaN mean No Garage. So, empty rows will be filled with "None"

# In[ ]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        CombineDf[col] = CombineDf[col].fillna("None")


# 11. GarageYrBlt<br>
# 12. GarageArea<br>
# 13. GarageCars<br>
# 
# The above three Numeric Features are filled with 0 for the missing rows. 0 means No Garage.

# In[ ]:


# Replacing the missing values with 0, since no garage = no cars in garage
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        CombineDf[col] = CombineDf[col].fillna(0)


# 14. BsmtQual<br>
# 15. BsmtCond<br>
# 16. BsmtExposure<br>
# 17. BsmtFinType1<br>
# 18. BsmtFinType2<br>
# 
# For the above Categorical Features related to Basedment, NaN means No Basement

# In[ ]:


# NaN values for these categorical basement features, means there's no basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    CombineDf[col] = CombineDf[col].fillna('None')


# In[ ]:


# Create a Multi-culumn display to visualize same category columns
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
        html_str+="<td>&nbsp&nbsp&nbsp</td>"
    #print(html_str)
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# In[ ]:


display_side_by_side (CombineDf['Electrical'].value_counts(dropna=False).to_frame(),                       CombineDf['Functional'].value_counts(dropna=False).to_frame(),                       CombineDf['Utilities'].value_counts(dropna=False).to_frame(),                       CombineDf['SaleType'].value_counts(dropna=False).to_frame(),                       CombineDf['KitchenQual'].value_counts(dropna=False).to_frame()
                     )


# 19. Electrical<br>
# 20. Functional<br>
# 21. Utilities<br>
# 22. SaleType<br>
# 23. KitchenQual<br>
# 
# The above Features have one or two rows of missing data. The best solution will be to update those with the Mode of the dataset

# In[ ]:


ColumnList = {'Electrical', 'Functional', 'Utilities', 'SaleType', 'KitchenQual' }
for col in ColumnList :
    #print (CombineDf[col].mode()[0])
    CombineDf[col] = CombineDf[col].fillna (CombineDf[col].mode()[0] )


# In[ ]:


display_side_by_side (
                      CombineDf['Exterior1st'].value_counts(dropna=False).to_frame(), \
                      CombineDf['Exterior2nd'].value_counts(dropna=False).to_frame() \
                     )


# 24. Exterior1st<br>
# 25. Exterior2nd<br>
# 
# The above Features have one or two rows of missing data. The best solution will be to update those with the Mode of the dataset

# In[ ]:


ColumnList = { 'Exterior1st', 'Exterior2nd' }
for col in ColumnList :
    #print (CombineDf[col].mode()[0])
    CombineDf[col] = CombineDf[col].fillna (CombineDf[col].mode()[0] )


# 26. BsmtFinSF1<br>
# 27. BsmtFinSF2<br>
# 28. BsmtFullBath<br>
# 29. BsmtHalfBath<br>
# 30. BsmtUnfSF<br>
# 31. TotalBsmtSF<br>
# Above six numerical Continuous features are updated with zero(0)

# In[ ]:


CombineDf['BsmtFinSF1' ] = CombineDf['BsmtFinSF1'].fillna(0)

CombineDf['BsmtFinSF2' ] = CombineDf['BsmtFinSF2'].fillna(0)

CombineDf['BsmtFullBath' ] = CombineDf['BsmtFullBath'].fillna(0)
CombineDf['BsmtHalfBath' ] = CombineDf['BsmtHalfBath'].fillna(0)
CombineDf['BsmtUnfSF' ] = CombineDf['BsmtUnfSF'].fillna(0)
CombineDf['TotalBsmtSF' ] = CombineDf['TotalBsmtSF'].fillna(0)


# In[ ]:


display_side_by_side (CombineDf['MasVnrType'].value_counts(dropna=False).to_frame(),                       #CombineDf['MasVnrArea'].value_counts(dropna=False).to_frame(), \
                      CombineDf['MSZoning'].value_counts(dropna=False).to_frame() )


# 32. MasVnrType<br>
# 33. MasVnrArea<br>
# 34. MSZoning<br>
# 
# The last remaining Features.
# MasVnrType - To be filled by the Mode of the dataset
# MasVnrArea - To be filled by zero (0)
# 

# In[ ]:


CombineDf['MasVnrType'] = CombineDf['MasVnrType'].fillna(CombineDf['MasVnrType'].mode()[0])

CombineDf['MasVnrArea'] = CombineDf['MasVnrArea'].fillna(0)


# **MSZoning**  
# There is no easy way to default this. The best approximation is to take the Mode of each Neighborhood.<br>
# This is not a perfect logic, but the this will ensure that for a particular Neighborhood, <br>
# if it is primarily Residential, then the default will be Residential, if it is primarily Commercial then the default will be Commercial  and so on.
# 

# In[ ]:


CombineDf['MSZoning'] = CombineDf.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


# Let's make sure we handled all the missing values

Emptyness = ListEmptiness (CombineDf)
Emptyness


# ## 2.2 Dealing with Skewed data

# In[ ]:


NumericColumns = CombineDf.select_dtypes([np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]).columns
#NumericColumns


# In[ ]:


# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=CombineDf[NumericColumns] , orient="h", palette="RdGy")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[ ]:


SkewFeatures = CombineDf[NumericColumns].apply(lambda x: skew(x)).sort_values (ascending=False)

HighSkews = SkewFeatures[SkewFeatures > 0.5]
SkewIndex = HighSkews.index

print('There are {} Numerical Features with High Skew Values'.format(SkewIndex.shape[0]))


# **Let us use the scipy function boxcox1p which does Box-Cox transformation**

# In[ ]:


for i in SkewIndex :
    CombineDf[i] = boxcox1p( CombineDf[i], boxcox_normmax(CombineDf[i] +1) )


# In[ ]:


SkewFeatures = CombineDf[NumericColumns].apply(lambda x: skew(x)).sort_values (ascending=False)

HighSkews = SkewFeatures[SkewFeatures > 0.5]
SkewIndex = HighSkews.index

print('There are {} Numerical Features with High Skew Values'.format(SkewIndex.shape[0]))


# **By Applying boxcox1p , we have reduced the Skewness of 10 features.**

# ## 2.2 Create Additional features (including conversion from String/Object type to numeric

# 1. YearSinceRemodel - Captures how recently the house was touched upon<br>
# 2. Total Home Quality - It combines OverallQual and OverallCond through a simple addition logic<br>
# 3. Total Square Ft of the house

# In[ ]:


CombineDf['YearsSinceRemodel'] = CombineDf['YrSold'].astype(int) - CombineDf['YearRemodAdd'].astype(int)
CombineDf['Total_Home_Quality'] = CombineDf['OverallQual'] + CombineDf['OverallCond']
CombineDf['TotalSF'] = CombineDf['TotalBsmtSF'] + CombineDf['1stFlrSF'] + CombineDf['2ndFlrSF']
CombineDf['Total_Bathrooms'] = (CombineDf['FullBath'] + (0.5 * CombineDf['HalfBath']) +                               CombineDf['BsmtFullBath'] + (0.5 * CombineDf['BsmtHalfBath']))


# ## 2.3 Drop some of the features that are Categorical, String type and having extreme Bias towards one value

# In[ ]:


CombineDf = CombineDf.drop(['Utilities', 'Street', 'PoolQC',], axis=1)


# ## 2.4 Get Correlation between Features and also between Features and SalePrice

# In[ ]:


train= pd.concat([CombineDf.iloc[:trainEnd, :], trainDf['TranSalePrice']], axis=1)
train_corr = train.corr()


# ### 2.4.1 Visualize the correlation

# In[ ]:


mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette (180, 30, as_cmap=True)

with sns.axes_style("white"):
     fig, ax = plt.subplots(figsize=(13,11))
     sns.heatmap(train_corr, vmax=.8, mask=mask, cmap=cmap, cbar_kws={'shrink':.5}, linewidth=.05);


# ## 2.5 And Finally the One-hot Encoding for the categorical values of Object/String type

# In[ ]:


DummyCombineDf = pd.get_dummies(CombineDf)


# In[ ]:


print ('Shape of the Dataset (Train+ Test) Rows:{}, Columns:{}'.format(DummyCombineDf.shape[0], DummyCombineDf.shape[1]))


# **With all Feature Engineering done, we shall split the Combined Dataset into Train and Test as per the original dataset**

# In[ ]:


X_Train = DummyCombineDf.iloc[ : trainEnd, :]
X_Test  = DummyCombineDf.iloc[trainEnd :, :] 

#y_train = trainDf[['TranSalePrice']]
y_train = trainDf['TranSalePrice'].reset_index(drop=True)


# In[ ]:


y_train.head(5)


# # 3.0 Setting up the Models

# In[ ]:


#Common Params
Kf = KFold(n_splits=4, random_state=42, shuffle=True) # Number of K-Folds

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
#                       objective='reg:linear',
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
#ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge_alphas = [ 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=Kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# In[ ]:


def cv_rmse(model, TrainFeature, TrainTarget):
    rmse = np.sqrt(-cross_val_score(model, TrainFeature, TrainTarget, scoring="neg_mean_squared_error", cv=Kf))
    return (rmse)


# In[ ]:



Scores = {}


print("About to start Scoring for First Set Algorithms Time:%s" % datetime.now())
for clf, label in zip([ridge, svr, rf], [ 'Ridge', 'SVM', 'Random Forest']):
    score = cv_rmse(clf, X_Train, y_train)
    print("Neg. MSE Score: %0.4f (+/- %0.4f) [%s] Time::%s" % ( score.mean(), score.std(), label, datetime.now()))
    Scores[label] = (score.mean(), score.std())
    


# In[ ]:


print("About to start Scoring for Second Set Algorithms Time:%s" % datetime.now())

for clf, label in zip([ xgboost, gbr, lightgbm], [ 'xgBoost', 'GradientBooster', 'lightGBM']):
    score = cv_rmse(clf, X_Train, y_train)
    print("Neg. MSE Score: %0.4f (+/- %0.4f) [%s] Time::%s" % ( score.mean(), score.std(), label, datetime.now()))
    Scores[label] = (score.mean(), score.std())


# In[ ]:





# In[ ]:


Scores


# # 6.0 Fit the Models

# In[ ]:


print (X_Train.shape, y_train.shape)


# In[ ]:


print('stack_gen Start Time:%s' %  datetime.now())
stack_gen_model = stack_gen.fit(np.array(X_Train), np.array(y_train) )
print('stack_gen End   Time:%s' % datetime.now())


# In[ ]:


print('lightgbm Start Time:%s' %  datetime.now())
lightgbm_gen_model = lightgbm.fit(X_Train, y_train )
print('lightgbm End   Time:%s' % datetime.now())


# In[ ]:


print('xgBoost Start Time:%s' %  datetime.now())
xgb_gen_model = xgboost.fit(X_Train, y_train )
print('xgBoost End   Time:%s' % datetime.now())


# In[ ]:


print('SVR Start Time:%s' %  datetime.now())
svr_gen_model = svr.fit(X_Train, y_train )
print('SVR End   Time:%s' % datetime.now())


# In[ ]:


print('Ridge Start Time:%s' %  datetime.now())
ridge_gen_model = ridge.fit(X_Train, y_train )
print('Ridge End   Time:%s' % datetime.now())


# In[ ]:


print('Random Forest Start Time:%s' %  datetime.now())
rf_gen_model = rf.fit(X_Train, y_train )
print('Random Forest End   Time:%s' % datetime.now())


# In[ ]:


print('GradientBoosting  Start Time:%s' %  datetime.now())
gbr_gen_model = gbr.fit(X_Train, y_train )
print('GradientBoosting  End   Time:%s' % datetime.now())


# ### **Blend Models to get a balanced prediction**

# In[ ]:


# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(XBlend):
    #print (XBlend.shape)
    #print(XBlend.columns)
    return (   (0.1  * ridge_gen_model.predict(XBlend))               + (0.2  * svr_gen_model.predict(XBlend))               + (0.1  * gbr_gen_model.predict(XBlend))              + (0.1  * xgb_gen_model.predict(XBlend))               + (0.1  * lightgbm_gen_model.predict(XBlend))               + (0.05 * rf_gen_model.predict(XBlend))               + (0.35 * stack_gen_model.predict(np.array(XBlend)))
            )


# In[ ]:





# In[ ]:


# Get final precitions from the blended model
print('Blended   Start Time:%s' %  datetime.now())
Blended_Yhat = blended_predictions(X_Train)
print('Blended   End Time:%s' %  datetime.now())


# In[ ]:


Blended_Yhat


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


blended_score = rmsle(y_train, blended_predictions(X_Train))

Scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# # 7.0 Evaluation of the Models Considered

# In[ ]:


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(Scores.keys()), y=[score for score, _ in Scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(Scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# As depicted in the Graph above, the blended model gives an output whose RMSE is far less compared to individual Models.<br>. Hence we should go with the blended model to predict the outcome

# # 8.0 Predict the Test Data 

# We shall use the Blended model to first predict the **Normalized** Price

# In[ ]:


#Apply Prediction to the Test dataset using the best Regression Algorithm (in this case Ridge)
testDf['TranSalePrice'] = blended_predictions(X_Test)


# Once the Normalized Price is predicted, we use the **inverse transformation** to derive the Real price from teh Normalized price

# In[ ]:


#Get the SalePrice via the inverse Normalization tranformation
testDf['SalePrice'] = np.floor(np.expm1(testDf['TranSalePrice']))


# Let us look at the Predicted "Normalized Price" and the "Real Price" side-by-side

# In[ ]:


testDf[['Id', 'TranSalePrice', 'SalePrice']].head(10).style.format({'SalePrice': "{:,.0f}"})


# 

# In[ ]:


my_submission = pd.DataFrame({'Id': testDf.Id, 'SalePrice': testDf.SalePrice})
# Use any filename. I choose submission here

my_submission.to_csv('submission.csv', index=False)

