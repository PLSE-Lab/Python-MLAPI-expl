#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


# COMMON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
from matplotlib import cm

# STATS
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import norm
from scipy.stats import skew

# SKLEARN MODELS
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler


# # Helper functions

# In[ ]:


# If passed a Dataframe:
#     Returns an index and the specified column with zeros and NaN removed
# If passed a Series:
#     Returns a series with zeros and NaN removed
def filterNanZeros(df, colname=False, nanOnly=False, inplace=False):
    if colname == False:
        s = df.copy()
        s.fillna(0, inplace=True)
        if not nanOnly:
            s = s[s != 0]
        return s
    else:
        if inplace == False:
            df = df.reset_index()[['index',colname]].copy()
        df[colname].fillna(0, inplace=True)
        if not nanOnly:
            df = df[df[colname] != 0]
        return df

# Gives the min value in a column that isn't 0 or NaN
def realMin(df, colname):
    return filterNanZeros(df, colname)[colname].min()

# Returns the difference between the max and non zero/Nan min of a column
def getColRange(df, colname=False):
    if colname == False:
        columns = []
        for col in list(df):
            columns = columns + [df[col].max() - realMin(df, col)]
        return np.array(columns)
    return df[colname].max() - realMin(df, colname)

# Returns all column names that have values within a range threshold and their range value
def getSmallCols(df, threshold):
    numericDtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    ranges = {}
    for colname in list(df):
        if df[colname].dtype in numericDtypes:
            colRange = getColRange(df, colname)
            if colRange < threshold:
                ranges[colname] = colRange
    return ranges

def getOutliers(df, threshold):
    dfCopy = df.copy()
    outliers = {}
    for colname in list(df):
        filteredDf = filterNanZeros(df, colname)
        filteredDf.iloc[:,1] = abs(stats.zscore(filteredDf.iloc[:,1])) > threshold
        dfCopy.update(filteredDf)
    for colname in list(dfCopy):
        dfCopy[colname] = dfCopy[colname].apply(lambda x: False if x != True else True)
    return dfCopy

def getOutliersModel(df, threshold):
    dfCopy = df.copy()
    outliers = {}
    for colname in list(df):
        filteredDf = filterNanZeros(df, colname)
        filteredDf.iloc[:,1] = abs(stats.zscore(filteredDf.iloc[:,1])) > threshold
        dfCopy.update(filteredDf)
    for colname in list(dfCopy):
        dfCopy[colname] = dfCopy[colname].apply(lambda x: False if x != True else True)
        dfCopy[colname] = dfCopy[colname].apply(lambda x: True if x == False else False)
    return dfCopy


def filterOutliers(df, dfOutliers):
    for colname in list(dfOutliers):
        df = df[list(dfOutliers[colname])]
        dfOutliers = dfOutliers[list(dfOutliers[colname])]
    return df

def convertColumnTypes(df, columnNames, dType):
    for colname in columnNames:
        df[colname] = df[colname].astype(dType, inplace=True)
    return df

timeStamp = datetime.now()
def getTimeElapsed():
    global timeStamp
    prev = timeStamp
    timeStamp = datetime.now()
    return timeStamp - prev

# Create data type lists
def getColumnsNamesByType(df, dTypes):
    columns = []
    for colname in df.columns:
        if df[colname].dtype in dTypes:
            columns.append(colname)
    return columns

def showPercentageMissing(df):
    rows, cols = df.shape
    trainIsNan = df.copy()
    trainIsNan = trainIsNan.applymap(lambda x: 0 if x != x and np.isnan(x) else 1)
    summed = (1 - trainIsNan.sum(axis=0)/rows)*100
    summed = summed[summed != 0]
    return summed.sort_values(ascending=False)

def import_data(path_train, path_test):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    return (train, test)

def boxCoxTransform(df, nanOnly=False):
    qBoxCox = df.copy()
    for colname in list(df):
        q = filterNanZeros(df,colname , nanOnly=nanOnly)
        q.iloc[:,1] = boxcox1p(q.iloc[:,1], boxcox_normmax(q.iloc[:,1] + 1))
        q.set_index('index', inplace=True)
        qBoxCox.update(q)
    return qBoxCox

def stripOutliers(arrays, outliers):
    for i in range(0, len(arrays)):
        arrays[i] = arrays[i].drop(arrays[i].index[outliers])
    return tuple(arrays)


# ## Scoring
# 
# Functions for evaluating model and prediction performance

# In[ ]:



# Creates an empty csv file for recording model performance scores
def create_scorecard(filename='scorecard.csv'):
    scorecard = pd.DataFrame(columns=['cv_score', 'comment', 'models', 'date'])
    scorecard.to_csv(filename, index=False)

# Records model performance metrics and prints difference against previous results
def record_score(score_path, cv_score, comment = '', model_names = []):
    
    from pathlib import Path
    my_file = Path(score_path)
    if not my_file.is_file():
        create_scorecard(score_path)
    
    scorecard = pd.read_csv(score_path)
    roundingVal = 10
    
    shape = scorecard.shape
    
    if shape[0] == 0:
        print('\t No previous score')
    else:
        best_cv_score_index = scorecard['cv_score'].idxmin()
        
        best_cv_score = scorecard.loc[best_cv_score_index]['cv_score']
        prev_cv_score = scorecard.loc[scorecard.shape[0]-1]['cv_score']

        cv_score_diff_vs_best = round(cv_score - best_cv_score,roundingVal)
        cv_score_diff_vs_prev = round(cv_score - prev_cv_score,roundingVal)

        print('\t cv_scores differences')
        print(f'\t\t cv_score diff vs best  :{cv_score_diff_vs_best}')
        print(f'\t\t cv_score diff vs prev  :{cv_score_diff_vs_prev}')
    
    cv_score = round(cv_score,roundingVal)
    
    new_score = pd.DataFrame(columns=['cv_score', 'comment','models', 'date'])
    new_score.loc[0] = [cv_score, comment, model_names, get_formatted_datetime()]
    
    new_scorecard = pd.concat([scorecard, new_score])
    new_scorecard.to_csv(score_path, index=False)

def get_formatted_datetime():
    return datetime.now().strftime("%y-%m-%d_%Hh%Mm%Ss")

# Returns the RMSLE of the difference between two submissions
# and an array of the differences
# and plots a Y_best vs Y_latest scatter graph
def compare_submissions(base_sub, pred_sub, col_name='SalePrice'):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    def rmsle(y_base, y_pred):
        return round(sqrt(mean_squared_error(np.log1p(y_base), np.log1p(y_pred))),10)
    
    if (not base_sub.shape == pred_sub.shape):
        raise ValueError('CSVs\' shape do not match')

    y_base = base_sub[col_name]
    y_pred = pred_sub[col_name]
    
    pltScatter(y_base, y_pred, X_label='Y_previous', Y_label='Y_predicted')
    
    return y_base - y_pred

# Cross validation scoring of models
def cv_scoring(models, X, y, models_to_test = ['ridge', 'lasso', 'elasticnet', 'svr', 'gbr', 'lightgbm', 'xgboost', 'stack_gen'], verbose=True):
    if verbose:
        print('Cross validation scoring running')
        
    meanScores = np.array([])
    weights = {}
    
    for modelName in models_to_test:
        score = cv_rmse(models[modelName], X, y)
        meanScores = np.append(meanScores,score.mean())
        weights[modelName] = score.mean()
        if verbose:
            print(f'\t {modelName} \t mean score: {round(score.mean(),5)} \t std.dev: ({round(score.std(),5)})')
    
    maxScore = meanScores.max()
    meanScore = meanScores.mean()
    
    for modelName in weights:
        weights[modelName] = maxScore - weights[modelName] * 0.9

    weightedSum = sum(weights.values())
    
    for modelName in weights:
        weights[modelName] = weights[modelName]/weightedSum
        
    if verbose:
        print(f'Scoring')
        print(f'\t cv_score :{meanScore}')
        
    return (meanScore, weights)

def cv_rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error"))
    return (rmse)

def pltScatter(X, Y, X_label='X', Y_label='Y'):
    fig = plt.figure()
    plt.scatter(X, Y)
    fig.suptitle(f'{Y_label} vs {X_label}')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()
    
def getOutlierIndexes(y_real, y_pred, outlierThreshold):
    absDiff = abs(y_real - y_pred)

    q = pd.DataFrame(absDiff)
    q['index'] = q.index
    q['zscore'] = abs(stats.zscore(absDiff))
    
    q.sort_values(by=['zscore'], ascending=False, inplace=True)
    
    outlierList = list(q[q['zscore'] > outlierThreshold]['index'])
    return (outlierList, q['zscore'])


# # Introduction

# # Exploratory Data Analysis
# 
# We will explore the data according thusly:
# 
# **Group the column types according to categories and themes**
# 
# This will help us better understand the data as well as make better choices later on.
# 
# **Look at the data types in the columns**
# 
# Important for the next step:
# 
# **Consider any data type conversions**
# 
# This will depend on what we think will be best for our model's predictive ability.
# 
# **Consider missing data**
# 
# Look at what is missing, and what to do with it.
# 
# **Calculate & visualise outliers**
# 
# This will help us decide the best outlier removal strategy.

# In[ ]:


# Import the datasets
train, test = import_data('../input/train.csv', '../input/test.csv')


# ## Data categories
# 
# According to the `data_description.txt` we can conclude that there are the following data categories present:
# 
# **T-CAT: Text representing a category**
# 
# `MSZoning: A	Agriculture`
# 
# **T-FEAT: Text representing a feature**
# 
# *(May include "NA" that represents "None")*
# 
# `Alley: NA 	No alley access`
# 
# **T-FEATS: Text representing a combination of features**
# 
# `Utilities: NoSewr	Electricity, Gas, and Water (Septic Tank)`
# 
# **T-SCALE: Text representing a feature that is included in a scale (low / medium / high)**
# 
# `LandSlope: Mod	Moderate Slope`
# 
# **T-FEAT2: Text representing a possible 2nd feature**
# 
# This is in addition to a primary feature of the same categories
# 
# `Condition2: Artery	Adjacent to arterial street`
# 
# **T-YN: Text representing a simple yes/no**
# 
# `CentralAir: Y	Yes`
# 
# **N-CAT: Number representing a category**
# 
# `MSSubClass: 20	1-STORY 1946 & NEWER ALL STYLES`
# 
# **N-QUAN: Number representing a quantity**
# 
# `LotFrontage: 300`
# 
# **N-SCALE: Number representing a quality scale**
# 
# `OverallCond: 6	Above Average`
# 
# **N-DATE: Number representing a date**
# 
# `YearBuilt: 1950`

# ## Data groups
# 
# Some columns can be grouped according to a theme or feature:
# 
# **Type of property**
# 
# `MSSubClass, MSZoning, BldgType, HouseStyle`
# 
# **Area of property**
# 
# `LotFrontage, LotArea`
# 
# **Area of features on property**
# 
# `GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MasVnrArea`
# 
# **Connection to property**
# 
# `Street, Alley, LotConfig, Condition1, Condition2, PavedDrive`
# 
# **Property shape**
# 
# `LotShape, LandContour, LandSlope`
# 
# **Features on property**
# 
# `Utilities, Heating, CentralAir, Electrical, Fireplaces, MiscFeature, MiscVal`
# 
# **Neighborhood**
# 
# `Neighborhood`
# 
# **Quality of house and features**
# 
# `Functional, OverallQual, OverallCond, ExterQual, ExterCond, HeatingQC, FireplaceQu, KitchenQual, GarageQual, GarageCond, PoolQC, Fence, BsmtQual`
# 
# **Significant dates**
# 
# `YearBuilt, YearRemodAdd, GarageYrBlt, YrSold, MoSold`
# 
# **House style and material**
# 
# `RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation`
# 
# **Basement**
# 
# `BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtUnfSF, TotalBsmtSF`
# 
# **Floorspace**
# 
# `1stFlrSF, 2ndFlrSF, LowQualFinSF, BsmtFinSF1, BsmtFinSF2`
# 
# **Baths**
# 
# `BsmtFullBath, BsmtHalfBath, FullBath, HalfBath`
# 
# **Rooms**
# 
# `Bedroom, Kitchen, TotRmsAbvGrd`
# 
# **Garage**
# 
# `GarageType, GarageFinish, GarageCars, GarageCond`
# 
# **Sale**
# 
# `SaleType, SaleCondition`

# ## Data types

# In[ ]:


list(train.dtypes.unique())


# We can see that there is only 3 types of data in the frame.

# ## Data Conversion
# 
# `MSSubClass, OverallQual, OverallCond, MoSold, YrSold, YearBuilt, YearRemodAdd` are all number data types, but they represent categories or dates.
# 
# `MSSubClass, OverallQual, OverallCond` are to be converted to string since they are categories.
# 
# `MoSold` follows in this example, since it makes little sense to keep it as a number.
# 
# `YrSold, YearBuilt, YearRemodAdd` are more complicated. We can extract features from them (like age of house when sold), as well as convert them to string if some years has better house prices than others (like after a recession).

# ## Missing Data
# 
# let's make a graph of all the significant missing data:

# In[ ]:


train, test = import_data('../input/train.csv', '../input/test.csv')
df = pd.concat([train,test], sort=False)

threshold = 2 
missing = showPercentageMissing(df)
largeMissing = missing[missing > threshold]
smallMissing = missing[missing <= threshold]

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f'Percentage missing data (> {threshold}%)', fontsize=28)

plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize

ax1 = fig.add_subplot(111)
largeMissing.plot.barh(ax=ax1)
plt.show()


# We can see that:
# * There are some features with most of their data missing
# * There are some features that seem grouped together with similar amounts missing
# 
# Let's look at the sub 5% data:

# In[ ]:


fig = plt.figure(figsize=(18, 10))
fig.suptitle(f'Percentage missing data (<= {threshold}%)', fontsize=28)

plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize

ax1 = fig.add_subplot(111)
smallMissing.plot.barh(ax=ax1)
plt.show()


# We will take a closer look at `LotFrontage`, all the basement features and the all the garage features.
# 
# For `MiscFeature, PoolQC, Alley, Fence, FireplaceQu`, we know that if there are missing values we may safely assume that it is none as per the data documentation.
# 
# The rest of the missing features only have 1% data missing or less, so we can make larger assumptions with them:
# 
# ```
# Functional: Assume typical
# Electrical: Assume standard 
# KitchenQual: Assume normal
# GarageYrBlt: Assume average 
# 
# Assume most common:
# Exterior1st, Exterior2nd, SaleType, MasVnrType
# ```

# ### Garage
# 
# According to the data description document, 'nan' indicates no garage.

# In[ ]:


df = pd.concat([train,test], sort=False)
list(df['GarageType'].unique())


# There are two cases where a `GarageType` is recorded but no further information:

# In[ ]:


df = pd.concat([train,test], sort=False)
zoneColumns = ['MSSubClass', 'MSZoning', 'BldgType', 'Neighborhood']
garageColumns = ['GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageType']
df[(df['GarageType'].isna() == False) & (df['GarageCond'].isna())][zoneColumns + garageColumns]


# Since there are only two missing rows, we will infer their values from the most common ones.

# ### Basement

# Just like Garage, if all the columns related to this feature is nan then we can infer that there is none on the property.
# 
# Let's see all cases of where only some of the columns are missing values:

# In[ ]:


df = pd.concat([train,test], sort=False)
columns = ['BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1']

df = df[columns]
df = df[df.isna().any(axis=1)]
df[(df['BsmtExposure'].isna() == False) | (df['BsmtFinType2'].isna() == False) | (df['BsmtCond'].isna() == False) | (df['BsmtExposure'].isna() == False) | (df['BsmtFinType1'].isna() == False)]


# This looks like the information can be filled with the most common data for each property's zoning and neighbourhood.

# **Conclusion**
# 
# A combination of 'No basement' and filling with estimated data will be used.

# ### Masonry
# 
# Let's look at what's missing:

# In[ ]:


df = pd.concat([train,test], sort=False)
SMasVnrType = df['MasVnrType']
SMasVnrArea = df['MasVnrArea']
print(f'Only missing SMasVnrArea: {len(df[(SMasVnrType.isna() == True) & (SMasVnrArea.isna() == False)])}')
print(f'Only missing SMasVnrType: {len(df[(SMasVnrType.isna() == False) & (SMasVnrArea.isna() == True)])}')
print(f'Missing both: {len(df[(SMasVnrType.isna() == True) & (SMasVnrArea.isna() == True)])}')


# So 23 cases of both columns being empty and one of only `SMasVnrType` being filled in.
# 
# Let's look at the possible values of `MasVnrType`:

# In[ ]:


df = pd.concat([train,test], sort=False)
df['MasVnrType'] = df['MasVnrType'].fillna('UNDETERMINED')
df['MasVnrType'].value_counts()


# There's already a `None` type present. Can we assume that missing values could be considered as `None` as well?
# 
# Let's see the most common masonry types:

# In[ ]:


df = train.copy()
df['MasVnrType'] = df['MasVnrType'].fillna('UNDETERMINED')
df.groupby(['MasVnrType'])['SalePrice'].count()


# 'None' and 'bricks' are the most common.
# 
# Let's what price groups each falls under:

# In[ ]:


df = train.copy()
df['MasVnrType'] = df['MasVnrType'].fillna('UNDETERMINED')

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

sns.distplot(df[df['MasVnrType'] == 'None']['SalePrice'])
plt.show()


# In[ ]:


sns.distplot(df[df['MasVnrType'] == 'UNDETERMINED']['SalePrice'])
plt.show()


# In[ ]:


sns.distplot(df[df['MasVnrType'] == 'BrkFace']['SalePrice'])
plt.show()


# No real difference between these different groups.

# **Conclusion**
# 
# Since rows with masonry type `NA` don't significantly correspond with a different price class we can assume that they are of 'None' type, and with zero masonry area.

# ### LotFrontage
# 
# There are many lotFrontages missing at 17% of the total. Let's confirm how many other lots have zero frontage:

# In[ ]:


len(train[train['LotFrontage'] == 0])


# No other properties have zero frontage. Let's see under what frontage types the missing values fall under:

# In[ ]:


train[train['LotFrontage'].isna()].groupby('LotConfig')['Id'].count().plot(kind='barh')
plt.show()


# Compared to the non-Nan rows:

# In[ ]:


train[train['LotFrontage'].isna() == False].groupby('LotConfig')['Id'].count().plot(kind='barh')
plt.show()


# We can see that the missing values follow roughly the same distribution, with most falling under `Corner` or `Inside`.
# 
# Let's look at the mean frontage of the properties grouped by frontage type:

# In[ ]:


group = train[train['LotFrontage'].isna() == False].groupby('LotConfig')
df = pd.DataFrame()
df['count'] = group['Id'].count()
df['mean'] = group['LotFrontage'].mean()

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

plt.style.use('default')

df['count'].plot(kind='barh', ax=ax1, title='COUNT')
df['mean'].plot(kind='barh', ax=ax2, title='MEAN')

plt.tight_layout()
plt.show()


# `Corner` types have the most frontage and `CulDeSac` types have the least, which makes sense.
# 
# Let's write a script to approximate `LotFrontage` according to the mean `LotFrontage` in its area and type.

# In[ ]:


df = train.copy()
df = df[df['LotFrontage'].isna() == False]

df['test'] = df.groupby(['MSSubClass', 'MSZoning', 'Neighborhood', 'LotConfig'])['LotFrontage'].transform(lambda x: x.median())
df['diff'] = abs(df['test'] - df['LotFrontage'])
print(f'Mean: {df["diff"].mean()}')
      
df = train.copy()
df = df[df['LotFrontage'].isna() == False]
print(f'Std Dev: {df["LotFrontage"].std()}')


# It only differs by 7.7 feet on average, one third of the standard deviation.
# 
# **Conclusion**
# 
# Lot frontage can be estimated by the mean of the neighbourhood.

# ## Outliers
# 
# We can plot features against their respective saleprice to identify rows that are abnormal.
# 
# Let's do this for the `LotArea` feature.
# 
# We will run a script that determines the zscore of each value in a column. We will then determine which ones are outlier, by marking the ones that exceed a threshold. In this case the threshold is 3, which is generally accepted as the threshold for outliers.

# In[ ]:


def getClusters(df, threshold):

    df = train.copy()
    df = train.fillna(0)

    # Filter by numeric columns
    numericColumns = getColumnsNamesByType(df, ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    df = df[numericColumns]

    # Remove numeric that do not apply here
    df.drop(['MSSubClass', 'Id'], axis=1, inplace=True)

    # Remove columns that don't have a good spread of values
    df = df.iloc[:, getColRange(df) > 20]

    # get an array indicating outlier columns
    return getOutliers(df, threshold).any(axis='columns').apply(lambda x: 0 if x == False else 1)

df = train.copy()
df = train.fillna(0)
clusters = getClusters(df, 3)
print(f'Amount of outliers: {clusters.sum()}')


# 103 is a lot for the size of the data set. Let's graph `LotArea`:

# In[ ]:


def plotOutliers(df, colname, clusters):

    scaler = MinMaxScaler()
    dfPlot = df[['SalePrice', colname]].astype(float)

    # Scale the data
    compareColumns = scaler.fit_transform(dfPlot)
    compareColumns = pd.DataFrame(compareColumns, columns = ['SalePrice', colname])

    cmap = cm.get_cmap('Accent')
    
    compareColumns.plot.scatter(
        x = colname,
        y = 'SalePrice',
        c = clusters,
        cmap = cmap,
        colorbar = False)
    plt.show()

plotOutliers(df, 'LotArea', clusters)


# Here we can see that rows outside of the general trend are removed, but so are any within it. Let's try again with a higher threshold:

# In[ ]:


clusters = getClusters(df, 6)
print(f'Amount of outliers: {clusters.sum()}')
plotOutliers(df, 'LotArea', clusters)


# At a threshold of 6, it doesn't filter out too many rows. We can see that all of the exceptional outliers are detected, but also with a couple in the general trend still being included.
# 
# The best method will be to run the entire model's fitting & prediction process and then adjust the sensitivity to find the best balance.
# 
# **Conclusion**
# 
# Filtering out outliers with by their zscore is an effective method. A threshold level of 3 is too low, and an appropriate level must be found by experimentation with the training data.

# ## Scaling
# 
# We will use sklearn's RobustScaler. Let's look at an example.
# 
# let's scale `SalePrice` and `LotArea`:

# In[ ]:


from sklearn.preprocessing import RobustScaler
train, test = import_data('../input/train.csv', '../input/test.csv')

test = train[['SalePrice', 'LotArea']]

salePrice = test['SalePrice'].copy()

plt.style.use('default')
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.scatter(x = test['LotArea'], y = test['SalePrice'])


transformer = RobustScaler().fit(test)
RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True)
test = transformer.transform(test)
ax2.scatter(x = test[:,1], y = test[:,0])

ax1.set_xlabel('LotArea')
ax1.set_ylabel('SalePrice')
ax2.set_xlabel('LotArea')
ax2.set_ylabel('SalePrice')

ax1.set_title('Before')
ax2.set_title('After')

sns.set_style("ticks", {"xtick.major.size":1, "ytick.major.size":1})

sns.distplot(salePrice, ax=ax3)
sns.distplot(test[:,0], ax=ax4)


plt.tight_layout()
plt.show()


# We can see that the shape of the data is preserved, only the scale of the values are changed.

# ## Correct Skewness
# 
# Ideally we'd like our data to follow a normal distribution.
# 
# Let's visualise the distribution of SalePrice values in the training data against a normal distribution for reference.
# 
# We will then transform it using a simple log function and then a more advanced statistics function called Box Cox:

# In[ ]:


def plotSkewness(df, colname):

    fig = plt.figure(figsize=(12, 2))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Get the log of saleprice
    # We use np.log1p here to avoid problems with zero values 
    logdf = np.log1p(df[colname])

    # Get the Box Cox transformation
    boxcox = boxcox1p(df[colname], boxcox_normmax(df[colname] + 1))
    
    # plot it
    sns.distplot(df[colname], fit=stats.norm, ax=ax1)
    sns.distplot(logdf, kde=False, fit=stats.norm, ax=ax2)
    sns.distplot(boxcox, kde=False, fit=stats.norm, ax=ax3)

    # Get the skew indexes
    originalSkew = round(skew(df[colname]),4)
    logSkew = round(skew(logdf),4)
    boxCoxSkew = round(skew(boxcox),4)

    # Titles
    ax1.set_title(f'Before, skew: {originalSkew}')
    ax2.set_title(f'Log, skew: {logSkew}')
    ax3.set_title(f'BoxCox, skew: {boxCoxSkew}')
    
    fig.suptitle(colname, va='bottom', size='16')
    plt.show()

plotSkewness(df, 'SalePrice')


# We compute the skew index of the original data to determine by what factor and direction the data leans away from the mean. A positive value shows that it leans right.
# 
# The dark line shows the best possible fit of an exact normal distribution.
# 
# By simply taking the log of the data, we get close to a uniform normal distribution, but the conditional Box Cox performs even better.
# 
# Of course these transformations are only applicable on data that are large distributions of numeric data, and would not apply to categorical features.
# 
# These transformation do not always work, for example:

# In[ ]:


df = train.copy()
numericColumns = getColumnsNamesByType(df, ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
df = df[numericColumns]
df.drop(['MSSubClass', 'Id'], axis=1, inplace=True)
df = df.iloc[:, getColRange(df) > 20]
df = df.fillna(0)

plotSkewness(df, 'LotFrontage')


# Here we see that the orginal data had the lowest skew index.

# In[ ]:


plotSkewness(df, 'LotArea')


# Here we see that the simple log produces a distribution with the lowest skew index instead of the Box Cox method. We will have to use discretion in which method we employ, if at all.

# # Data Preparation
# 
# Here we define functions we will use later:

# In[ ]:


def processData(train, test, outlierThreshold, outliers=[]):
    train = train.copy()
    test = test.copy()

    # Save the original sale price values for comparison later
    y = train['SalePrice']

    # Get the log of the sale price info and store it for model fitting later
    yLog = np.log1p(y)

    # Drop columns that are irrelevant
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)
    train.drop(['SalePrice'], axis=1, inplace=True)

    # Combine the train and test sets so that we can transform both simultaneously
    features = pd.concat([train, test]).reset_index(drop=True)    

    # Fill in missing values
    # ============

    # Here we assume the standard / typical case
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")

    # Here we assume the most common assignment across the board
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

    # Missing values indicate "none"
    features["MiscFeature"].fillna("None", inplace=True)
    features["PoolQC"].fillna("None", inplace=True)
    features["Alley"].fillna("None", inplace=True)
    features["Fence"].fillna("None", inplace=True)
    features["FireplaceQu"].fillna("None", inplace=True)
    features['GarageType'].fillna("None", inplace=True)
    features['GarageFinish'].fillna("None", inplace=True)
    features['GarageQual'].fillna("None", inplace=True)
    features['BsmtExposure'].fillna("None", inplace=True)
    features['BsmtFinType1'].fillna("None", inplace=True)
    features['BsmtFinType2'].fillna("None", inplace=True)

    # Assume zero
    features["GarageArea"] = features["GarageArea"].fillna(0)
    features["GarageCars"] = features["GarageCars"].fillna(0)
    features["MasVnrArea"] = features["MasVnrArea"].fillna(0)
    features["GarageYrBlt"] = features["GarageYrBlt"].fillna(0)
    features["BsmtFinSF1"] = features["BsmtFinSF1"].fillna(0)
    features["BsmtUnfSF"] = features["BsmtUnfSF"].fillna(0)
    features["TotalBsmtSF"] = features["TotalBsmtSF"].fillna(0)
    features["BsmtFinSF2"] = features["BsmtFinSF2"].fillna(0)

    # Ensure that there is an indicator for where no feature is present, if it has a corresponding size/amount value
    features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # For missing values assume median according to type and area
    features['LotFrontage'] = features.groupby(['MSSubClass', 'MSZoning', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.median())
    
    # For whatever remains, fill most common by Neighborhood
    features['LotFrontage'] = features.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.median())
    
    
    # Convert numerical categorical data to string
    smallNumerics = list(getSmallCols(features, 10))
    convertList = smallNumerics + ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YearBuilt', 'YearRemodAdd']
    features = convertColumnTypes(features, convertList, str)

    numericColumns = getColumnsNamesByType(features, ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    
    # perform log transformation depending on best method
    for colname in numericColumns:
        colLog = np.log1p(features[colname])
        colBoxCox = boxcox1p(features[colname], boxcox_normmax(features[colname] + 1))

        skewOriginal = abs(skew(features[colname]))
        skewLog = abs(skew(colLog))
        skewBoxCox = abs(skew(colBoxCox))

        columns = [features[colname], colLog, colBoxCox]
        skewValues = [skewOriginal, skewLog, skewBoxCox]

        features[colname] = columns[skewValues.index(min(skewValues))]
        
    # Remove outliers
    quantitative = features[numericColumns]
    dfOutliers = getOutliersModel(quantitative.iloc[:len(y), :], outlierThreshold)

    yLen = len(y)
    y = filterOutliers(y, dfOutliers)
    yLog = filterOutliers(yLog, dfOutliers)
    features = pd.concat([filterOutliers(features.iloc[:yLen, :], dfOutliers), features.iloc[yLen:, :]])

    # Get dummy data
    features = pd.get_dummies(features).reset_index(drop=True)

    # Split back to train and test
    X = features.iloc[:len(y), :]
    X_test = features.iloc[len(y):, :]

    return (X, yLog, y, X_test)


# ## Models
# 
# Build, fitting and prediction functions

# In[ ]:


# Fetches the models we can use
def get_models():
    
    models = {}
    
    # Init models with default values
    # We use make_pipeline here to automate RobustScaler
    ridge = make_pipeline(RobustScaler(), RidgeCV())
    lasso = make_pipeline(RobustScaler(), LassoCV())
    elasticnet = make_pipeline(RobustScaler(),ElasticNetCV())
    svr = make_pipeline(RobustScaler(), SVR())
    gbr = GradientBoostingRegressor()
    lightgbm = LGBMRegressor()
    xgboost = XGBRegressor()

    randomForest = make_pipeline(RobustScaler(), RandomForestRegressor())
    decisionTree = make_pipeline(RobustScaler(), DecisionTreeRegressor())

    models['ridge'] = ridge  
    models['lasso'] = lasso  
    models['elasticnet'] = elasticnet  
    models['svr'] = svr  
    models['gbr'] = gbr  
    models['lightgbm'] = lightgbm  
    models['xgboost'] = xgboost  
    models['randomForest'] = randomForest
    models['decisionTree'] = decisionTree
    
    return models

# Fits models to data
def fitModels(models, X, y, models_to_fit = ['ridge', 'lasso', 'elasticnet', 'svr', 'gbr', 'lightgbm', 'xgboost', 'stack_gen', 'randomForest', 'decisionTree'], verbose=True):
    if verbose:
        print('Start fitting')
    startTime = datetime.now()
    latestTime = datetime.now()
    
    fittedModels = {}
    
    for modelName in models_to_fit:
        if verbose:
            print(f'\t Fitting {modelName}')
        fittedModels[modelName] = models[modelName].fit(X, y)
        
        if verbose:
            print(f'\t Fit time: {(datetime.now() - latestTime).seconds}s')
        
        latestTime = datetime.now()   

    return fittedModels

# Predicts y using fitted models and weights
def predict(models, X, weights=False, exp=False):
    predictions = None
    if weights == False:
        weights = numpy.ones(len(models))
    
    for modelName in models:
        if (predictions is None):
            predictions = models[modelName].predict(X) * weights[modelName]
        else:
            predictions = np.vstack([predictions, models[modelName].predict(X) * weights[modelName]])
        
    shape = predictions.shape
    if len(shape) > 1:
        predictions = predictions.sum(axis=0)
    
    if exp: return np.floor(np.expm1(predictions))
    return predictions


# # Machine Learning

# In[ ]:


# Main prediction function
def machineLearning(params):
    
    train = params['train']
    test = params['test']
    modelNames = params['modelNames']
    filenames = params['filenames']
    comment = params['comment']
    outlierThreshold = params['outlierThreshold']
    
    # Handles model fitting and shows a RMLE score of each
    def fitAndScore(X, y, modelNames, comment):
        models = get_models()
        fitted_models = fitModels(models, X, y, modelNames)

        score, weights = cv_scoring(models, X, y, modelNames)
        record_score('scorecard.csv', score, comment, modelNames)
        print(f'Weights: {weights}')
        
        return (fitted_models, weights)
    
    # Plots a scatter graph of predicted y against real y
    def showPredicted(fitted_models, X, y_original, weights, X_label, Y_label):
        train_y_pred = predict(fitted_models, X, weights, exp=True)
        pltScatter(y_original, train_y_pred, X_label, Y_label)
        return train_y_pred
    
    # DATA PREPARATION
    # ===================================
    X, y, y_original, X_sub = processData(train, test, outlierThreshold)
    
    # MODEL FITTING AND SCORING
    # ===================================
    fitted_models, weights = fitAndScore(X, y, modelNames, comment)
    
    # TRAIN Y PREDICTION ANALYSIS
    # ===================================
    X_label='Real Training SalePrice'
    Y_label='Predicted Training SalePrice'
    train_y_pred = showPredicted(fitted_models, X, y_original, weights, X_label, Y_label)
    outliers, zscore = getOutlierIndexes(train_y_pred, y_original, outlierThreshold)
    print(f'outliers: {outliers}')
    print(f'zscore mean: {round(zscore.mean(),4)} \t std dev: {round(zscore.std(),4)}')
    
    # TEST Y PREDICTION
    # ===================================
    test_y_pred = predict(fitted_models, X_sub, weights, exp=True)

    # SAVE RESULTS
    # ===================================
    submission = pd.read_csv(filenames['sample_submission_filename'])
    submission.iloc[:,1] = test_y_pred
    submission.to_csv(filenames['submission_filename'], index=False)


# # Main Function
# 
# This is where we start with our model fitting and prediction process.
# 
# Let's set up some environmental variables and import our data.

# In[ ]:


comment = 'Running our model'
filenames = {}
filenames['submission_filename'] = 'submission.csv'
filenames['sample_submission_filename'] = '../input/sample_submission.csv'

train, test = import_data('../input/train.csv', '../input/test.csv')

outlierThreshold = 3
modelNames = ['ridge', 'lasso', 'svr']


# Let's prepare the data:

# In[ ]:


X, y, y_original, X_sub = processData(train, test, outlierThreshold)
print(f' X: {X.shape}')
print(f' y: {y.shape}')
print(f' y_original: {y_original.shape}')
print(f' X_sub: {X_sub.shape}')


# Looks good
# 
# Let's fit the models.
# 
# The `fitModels()` function will take the train `X` data and fit the models we selected in `modelNames` and fit it to the real `y` values.
# 

# In[ ]:


models = get_models()
fitted_models = fitModels(models, X, y, modelNames)


# `cv_scoring()` takes each fitted model and runs a cross validation routine on the training data, similar to the train/test split method we would normally employ. It uses a portion of the train data to predict the other portion's y, repeats this process a number of times, and returns the Root mean square result of each attempt.
# 
# We then take the log of the mean of each attempt, and print it to screen.

# In[ ]:


score, weights = cv_scoring(models, X, y, modelNames)
record_score('scorecard.csv', score, comment, modelNames)


# We keep a record of previous attempts' scores for reference.
# 
# We also get an estimation of the weights to use for each model's prediction based on its accuracy:

# In[ ]:


print(f'Weights: {weights}')


# It's completely unscientific and just based on personal judgement, but is weighted toward the models that score better.
# 
# Now let's look at our predicted y for the training data against the real y of the training data:

# In[ ]:


# Plots a scatter graph of predicted y against real y
def showPredicted(fitted_models, X, y_original, weights, X_label, Y_label):
    train_y_pred = predict(fitted_models, X, weights, exp=True)
    pltScatter(y_original, train_y_pred, X_label, Y_label)
    return train_y_pred

X_label='Real Training SalePrice'
Y_label='Predicted Training SalePrice'
train_y_pred = showPredicted(fitted_models, X, y_original, weights, X_label, Y_label)


# The relationship between each axis is clear, and follows a linear trend, which is exactly what we want.
# 
# There are no extreme outliers, so that part of the data preparation is working.
# 
# We can get the zscore of these outliers:

# In[ ]:


outliers, zscore = getOutlierIndexes(train_y_pred, y_original, outlierThreshold)
print(f'Outlier threshold: {outlierThreshold}')
print(f'Number of outliers: {len(outliers)}')
print(f'zscore mean: {round(zscore.mean(),4)}')
print(f'std dev: {round(zscore.std(),4)}')


# At an outlier threshold of 3, 18 y predictions are deemed irregular. Make of that what you will.
# 
# Time to do the actual prediction:

# In[ ]:


test_y_pred = predict(fitted_models, X_sub, weights, exp=True)

# SAVE RESULTS
# ===================================
submission = pd.read_csv(filenames['sample_submission_filename'])
submission.iloc[:,1] = test_y_pred
submission.to_csv(filenames['submission_filename'], index=False)


# At time of writing this method gets a score of **0.13090**, which ranks **208th** on the rolling leaderboard of 4000+ submissions.
