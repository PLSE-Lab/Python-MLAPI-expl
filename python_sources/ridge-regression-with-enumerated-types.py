#!/usr/bin/env python
# coding: utf-8

# To this point I've simply used the enumerated types to pick one of five different linear models. I want to see what happens when we use the enumerated types as an input to the linear models. I'm going to start what I've done prior but not going to explain it (check my earlier kernels if you're interested)...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import data
df = pd.read_csv('../input/train.csv')

# Split into numeric and enumerated types
enumSeries = []
numSeries = []
nonDiscreteSeries = []
for colname in df.columns:
    datatype = str(df[colname].dtype)
    if colname == 'Id' or colname == 'SalePrice':
        pass
    elif datatype == 'int64' or datatype == 'float64':
        numSeries.append(colname)
    else:
        enumSeries.append(colname)
        
# Fill in some missing values
df.fillna(pd.Series([0,1970,0, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], index=['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10', 'TotalBsmtSF', 'GarageArea', 'BsmtFullBath', 'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFinSF2', 'BsmtFinSF1']), inplace=True)
assert df[pd.isnull(df[numSeries]).any(axis=1)][numSeries].shape[0] == 0

# Take logarithms of some values
df['LotAreaLog10'] = np.log10(df['LotArea'].values)
df['1stFlrSFLog10'] = np.log10(df['1stFlrSF'].values)
df['GrLivAreaLog10'] = np.log10(df['GrLivArea'].values)
df['SalePriceLog10'] = np.log10(df['SalePrice'].values)
if 'LotArea' in numSeries:
    numSeries.remove('LotArea')
    numSeries.append('LotAreaLog10')
if '1stFlrSF' in numSeries:
    numSeries.remove('1stFlrSF')
    numSeries.append('1stFlrSFLog10')
if 'GrLivArea' in numSeries:
    numSeries.remove('GrLivArea')
    numSeries.append('GrLivAreaLog10')

df[numSeries].head()


# The following code is almost as long as the previous but essentially takes all enumerated values within each series and sorts them by the mean sale price. Importantly null values are treated as just another enumerated type. The sorted enumerated values are then assigned nominal numeric values but so these values ascend with the sorting. With this more thoughtful assignment of numeric values it is hoped the ridge regression will perform better. The output is:
# 
# ```
# SERIES-ENUM: NUMERIC (COUNT)
# ```
# 
# where `SERIES` is the series name, `ENUM` is the enumerated value, `NUMERIC` the nominal numeric value and `COUNT` the count of that enumerated value in the training set
# 
# For cases where there are no null values, they are assigned the mean numeric value. You can easily spot these because they are unsorted and have a `COUNT` of zero. 

# In[ ]:


import operator

allEnums = []
for seriesName in enumSeries:
    allEnums.append([])
    enums = df[seriesName].value_counts()
    for enum in enums.index.values:
        allEnums[-1].append({ 'name': enum, 'series': seriesName, 'mean': df.query(seriesName + ' == "' + enum + '"')['SalePrice'].mean(), 'count': df.query(seriesName + ' == "' + enum + '"')['SalePrice'].count() })
    if pd.isnull(df[seriesName]).any():
        val = 0.0
        allEnums[-1].append({ 'name': None, 'series': seriesName, 'mean': df.query(seriesName + ' != ' + seriesName)['SalePrice'].mean(), 'count': df.query(seriesName + ' == "' + enum + '"')['SalePrice'].count() })
    else:
        val = 1.0                    
    allEnums[-1] = sorted(allEnums[-1], key=lambda k: k['mean'])
    meanSumForNull = 0.0
    meanCntForNull = 0
    for enumDict in allEnums[-1]:
        enumDict['value'] = val
        meanSumForNull += val * enumDict['count']
        meanCntForNull += enumDict['count']
        val += 1.0
    if not pd.isnull(df[seriesName]).any():
        allEnums[-1].append({ 'name': None, 'series': seriesName, 'mean': 0, 'value': meanSumForNull / float(meanCntForNull), 'count': 0 })    
for enumCollection in allEnums:
    for enumDict in enumCollection:
        if enumDict['name']:
            print(enumDict['series'] + '-' + enumDict['name'] + ': ' + str(enumDict['value']) + ' (' + str(enumDict['count']) + ')')
        else:
            print(enumDict['series'] + '-NULL: ' + str(enumDict['value']) + ' (' + str(enumDict['count']) + ')')
                            


# From here it's a pretty simple to put our numeric values into action...

# In[ ]:


for enumCollection in allEnums:
    srcCol = enumCollection[0]['series']
    destCol = enumCollection[0]['series'] + 'Numeric'
    mapping = {}
    naValue = 0.0
    for enumDict in enumCollection:
        if enumDict['name']:
            mapping[enumDict['name']] = enumDict['value']
        else:
            naValue = enumDict['value']
    df[destCol] = df[srcCol].map(mapping)
    df[destCol].fillna(naValue, inplace=True)
    if not destCol in numSeries:
        numSeries.append(destCol)
df[numSeries].head()


# Let's find our $\alpha$ for ridge regression...

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = { 'alpha': [ 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20 ]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(df[numSeries].values, df['SalePriceLog10'])
print(ridge_regressor.best_params_)


# With $\alpha = 1$, let's the get he mean absolute erorr:

# In[ ]:


from sklearn.metrics import mean_absolute_error
ridge = Ridge(alpha=1)
ridge.fit(df[numSeries].values, df['SalePriceLog10'])
ridgePredictions = np.power(10, ridge.predict(df[numSeries].values))
targets = np.power(10, df['SalePriceLog10'])
mae = mean_absolute_error(ridgePredictions, targets)
mae


# This suggests we might have a new winning model. Let's prepare our test data in the same manner...

# In[ ]:


tdf = pd.read_csv('../input/test.csv')
tdf.fillna(pd.Series([0,1970,0, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], index=['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10', 'TotalBsmtSF', 'GarageArea', 'BsmtFullBath', 'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFinSF2', 'BsmtFinSF1']), inplace=True)
tdf['LotAreaLog10'] = np.log10(tdf['LotArea'].values)
tdf['1stFlrSFLog10'] = np.log10(tdf['1stFlrSF'].values)
tdf['GrLivAreaLog10'] = np.log10(tdf['GrLivArea'].values)
for enumCollection in allEnums:
    srcCol = enumCollection[0]['series']
    destCol = enumCollection[0]['series'] + 'Numeric'
    mapping = {}
    naValue = 0.0
    for enumDict in enumCollection:
        if enumDict['name']:
            mapping[enumDict['name']] = enumDict['value']
        else:
            naValue = enumDict['value']
    tdf[destCol] = tdf[srcCol].map(mapping)
    tdf[destCol].fillna(naValue, inplace=True)
    if not destCol in numSeries:
        numSeries.append(destCol)
assert tdf[pd.isnull(tdf[numSeries]).any(axis=1)][numSeries].shape[0] == 0
tdf[numSeries].head()


# And now run our model...

# In[ ]:


ridgeFinalPredictions = np.power(10, ridge.predict(tdf[numSeries].values))
assert ridgeFinalPredictions.shape[0] == tdf['Id'].shape[0]
res = pd.DataFrame([tdf['Id'], ridgeFinalPredictions]).transpose().rename(columns={'Unnamed 0': 'SalePrice'})
res.Id = res.Id.astype(int)
res.to_csv('submission.csv', index=False)
res

