#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle. Let's have a look...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/train.csv')
df.head()


# Let's split into numeric and non-numeric columns and let's start with the numeric...

# In[ ]:


enumSeries = []
numSeries = []
for colname in df.columns:
    datatype = str(df[colname].dtype)
    if colname == 'Id' or colname == 'SalePrice':
        pass
    elif datatype == 'int64' or datatype == 'float64':
        numSeries.append(colname)
    else:
        enumSeries.append(colname)
df[numSeries].head()


# Let's try for a histogram of sale prices. There is a rumour these might work best on a logarithmic scale. So let's try plotting straight at the moment...

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(df['SalePrice'], bins=50)


# And now on a logarithmic scale...

# In[ ]:


plt.hist(np.log10(df['SalePrice']), bins=50)


# Let's use the Pearson correlation co-efficient to get a sense of what our best predicting features are. *LotArea* is suprisingly useless - perhaps you can get comparitively large lots with a pretty small house in a terrible location. *OverallQual* has one of the best correlation co-efficients as do a bunch of living area indicators: *GrLivArea*, *TotalBsmtSF*, *1stFlrSF* and *GarageArea*.
# 
# Ultimately I want our first guess to be based on *OverallQual*, *GrLivArea*, *GarageCars* and *1stFlrSF* because those seem like good predictive features that are relatively independent of one another. We can pull in more and build a better model as time progresses.

# In[ ]:


numSeriesForVq = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF']
df[numSeries].corrwith(np.log10(df['SalePrice'])).sort_values(ascending=False)


# There's a rumour that areas should also be ${log}_{10}$. Let's try that...

# In[ ]:


pd.concat([np.log10(df['GrLivArea']), np.log10(df['1stFlrSF'])], axis=1).corrwith(np.log10(df['SalePrice']))


# Looks correct let's do that...

# In[ ]:


ndf = pd.concat([df, pd.Series(np.log10(df['GrLivArea']),name='GrLivArea10'), pd.Series(np.log10(df['1stFlrSF']),name='1stFlrSF10'), pd.Series(np.log10(df['SalePrice']),name='SalePrice10')], axis=1,)
numSeriesForVq = ['OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10']
ndf[numSeriesForVq + ['SalePrice10'] + ['SalePrice']].head()


# Let's try a multi-variate linear regression...

# In[ ]:


from sklearn.linear_model import LinearRegression
x_train = ndf[numSeriesForVq].values
y_train = ndf['SalePrice10'].values
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
colors = ['blue', 'red', 'green', 'orange']
choice = 0
for xval in numSeriesForVq:
    plt.scatter(ndf[xval], y_train, color=colors[choice])
    plt.xlabel(numSeriesForVq[choice])
    plt.ylabel('SalePrice10')
    plt.show()
    choice += 1


# Let's test our linear regression...

# In[ ]:


from matplotlib.ticker import FormatStrFormatter

choices = np.sort(np.random.choice(x_train.shape[0], 10))
predictions = np.power(10, lin_reg.predict(x_train))
targets = np.power(10, y_train)
preds = predictions[choices] / 1e3
acts = targets[choices] / 1e3
ind = np.arange(len(preds))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width / 2, preds, width, label='Prediction')
rects2 = ax.bar(ind + width / 2, acts, width, label='Actual')
ax.set_ylabel('Price')
ax.set_xticks(ind)
ax.set_xticklabels(tuple(choices))
ax.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0fk'))
fig.tight_layout()
plt.show()


# Let's get some information on how our model is performing. $22.8k ain't bad.

# In[ ]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(predictions, targets)
mae


# And did we do better than just using *OverallQual* on its own? Yes we did!

# In[ ]:


print(np.corrcoef(ndf['OverallQual'].values, np.log10(targets))[0][1])
print(np.corrcoef(np.log10(predictions), np.log10(targets))[0][1])


# This tries to work out if there are any easy wins from the remaining numeric data. The answer is no (you can review the tail too but it's no more exiciting).

# In[ ]:


err = targets - predictions
df[numSeries].corrwith(pd.Series(err)).sort_values(ascending=False).head()


# Let's now turn the enumerated types. First let's get all enumerated types of significant size (i.e. they exist for more than between $1/10$th and $9/10$th of the training set).

# In[ ]:


allEnums = []
for enumSeriesName in enumSeries:
    valueCounts = ndf[enumSeriesName].value_counts()
    for key in valueCounts.index:
        if valueCounts[key] > df.shape[0] / 10 and valueCounts[key] < 9 * df.shape[0] / 10:
            allEnums.append({'series': enumSeriesName, 'key': key})
allEnums


# Now let's take those and restrict our dataset by it. For clarity, I'm going to do one and then in the next step I'm going to do an operation...

# In[ ]:


enumDetails = allEnums[0]
print(enumDetails['series'] + ' == "' + enumDetails['key'] + '"')
rdf = ndf.query(enumDetails['series'] + ' == "' + enumDetails['key'] + '"')
rdf.head()


# The opeartion is simply to compare the mean of a certain enumerated value with that of the entire training set. If the mean is significantly different that would indicate the enumerated value is a useful discriminant. My maths teacher used to say, "engineers get away with things, mathematicians never dream of." Since the logarithmic sale prices are normal-ish, we'll divide by the standard deviation of the training set. If we got say a three we'd know we have a really good discriminant. We haven't and it is rare to find such a discriminant in real-world data but some of the discriminants are still useful so let's take the first three and build $2^3=8$ different linear models from them.

# In[ ]:


allMean = ndf['SalePrice10'].mean()
allStd = ndf['SalePrice10'].std()
enumInds = []
enumScores = []
enumCount = []
for enumDetails in allEnums:
    rdf = ndf.query(enumDetails["series"] + ' == "' + enumDetails["key"] + '"')
    enumInds.append(enumDetails["series"] + '-' + enumDetails["key"])
    enumScores.append((rdf['SalePrice10'].mean() - allMean) / allStd)
    enumCount.append(rdf.shape[0])
series = pd.Series(enumScores, index = enumInds)
countSeries = pd.Series(enumCount, index=enumInds)
res = pd.DataFrame([series, series.abs(), countSeries]).transpose().rename(columns={0: 'NormalizedDiff', 1: 'Absolute', 2: 'Count'}).sort_values(by=['Absolute'], ascending=False).head()
res


# Let's partition the dataset first based on these three values. We should end up with eight subsets.

# In[ ]:


def buildQuery(vals):
    query = ''
    for ind in range(len(vals)):
        if ind > 0:
            query += ' and '
        if vals[ind]:
            query += res.iloc[ind].name.replace('-', ' == "') + '"'
        else:
            query += res.iloc[ind].name.replace('-', ' != "') + '"'
    return  query

import itertools
lst = list(itertools.product([0, 1], repeat=3))
divisions = []
for val in lst:
    query = buildQuery(val)
    divisions.append({ "query": query, "df": ndf.query(query) })
    print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))


# Okay but now we have a problem the RM zoning despite being our best discriminant is actually pretty rare and when we partition the training set further it the set becomes virtually useless. So maybe we change our queries a little and combine the last four...

# In[ ]:


import itertools
lst = list(itertools.product([0, 1], repeat=3))
divisions = []
for val in lst[0:4]:
    query = buildQuery(val)
    divisions.append({ "query": query, "df": ndf.query(query) })
    print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))
query = 'MSZoning == "RM"'
divisions.append({ "query": query, "df": ndf.query(query) })
print(divisions[-1]['query'] + ': ' + str(divisions[-1]['df'].shape[0]))


# That looks good. Now let's do a linear regression for each query.

# In[ ]:


for division in divisions:
    x_train = division['df'][numSeriesForVq].values
    y_train = division['df']['SalePrice10'].values
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    division['regression'] = lin_reg
    division['x_train'] = x_train
    division['y_train'] = y_train


# So let's run the training set with our new model and see how we do...

# In[ ]:


from matplotlib.ticker import FormatStrFormatter

predictions = np.array([])
targets = np.array([])
for division in divisions:
    predictions = np.concatenate((predictions, np.power(10, division['regression'].predict(division['x_train']))))
    targets = np.concatenate((targets, np.power(10, division['y_train'])))
choices = np.sort(np.random.choice(len(predictions), 10))
preds = predictions[choices] / 1e3
acts = targets[choices] / 1e3
ind = np.arange(len(preds))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width / 2, preds, width, label='Prediction')
rects2 = ax.bar(ind + width / 2, acts, width, label='Actual')
ax.set_ylabel('Price')
ax.set_xticks(ind)
ax.set_xticklabels(tuple(choices))
ax.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0fk'))
fig.tight_layout()
plt.show()


# Is this any better? It may or may not look it...

# In[ ]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(predictions, targets)
mae


# Just a touch the mean absolute error now reduces to \$21.1k down from \$22.8k. Does it crack 0.9 for correlation?

# In[ ]:


print(np.corrcoef(np.log10(predictions), np.log10(targets))[0][1])


# This is all done let's prepare the final submission. Note the new `fillna` to fill in the empty set...

# In[ ]:


tdf = pd.read_csv('../input/test.csv')
ntdf = pd.concat([tdf, pd.Series(np.log10(tdf['GrLivArea']),name='GrLivArea10'),pd.Series(np.log10(tdf['1stFlrSF']),name='1stFlrSF10')], axis=1,)
ntdf = ntdf.fillna(pd.Series([5, 3, 3, 0], index=['OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10']))
for division in divisions:
    divisionTdf = ntdf.query(division['query'])
    division['tdf'] = divisionTdf
    division['x_test'] = divisionTdf[numSeriesForVq].values
    division['x_ids'] = divisionTdf['Id'].values
    division['predictions'] = np.power(10, division['regression'].predict(division['x_test']))
xf = pd.DataFrame(ntdf.query(divisions[4]['query'])[numSeriesForVq+['Id']])

predictions = np.array([])
ids = np.array([])
for division in divisions:
    predictions = np.concatenate((predictions, division['predictions']))
    ids = np.concatenate((ids, division['x_ids']))
res = pd.DataFrame([ids, predictions]).transpose().rename(columns={0: 'Id', 1: 'SalePrice'}).sort_values(by=['Id'])
res.Id = res.Id.astype(int)
res.to_csv('submission.csv', index=False)
res

