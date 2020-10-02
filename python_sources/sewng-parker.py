#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# import modules 

import os
import calendar
import numpy as np 
import networkx as nx
import pandas as pd
from pandas.plotting import scatter_matrix, parallel_coordinates
import seaborn as sns
from sklearn import preprocessing 
import matplotlib.pylab as plt


# In[ ]:


# Purpose of this notebook is finding some key factors that decide housing price


# # Firtstly, bring data from csv file
# 

# In[ ]:


houseprice_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
houseprice_df.shape
print(houseprice_df)


# In[ ]:


test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.head()


# In[ ]:


# Descriptive statistics

print('Number of rows', len(houseprice_df['MSSubClass']))
print('Mean of MSSubClass', houseprice_df['MSSubClass'].mean())
houseprice_df.describe()


# In[ ]:


# Sampling
houseprice_df.sample(10)

# Oversampling
weights = [0.9 if years>2008 else 0.01 for years in houseprice_df.YearBuilt]
houseprice_df.sample(10, weights=weights)


# In[ ]:


# Search for all of columns
houseprice_df.columns


# **Numeric Data Analysis**

# In[ ]:


# heatmap of correlations 
corr = houseprice_df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap="RdBu")

## Include information about values (example demonstrates how to control the size of the plot)
fig, ax = plt.subplots()
fig.set_size_inches(20, 15)
sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu", center=0, ax=ax)


# Now, we know which factors have high correlation with SalePrice

# In[ ]:


# Histogram of SalePrice
ax = houseprice_df.SalePrice.hist()
ax.set_xlabel('SalePrice'); ax.set_ylabel('count')


# In[ ]:


# Analyze high-correlation factors in detail

saleprice_ts = pd.Series(houseprice_df.SalePrice.values, index=houseprice_df.YearBuilt)
houseprice_df.plot.scatter(x='OverallQual',y='SalePrice', legend=False)
houseprice_df.plot.scatter(x='YearBuilt',y='SalePrice', legend=False)
houseprice_df.plot.scatter(x='GrLivArea',y='SalePrice', legend=False)
houseprice_df.plot.scatter(x='GarageCars',y='SalePrice', legend=False)
houseprice_df.plot.scatter(x='OverallCond',y='SalePrice', legend=False)


# In[ ]:


#box plot overallqual/saleprice
## Reference_notebook by  Pedro Marcello, 2017

var = 'OverallQual'
data = pd.concat([houseprice_df['SalePrice'], houseprice_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


# In[ ]:


# Covariance Matrix among key factors
df=pd.DataFrame(houseprice_df, columns=['OverallQual', 'YearBuilt', 'GrLivArea','GarageCars'])
covMatrix = pd.DataFrame.cov(df).round(1)
covMatrix


# Until now, we analyze numeric data. However, I guess there would be another key factor that is not expressed in numbers

# # **String Data Analysis**
# 

# In[ ]:


# barchart of mean SalePrice 
houseprice_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df=pd.DataFrame(houseprice_df)

## compute mean SalePrice per MSZoning (C,FV,RH,RL,RM)
ax = houseprice_df.groupby('MSZoning').mean().SalePrice.plot(kind='bar')
ax.set_ylabel('Avg. SalePrice')


# In[ ]:


## compute mean SalePrice per BldgType (1Fam,2fmCon,Duplex,Twins,TwnhsE)

ax = houseprice_df.groupby('BldgType').mean().SalePrice.plot(kind='bar')
ax.set_ylabel('Avg. SalePrice')


# In[ ]:


## compute mean SalePrice per BldgType (Ex, Fa, Gd, Po, TA)

ax = houseprice_df.groupby('GarageCond').mean().SalePrice.plot(kind='bar')
ax.set_ylabel('Avg. SalePrice')


# In[ ]:


# group by MSZoning and Building Condition

houseprice_df.groupby(['MSZoning', 'BldgType'])['SalePrice'].mean()


# In[ ]:


# Analyzed by group (MSZoning - GarageCond - SalePrice)
houseprice_df.groupby(['MSZoning', 'GarageCond'])['SalePrice'].mean()


# In[ ]:


# Analyzed by group (BldgType - GarageCond - SalePrice)
houseprice_df.groupby(['BldgType', 'GarageCond'])['SalePrice'].mean()


# Those string factors are quite influential to SalePrice, but I feel insufficiency of the level of correlation. According to 'Korea Association of Property Appraisers', location requirements is one of the most influential constituents deciding housing price. Therefore, I think that 'Neighborhood' can be the key factor among string data information.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# SalePrice according to the neighborhood
ax = houseprice_df.plot.scatter(x='OverallQual', y='SalePrice', figsize=(20, 15))
points = houseprice_df[['OverallQual','SalePrice','Neighborhood']]
_ = points.apply(lambda x: 
             ax.text(*x, rotation=20, horizontalalignment='left',
                     verticalalignment='bottom', fontsize=9), axis=1)


# In[ ]:


# the number of data each neighborhood condition has
houseprice_df.Neighborhood.value_counts()


# In[ ]:


# Neighborhood Option between OverallQual 1 ~ 3
houseprice_df['Qual_bin'] = pd.cut(houseprice_df.OverallQual, range(0, 4), 
   labels=False)
houseprice_df.groupby(['Qual_bin', 'Neighborhood'])['SalePrice'].mean()


# In[ ]:


# Neighborhood Option between Overall Qual 4 ~ 6
houseprice_df['Qual_bin2'] = pd.cut(houseprice_df.OverallQual, range(3, 7), 
   labels=False)
houseprice_df.groupby(['Qual_bin2', 'Neighborhood'])['SalePrice'].mean()


# In[ ]:


# Neighborhood Option between OverallQual 8 ~ 10
houseprice_df['Qual_bin3'] = pd.cut(houseprice_df.OverallQual, range(7, 11), 
   labels=False)
houseprice_df.groupby(['Qual_bin3', 'Neighborhood'])['SalePrice'].mean()


# From this, we are able to know that speicific neighborhood requirements are only existed in high overall quality. Also, those high-grade neighborhood conditions lead to high price.

# # **Linear Regression Model**

# We know key factors superficially, but cannot assure the correctness. Thus, we should verify the correctness of our estimation. For this verification, I utilize 'linear regression model'.

# In[ ]:


# in order to remove rows with missing values 
reduced_df = houseprice_df.dropna()
print('Number of rows after removing rows with missing values: ', houseprice_df['GarageCars'].count())


# In[ ]:


# Prior to linear regression, sort out numeric data (it's because infinite or NaN data cannot be recognized in Python)

df=pd.DataFrame(houseprice_df, columns=['OverallQual','YearBuilt', 'YearRemodAdd', 'MasVnrArea','TotalBsmtSF', 
                                        '1stFlrSF', 'GrLivArea','TotRmsAbvGrd', 'Fireplaces',
                                        'GarageYrBlt','GarageCars', 'GarageArea','SalePrice'])
df.isna()


# In[ ]:


df.dropna(how='all').all


# In[ ]:


df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)


# In[ ]:


pip install dmba


# After dropping inappropriate value, we finally make train and valid data set.
# Then, we can assess whether the previous etimation is valid or not.
# In this part, I try to show the train model and valid model respectively.

# In[ ]:


import matplotlib.pylab as plt
from dmba import regressionSummary, classificationSummary, liftChart, gainsChart
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# create list of predictors and outcome
excludeColumns = ('SalePrice')
predictors = [s for s in df.columns if s not in excludeColumns]
outcome = 'SalePrice'
                  
# partition data
X = df[predictors]
y = df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
model = LinearRegression()
model.fit(train_X, train_y)
train_pred = model.predict(train_X)
train_results = pd.DataFrame({'SalePrice':train_y, 'predicted':train_pred, 'residual':train_y - train_pred})
train_results.head() 


# In[ ]:


# partition data
X = df[predictors]
y = df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
model = LinearRegression()
model.fit(train_X, train_y)
train_pred = model.predict(train_X)
train_results = pd.DataFrame({'SalePrice':train_y, 'predicted':train_pred, 'residual':train_y - train_pred})
train_results.head() 

model = LinearRegression()
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
train_results = pd.DataFrame({'TOTAL_VALUE': train_y, 'predicted': train_pred, 'residual': train_y - train_pred})

# show sample of predictions
train_results.head()


# In[ ]:


valid_pred = model.predict(valid_X)
valid_results = pd.DataFrame({'SalePrice': valid_y, 'predicted': valid_pred, 'residual': valid_y - valid_pred})
valid_results.head()


# In[ ]:


# import the utility function regressionSummary
from dmba import regressionSummary

X = df[predictors]
y = df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
model = LinearRegression()
model.fit(train_X, train_y)
train_pred = model.predict(train_X)
train_results = pd.DataFrame({'SalePrice':train_y, 'predicted':train_pred, 'residual':train_y - train_pred})
train_results.head() 


# training set
regressionSummary(train_results.SalePrice, train_results.predicted)

# validation set
regressionSummary(valid_results.SalePrice, valid_results.predicted)


# In[ ]:


pip install pydotplus


# In[ ]:


import math
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from sklearn.metrics import accuracy_score, roc_curve, auc
import pydotplus


# In[ ]:


print(pd.DataFrame({'Predictor':X.columns,'coefficient':model.coef_}))


# In[ ]:


valid_pred = model.predict(valid_X)
result=pd.DataFrame({'Predicted':valid_pred, 'Actual':valid_y, 'Residual':valid_y-valid_pred})
print(result.head(20))


# In[ ]:


pred_error_train = pd.DataFrame({
    'residual': train_y - model.predict(train_X), 
    'data set': 'training'
})
pred_error_valid = pd.DataFrame({
    'residual': valid_y - model.predict(valid_X), 
    'data set': 'validation'
})
boxdata_df = pred_error_train.append(pred_error_valid, ignore_index=True)

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(9, 4)
common = {'bins': 100, 'range': [-100000, 100000]}
pred_error_train.hist(ax=axes[0], **common)
pred_error_valid.hist(ax=axes[1], **common)
boxdata_df.boxplot(ax=axes[2], by='data set')

axes[0].set_title('training')
axes[1].set_title('validation')
axes[2].set_title(' ')
axes[2].set_ylim(-100000, 100000)
plt.suptitle('Prediction errors') 
plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.35)

plt.show()


# Throught the linear regression model, We can identify that our estimation is valid.
# However, as you see in graphs above, there are still a few errors. In my opinion, the reason is that
# string data were not used in this linear regression model, despite their importance and impact.
# Accordingly, it is more accurate approach to consider the string data.
# In order to do this, I make another data frame and utilize it with Cluster analysis.

# # **Cluster Analysis**

# At first, make data frame named 'neighbortest_df' as I make 'df' in previous step (linear regression).
# Plus, use 'Neighborhood' column as an index.

# In[ ]:


neighbortest_df=pd.DataFrame(houseprice_df, columns=['OverallQual','YearBuilt', 'YearRemodAdd', 'MasVnrArea','TotalBsmtSF', 
                                        '1stFlrSF', 'GrLivArea','TotRmsAbvGrd', 'SalePrice','Fireplaces',
                                        'GarageYrBlt','GarageCars', 'GarageArea','Neighborhood'])
neighbortest_df.set_index('Neighborhood', inplace=True)

# apply scale function
neighbortest_df = neighbortest_df.apply(lambda x: x.astype('float64'))
neighbortest_df.head(20)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# scikit-learn
neighbortest_df_norm = neighbortest_df.apply(preprocessing.scale, axis=0)

# pandas uses sample standard deviation
neighbortest_df_norm = (neighbortest_df - neighbortest_df.mean())/neighbortest_df.std()

# compute normalized distance based on SalePrice and OverallQual
d_norm = pairwise.pairwise_distances(neighbortest_df_norm[['SalePrice', 'OverallQual']], 
                                     metric='euclidean')
pd.DataFrame(d_norm, columns=neighbortest_df.index, index=neighbortest_df.index).head(20)


# As a result of this normalized distance chart, the distance between each neighborhood followed the difference in overall quality. For example, neighborhood group,{NoRidge, NridgHt, Somerst, etc}, which shows less distance toward each other belongs to the group of 'OverallQual 10'

# In[ ]:


neighbortest_df=neighbortest_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
neighbortest_df_norm = neighbortest_df.apply(preprocessing.scale, axis=0)
neighbortest_df_norm = (neighbortest_df - neighbortest_df.mean())/neighbortest_df.std()


Z = linkage(neighbortest_df_norm.head(60), method='average')

fig = plt.figure(figsize=(20, 15))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Average linkage)')
plt.xlabel('Neighborhood')
dendrogram(Z, labels=neighbortest_df_norm.index, color_threshold=2.75, leaf_rotation=45, leaf_font_size=12)
plt.axhline(y=2.75, color='black', linewidth=0.4, linestyle='dashed')
plt.show()


# In[ ]:


memb = fcluster(linkage(neighbortest_df_norm, 'average'), 15, criterion='maxclust')
memb = pd.Series(memb, index=neighbortest_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# It is difficult to clearly classify and make links among neighborhoods factor, since the number of data is too big to deal with handily. (1460rows)

# In[ ]:


neighbortest_df_norm.index = ['{}: {}'.format(cluster, Neighborhood) for cluster, Neighborhood
                              in zip(memb, neighbortest_df_norm.index)]
sns.clustermap(neighbortest_df_norm.head(60), method='average', col_cluster=False,  cmap="mako_r")
plt.show()


# Fortunately, this chart shows us the correlation between 'neighborhood' and other numeric factors.
# As we see, 'StoneBr', 'NoRidge', 'NridgHt', which belongs to high quality group has strong-positive relationship with SalePrice.

# In[ ]:


kmeans = KMeans(n_clusters=6, random_state=0).fit(neighbortest_df_norm)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=neighbortest_df_norm.columns)
pd.set_option('precision', 3)
print(centroids)
pd.set_option('precision', 6)


# In[ ]:


withinClusterSS = [0] * 6
clusterCount = [0] * 6
for cluster, distance in zip(kmeans.labels_, kmeans.transform(neighbortest_df_norm)):
    withinClusterSS[cluster] += distance[cluster]**2
    clusterCount[cluster] += 1
for cluster, withClustSS in enumerate(withinClusterSS):
    print('Cluster {} ({} members): {:5.2f} within cluster'.format(cluster, 
        clusterCount[cluster], withinClusterSS[cluster]))


# In[ ]:


# calculate the distances of each data point to the cluster centers
distances = kmeans.transform(neighbortest_df_norm)

# reduce to the minimum squared distance of each data point to the cluster centers
minSquaredDistances = distances.min(axis=1) ** 2

# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_}, 
    index=neighbortest_df_norm.index)

# Group by cluster and print information
for cluster, data in df.groupby('cluster'):
    count = len(data)
    withinClustSS = data.squaredDistance.sum()
    print(f'Cluster {cluster} ({count} members): {withinClustSS:.2f} within cluster ')


# In[ ]:


centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]

plt.figure(figsize=(10,7))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=4)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(-0.5,7.5)
centroids


# In[ ]:


centroids


# In[ ]:


centroids_df=centroids
centrodis_df=centroids.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# create list of predictors and outcome
excludeColumns = ('SalePrice', 'cluster')
predictors = [s for s in centroids_df.columns if s not in excludeColumns]
outcome = 'SalePrice'
                  

X2 = centroids_df[predictors]
y2 = centroids_df[outcome]
train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X2, y2, test_size=0.3, random_state=1)
model2 = LinearRegression()
model2.fit(train_X2, train_y2)
train2_pred = model2.predict(train_X2)
train2_results = pd.DataFrame({'SalePrice':train_y2, 'predicted':train2_pred, 'residual':train_y2 - train2_pred})
train2_results.head() 


# In[ ]:


valid2_pred = model2.predict(valid_X2)
valid2_results = pd.DataFrame({'SalePrice': valid_y2, 'predicted': valid2_pred, 'residual': valid_y2 - valid2_pred})
valid2_results.head()


# In[ ]:


# training set
regressionSummary(train2_results.SalePrice, train2_results.predicted)

# validation set
regressionSummary(valid2_results.SalePrice, valid2_results.predicted)


# In[ ]:


centroids_df['cluster'] = centroids_df.cluster.apply(lambda x: x.split('r')[-1])

print(centroids_df)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

df2 = pd.DataFrame(centroids_df)
model2 = AgglomerativeClustering(n_clusters=6)

model2.fit(df2)

y2_predict = model2.fit_predict(df2)
print(y2_predict) 

df2['cluster'] = y2_predict
print(df2)


# In[ ]:


test_df=pd.DataFrame(test, columns=['OverallQual','YearBuilt', 'YearRemodAdd', 'MasVnrArea','TotalBsmtSF', 
                                        '1stFlrSF', 'GrLivArea','TotRmsAbvGrd','Fireplaces',
                                        'GarageYrBlt','GarageCars', 'GarageArea'])
test_df=test_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

y3_predict=model2.fit_predict(test_df)
submission = y3_predict


# In[ ]:


print(submission)


# In[ ]:


submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


submission.to_csv('First_Kaggle_Submission.csv')


# In[ ]:




