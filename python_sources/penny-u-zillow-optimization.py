#!/usr/bin/env python
# coding: utf-8

# # Exploration of Zillow Data
# ### This is a work in progress and is still being updated.

# In[ ]:


# Load in libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Read in datasets / information 
y_train = pd.read_csv('../input/train_2016_v2.csv', index_col='parcelid', low_memory=False)
x_train = pd.read_csv('../input/properties_2016.csv', index_col='parcelid', low_memory=False)
datadict = pd.read_excel('../input/zillow_data_dictionary.xlsx', index_col='Feature')
datadict.index = datadict.index.str.replace('\'', '')


# In[ ]:


# Let's merge the x and y dataframes -- this will give us a complete set
train = pd.merge(x_train, y_train, left_index=True, right_index=True, how='right')
print(train.shape)


# Let's brainstorm a list of things that are going to play into the home price. But we are not just interested in the price -- we're interested in the *error* in predicting home price. So this may be things that Zillow is not taking into account under normal circumstances.
# 
# - Location, incl. state, county and neighborhood
# - Public services available, e.g. schools
# - Timing, incl. season and month
# - Remodeled home
# - Other flipping homes
# - Gentrification booms
# 
# Now, let's start to look at the actual data. First, let's look at which fields have a lot of NULL data, which might influence the interpretation of our summary statistics.'

# In[ ]:


# Let's find the values that are most present
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(1,1, figsize=(8,6))
numnulls = x_train.notnull().sum()
numnulls.sort_values().plot(kind='barh', figsize=(4,8), ax=ax)
ax.set_title('Number of Valid Values per Column')
plt.show()


# ### Column Inter-Correlations
# We have different types of data here: 
# - Continuous variables (floor size, year built)
# - Grouping variables (zip code, architecture type)
# 
# It also looks like there are some redundancies in here. For example, "tax amount", "taxvaluedollarcnt" and "structuretaxvaluedollarcnt" are probably all correlated. However, if an entry is missing one of these features (or more), the algorithm may under/over-weight them. 
# 
# So, let's check out whether a few sets of columns are correlated or not using seaborn's "pair plot"

# In[ ]:


# Different columns that seem like they might be related
g1 = ['taxamount', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt','landtaxvaluedollarcnt'] 
#g2 = ['assessmentyear', 'yearbuilt']
#g3 = ['bedroomcnt', 'bathroomcnt', 'calculatedbathnbr', 'fullbathcnt', 'roomcnt']
#g4 = ['calculatedfinishedsquarefeet','finishedsquarefeet12', 'lotsizesquarefeet']

gp = g1
# The pairplot creates a matrix of plots: on the diagonal is a histogram/density plot for 
# a single column. On the off-diagonals are bivariate correlation plots, which help us get a sense
# of how much independence there is between columns.
axes = sns.pairplot(train[gp].dropna(), diag_kind='kde',
                    markers='.', dropna=True)

plt.suptitle('Correlations Among Tax-Related Columns', fontsize=18, y=1.05)
plt.show()


# ### Independent Component Analysis
# One way of handling these inter-correlations is to "decompose" the dataframe into smaller units. We could do this for small groups of values to get a stronger "tax" feature, for example. Or, we can do it to the entire dataframe. 
# 
# Let's use Independent Components Analysis to see what we can find from all the columns in the dataset. 

# In[ ]:


# Since there are a lot of NaN values, we need to fill them in, or else the algorithm
# has to drop the data record entirely.
from sklearn.preprocessing import Imputer
# We only want to compute components for non-ID-like, continuous variables
ica_cols = [col for col in train.columns if (train[col].dtype == 'float64') & (not 'id' in col)]
ica_cols.remove('logerror')
imp = Imputer(missing_values='NaN', strategy='median')
imp_qcols = imp.fit_transform(X=train[ica_cols])


# Now, let's import FastICA and set our number of components. This is somewhat arbitrary, and
# the number chosen *will* affect the components output. Let's try 5.
from sklearn.decomposition import FastICA
ncomp = 5
ica = FastICA(n_components=ncomp)
ica_data = ica.fit_transform(imp_qcols)

# Now, let's re-format the data for viewing
comp_names = ['component_{}'.format(x+1) for x in np.arange(ncomp)]
ica_dx = pd.DataFrame(data = ica.mixing_, 
                      columns=comp_names, 
                      index=ica_cols)
ica_df = pd.DataFrame(ica_data, 
                      columns=comp_names, 
                      index=train.index)

# Let's look at the relative weights of each column for each component
ica_dx.astype(int)


# In[ ]:


# Now let's check whether these columns are truly "independent" -- and whether they predict logerror
sns.pairplot(ica_df.join(train['logerror'], how='left'),
             diag_kind='kde', markers='.')
plt.show()


# ## See What Columns Predict Membership in "Outlier" Class 
# 
# Another way we can approach this problem is to to try and target and reduce "outlier" properties, since these are disproportionately going to throw off the Zestimate. 
# 
# What we're going to do below is:
# - Define an outlier group (cut-off of abs(logerror) > 0.25)
# - For each float-like column, we'll test whether there is a difference in the mean of the distribution between the outlier and insider groups.

# In[ ]:


from scipy.stats import ttest_ind

# Distinguish outliers from other entries
outs = train.logerror.abs() > 0.25
inside = train.loc[outs==False]
outside = train.loc[outs]

# Initialize results data frame
insideout = pd.DataFrame(columns=['tstat', 'pval', 'n_inside', 'n_outside'])

# Loop through cols and get difference stats for in / out
float_cols = [col for col in train.columns if (train[col].dtype == 'float64') & (not 'id' in col)]
for col in float_cols:
    try:
        s = pd.Series(name=col)
        s['tstat'], s['pval'] = ttest_ind(inside[col], outside[col], nan_policy='omit')
        s['n_inside'] = inside[col].notnull().sum()
        s['n_outside'] = outside[col].notnull().sum()
        insideout = insideout.append(s)
    except TypeError as exc:
        print('{}: {}'.format(col, exc))
        
# Let's look at columns that seem to be different, but only those that are populated by outlier data
(insideout.loc[(insideout.tstat.abs() > 5) & (insideout.n_outside > 3000)]
    .sort_values(by='tstat'))


# In[ ]:


# Let's take a look at a few of these:
metric = 'yearbuilt'

def plot_boxplot_column(col):    
    fig, ax = plt.subplots(1,1)
    train.boxplot(column=col, by=outs.values, ax=ax)

    # Have to play around a bit with the legend and title...
    fig.suptitle('{} by Outlier Designation'.format(metric))
    ax.set_title('')
    ax.set_xlabel('Outlier?')
    ax.set_ylabel(metric)
    #ax.legend().set_visible(False)    
    
    return  ax 

ax = plot_boxplot_column(metric)
#ax.set_ylim([0, 10])
plt.show()


# ## Finally, let's bring this home.

# In[ ]:


# Set up our training data
cat_cols = [col for col in train.columns if ('id' in col) or (col in ['fips'])]
float_cols = [col for col in train.columns if (train[col].dtype == 'float64') 
                                              & (not col in cat_cols)
                                              & (not col in ['latitude', 'longitude', 
                                                             'censustractandblock', 'logerror'])]
X_train = train[float_cols].fillna(train.median()).values
y_train = train['logerror'].values
feature_names = train[float_cols].columns.values

from sklearn.svm import SVR
model = SVR(C=1)
model.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_train)


# In[ ]:


# Get a baseline, no seasonal differnces
predictions = pd.Series(model.predict(X_train), index=train.index)

# The output file should have 18232 prediction rows, with six predicted values each
predict_df = pd.DataFrame(data=s, columns=['201610'])
for col in ['201611','201612','201710','201711','201712']:
    predict_df[col] = predict_df['201610']
    
predict_df.to_csv('submission.csv')


# In[ ]:


import os
os.listdir()


# In[ ]:




# Import Random Forest Regressor
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(n_estimators=10, max_depth=5, 
                            max_features=0.3, criterion='mae', 
                            n_jobs=-1, random_state=0)
model.fit(X_train, y_train)

## Get importance of each feature, and order it
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20] # [::-1] reverses the order, [:20] takes top 20

# Plot it!
plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()
import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

