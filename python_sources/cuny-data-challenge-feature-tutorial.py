#!/usr/bin/env python
# coding: utf-8

# ## tutorial on feature importance, feature selection methods, and finding alternative features!
# In this notebook we're going to:
# 
# 1. Build a feature set using data provided in the competition
# 2. Find some new data and enrich our analysis with it
# 3. Try to figure out which features are helpful and which aren't
# 4. Discuss methods for determining a good 'feature set'
# 
# 

# ### Creating a starting feature set
# First, we're going to repeat many of the steps from [**kernel_3**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel3). If this looks new, I recommend starting with the kernel instead.

# In[ ]:


# load some important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# load the inspections data and split into train/test
d = pd.read_csv('../input/cuny-data-challenge-2019/inspections_train.csv', parse_dates=['inspection_date'])
x_train, x_test = train_test_split(d, test_size=0.25)

# load the venue data
venue_stats = pd.read_csv('../input/cuny-data-challenge-2019/venues.csv').set_index('camis')

# load the violations data
violations = pd.read_csv('../input/cuny-data-challenge-2019/violations.csv', parse_dates=['inspection_date'])

# merge the venue stats with the inspection data
x_train = x_train.merge(venue_stats, 'left', left_on='camis', right_index=True)
x_test = x_test.merge(venue_stats, 'left', left_on='camis', right_index=True)

# create the violation count feature
violation_counts = violations.groupby(['camis', 'inspection_date']).size()
violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])
violation_counts.columns = ['n_violations']

# add the violation counts to the main data set
x_train = x_train.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_test = x_test.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)

# add the inspect / re-inspect feature to our train / test data
x_train['re_inspect'] = x_train.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_train['initial_inspect'] = x_train.inspection_type.str.contains('initial', regex=False, case=False).map(int)

x_test['re_inspect'] = x_test.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_test['initial_inspect'] = x_test.inspection_type.str.contains('initial', regex=False, case=False).map(int)

# add our borough specific encoding
boro_dict = {
    'Missing': 0,
    'STATEN ISLAND': 0,
    'BROOKLYN': 1,
    'MANHATTAN': 1,
    'BRONX': 2,
    'QUEENS': 2    
}

x_train['boro_idx'] = [boro_dict[_] for _ in x_train.boro]
x_test['boro_idx'] = [boro_dict[_] for _ in x_test.boro]

# add the inspection month
x_train['inspection_month'] = (x_train.inspection_date.dt.strftime('%m').map(int) + 6) % 12
x_test['inspection_month'] = (x_test.inspection_date.dt.strftime('%m').map(int) + 6) % 12

# add the cuisine hitrates
cuisine_hitrates = x_train.groupby(['cuisine_description']).agg({'passed':'mean', 'id':'count'}).        rename(columns={'id':'ct'}).sort_values('passed')[['passed']]
cuisine_hitrates.columns = ['cuisine_hr']

x_train = x_train.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True)
x_test = x_test.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True)

model_features = ['n_violations', 'inspection_month', 'cuisine_hr', 'boro_idx', 're_inspect', 'initial_inspect']


# ### Let's use MORE data!
# I wanted to see if we could potentially enrich our analysis with some additional information, so I clicked on the **"+ Add Dataset"** button at the top of this screen and typed in "NYC data". I found some open source IRS data that another user had already uploaded with a "Public Domain" license. It looks reasonable- check it out [**here**](https://www.kaggle.com/jakerohrer/zip-codes-and-stats/).
# 
# I added this data set to this kernel, and now it appears in my `input` folder. Take a look:

# In[ ]:


# we should now see a directory for our CUNY data as well as a new folder!
os.listdir('../input/')


# Let's load this data and find out the median of the `Median` field, which appears to be the median household income for each zipcode.

# In[ ]:


zip_data = pd.read_csv('../input/zip-codes-and-stats/medzip.csv').set_index('Zipcode')
zip_data.Median.median()


# Oh no- that broke! How come? The error message says it couldn't convert string to float... why is that? Take a look at the `Median` field. That looks like a comma in the value, and the `dtype` listed says `object` when we should really be seeing a numeric value.

# In[ ]:


zip_data.Median.head()


# Luckily, I went to the `pandas` [**documentation for the `read_csv()` function**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html). Looks like there's a specific function argument, `thousands`, to tackle this issue. Looks like it's working now:

# In[ ]:


zip_data = pd.read_csv('../input/zip-codes-and-stats/medzip.csv', thousands=',').set_index('Zipcode')
zip_data.columns = ['zip', 'med', 'population']
zip_data.head()


# Now that we have well-behaved zip data, let's merge it with our training data. We'll merge on zip code, and we really only want the median income and population data.

# In[ ]:


x_train = x_train.merge(zip_data[['med', 'population']], 'left', left_on='zipcode', right_index=True)
x_test = x_test.merge(zip_data[['med', 'population']], 'left', left_on='zipcode', right_index=True)


# Let's add the new features to our list and check out our dataframe.

# In[ ]:


new_features = model_features + ['med', 'population']
target = ['passed']
x_train[new_features + target].head()


# Whenever we add new features, it's import to evaluate them to make sure we don't have missing values or outliers. Let's take a look

# In[ ]:


x_train[new_features].isna().sum()


# Ok, it looks like we're missing some zip codes. It's only about 3\% of our data, but we shouldn't drop it unless we have to. Handling missing data is an important step and there isn't always a right answer. In this case, let's fill them with the average values for each field.

# In[ ]:


x_train['med'] = x_train.med.fillna(x_train.med.mean())
x_train['population'] = x_train.population.fillna(x_train.population.mean())


# ### Feature importance
# 
# There are many ways of determining the value of a feature, but we can separate them broadly into a few categories.
# 
# - **filter methods**: This is a form of *univariate* feature selection (meaning we only look at one feature at a time). In using a filter method, we'll evaluate a feature and the value we're trying to predict using a scoring method (eg: evaluating the relationship between `n_violations` and `passed`).
# - **wrapper methods**: A wrapper method is a *multivariate* selection methodology where we follow a series of steps to train a model on a set of features and then add or remove features to improve our loss function.
# - **embedded methods**: Embedded methods are also *multivariate*, and leverage the fact that some machine learning models will attempt to *do their own feature selection*. We can evaluate feature importance using one of these models (and maybe even use it with our other models!)
# 
# Now let's import some tools, and then separate our features into numeric and categorical types.

# In[ ]:


# for filter method
from sklearn.feature_selection import f_classif, SelectKBest

# for wrapper method
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# for embedded method
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

numeric_features = ['n_violations', 'cuisine_hr', 'med', 'population']
categorical_features = ['inspection_month', 'boro_idx', 're_inspect', 'initial_inspect']


# #### Filter methods
# 
# For now, let's just focus on the numeric features. This problem contains a lot of categorical variables, and there's no one right way to handle those. There are many options, which means you get to be creative in how you communicate the value of those features to the model.
# 
# First, let's use `SelectKBest` to pick to rank our features according to the `f_classif` score.

# In[ ]:


# create the selector object with scoring metric.
# we are specifying the 4 best features here, but in reality we'd probably want fewer!
selector = SelectKBest(f_classif, 4)

# fit the selector on our numeric features and our target feature
# note: 'ravel' is a way to flatten a dataframe into a vector
selector.fit(x_train[numeric_features], x_train[target].values.ravel())

# convert the scores into a dataframe object
feature_scores = pd.DataFrame({'features': numeric_features,
                               'scores': selector.scores_})
feature_scores


# It looks like `n_violations` is by far our most important feature. The `med` and `population` features we recently added might not even be important at all.

# #### Wrapper methods
# 
# Next we'll try a feature subset search algorithm called [**Recursive Feature Elimination**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV). We're going to be using a variant of it called **RFECV** that applies the algorithm with cross-validation automatically. If you're not familiar with cross-validation, check out the kernel on it; it's really important!

# In[ ]:


# set up the model
mod = LogisticRegression(solver='sag', penalty='l2', max_iter=500)

# set up the selector
selector = RFECV(mod, cv=5, scoring='neg_log_loss')

# we're going to get a convergence warning, so we'll filter those out here
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # fit the selector
    selector.fit(x_train[numeric_features + categorical_features], x_train[target].values.ravel())

# convert the scores into a dataframe object
feature_scores = pd.DataFrame({'features': numeric_features + categorical_features,
                               'scores': selector.grid_scores_*-1})
feature_scores.sort_values('scores')


# #### Embedded methods
# 
# We'll try training a `RandomForestClassifier` on the data next. This model is fairly standard for embedded feature selection because the construction process involves repeated random sampling of features and bootstrapping of observations. 

# In[ ]:


mod = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20)
mod.fit(x_train[numeric_features + categorical_features], x_train[target].values.ravel())

feature_scores = pd.DataFrame({'features': numeric_features + categorical_features,
                               'scores': mod.feature_importances_})
feature_scores.sort_values('scores', ascending=False)


# ### Summary
# It's  clear that each of our feature selection methods yielded slightly different results, but we're able to see some patterns emerging. This part of data science is really more art than science; you should work to blend your understanding of the feature importance scores with your knowledge of the domain. If a feature you think should be important isn't getting a high score, are you sure you're encoding it properly? Should your normalize it somehow? Should you remove outliers? Is it in only important in *combination* with some other feature or aspect of the problem?
# 
# In addition to locating your most useful features, you should also use this as an opportunity to remove your features that don't look useful- especially if this conforms to your view of the problem.|
