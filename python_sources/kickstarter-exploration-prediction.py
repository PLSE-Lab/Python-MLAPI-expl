#!/usr/bin/env python
# coding: utf-8

# ## Brief contents
# ### 0. Introduction
# * 0.1 How Kickstarter works
# * 0.2 Summary of the data
# * 0.3 Import packages
# 
# ### 1. Genaral
# * 1.1 Handling weird values
# * 1.2 Modify data columns
# * 1.3 Numbers of projects proposed/ successful rate
# * 1.4 Statistics of pledged amount
# 
# ### 2. Region
# * 2.1 Projects proposed across regions
# 
# ### 3. Caterogy
# * 3.1 Projects proposed across categories
# * 3.2 Pledged amount across categories
# * 3.3 Success rate across categories
# 
# ### 4. Backers
# * 4.1 Discover categories that attract most backers
# * 4.2 The distribution of backers
# * 4.3 Relationship between backers & pledged amounts
# 
# ### 5. Modeling
# * 5.1 Pledged amount prediction - Linear Regression
# * 5.2 Project state prediction - Random Forest
# 
# --------
# --------

# ## 0. Introduction
# ------
# #### 0.1  About Kickstarter 
# Launched in 2009, Kickstarter has now became a world famous online crowdfunding platform. The platform mainly focus on creativity and merchandising, which has made some of the most fantastic ideas to come true.
# 
# The crowdfunding process is simple:
# * Project owners propose their projects and provide related information such as idea, pricing, schedule, etc. 
# * Backers back the projects that seems attractive
# 
# The purpose of this kernel is to  explore the data collected from Kickstarter, trying to understand some characteristics of the platform.

# #### 0.2 Summary of the data
# We have 2 data files in this event:
# 1. ks-projects-201612.csv
# 2.  ks-projects-201801.csv 
# 
# The `ks-projects-201612` data file is contained by the `ks-projects-201801`. We will use the  `ks-projects-201801` for further exploration.
# 
# Some characteristic of this dataset:
# - 378661 rows, 15 columns
# - Metadata
# 
# |              | `Description`  |	`Attribute`  |
# |------------------------|
# | `ID`         |internal id      | Numeric |
# | `name`  | name of the project | String |
# | `category` | sub category (159) | String |
# | `main_category`| main category (15) | String |
# | `currency` | currency defined by project owner | String |
# | `dealine` | deadline | DateTime|
# | `goal` | fundraising goal | Numeric|
# | `launched` | launched time | DateTime|
# | `pledged` | the pledged amount | Numeric |
# | `state` | state of the project (successful, failed, etc.) | String|
# | `backers` | number of backers | Numeric |
# | `country` | country | String |
# | `usd_pledged` | pledged amount in USD | Numeric |
# | `usd_pledged real` | pledged amount in USD | Numeric |
# | `usd_ goal_real` | pledged goal in USD | Numeric |
# 
# #### 0.3 Import packages
#     

# In[68]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
import squarify
import time
import os


# --------

# ## 1. General

# #### 1.1 Handling weird values
# - Check decriptive statistics
# - Check missing values

# In[69]:


ksdf = pd.read_csv('../input/ks-projects-201801.csv', sep=',', engine='python')
ksdf.head()


# In[70]:


ksdf.describe()


# In[71]:


ksdf.isnull().sum()


# According to the description of data:
# - **usd pledged** -> USD conversion made by KS for the pledged value
# - **usd_pledged_real** ->  USD conversion made by fixer.io api
# 
# We will use **usd_pledged_real** for further analysis

# In[72]:


# Check missing values in the column "name"
ksdf[pd.isnull(ksdf['name'])].index


# In[73]:


ksdf[ksdf.index == 166851]
ksdf[ksdf.index == 307234]


# Even though there are some missing values in **name**, those are real projects. We will keep them.

# #### 1.2 Modify data columns
# - According to previous cell, I will drop **usd pledged** columns
# - Add **pledged_ratio** column that follows the rule:
#  > pledged_ratio = usd_pledged_real/ usd_goal_real
# - Retrieve **year** & **date** and transform the information as a new column **time** that follows the rule:
# > `(year - 2009) * 12 + month`
# >>Since 2009 seems to be the first year that contains meaningful data, we will use this alorithm to get timely manner information
# - Generate a new dataframe **ksdf_year** for cross-year comparison

# In[74]:


# Delete usdf pledged column
ksdf.drop(['usd pledged'], axis = 1, inplace = True)


# In[75]:


# Add pledged_ratio column
ksdf['pledged_ratio'] = ksdf['usd_pledged_real']/ ksdf['usd_goal_real']


# In[76]:


def year_cut(string):
    return string[0:4]

def month_cut(string):
    return string[5:7]

ksdf['year'] = ksdf['launched'].apply(year_cut)
ksdf['month'] = ksdf['launched'].apply(month_cut)

ksdf['year'] = ksdf['year'].astype(int)
ksdf['month'] = ksdf['month'].astype(int)
ksdf['time'] = (ksdf['year'].values - 2009)*12 + (ksdf['month']).astype(int)


# In[77]:


print (ksdf.columns)


# In[78]:


ksdf['year'].value_counts()


# In[79]:


ksdf_year = {}
for year in range(2009, 2019):
    ksdf_year[year] = ksdf[ksdf['year'] == year]['year'].count()


# In[80]:


ksdf_year = pd.Series(ksdf_year)
ksdf_year = pd.DataFrame(ksdf_year)
ksdf_year = ksdf_year.rename(columns = {0: "counts"})
ksdf_year


# 

# #### 1.3 Number of projects proposed/  successful rate
# - Overall number of proposed projects
# - Overall successful rate
# - Cross year comparison of above 2 items

# In[81]:


ksdf['state'].value_counts()


# In[82]:


squarify.plot(sizes=[197719,133956, (38779 + 3562 + 2799 + 1846)], 
              label=["Failed (52.22%)", "Successful (35.38%)", "Others (10.24%)",], color=["blue","red","green"], alpha=.4 )
plt.title('State', fontsize = 20)
plt.axis('off')
plt.show()


# In[83]:


success_timely = []

for year in range(2009, 2019):
    success = len (ksdf[(ksdf['year'] == year) & (ksdf['state'] == 'successful')]['state'])
    overall = len (ksdf[ksdf['year'] == year]['year'])
    ratio = success/ overall
    success_timely.append(ratio)
    print ("Year = ",year, ratio * 100, '%')


# In[84]:


ksdf[ksdf['year'] == 2018]['state'].value_counts()


# In[85]:


ksdf_year['success_ratio'] = success_timely
ksdf_year.head


# In[86]:


ksdf[ksdf['year'] == 2017]['backers'].count()


# In[87]:


backers_year = {}
for year in range(2009, 2019):
    backers_count = ksdf[ksdf['year'] == year]['backers'].sum()
    backers_year[year] = backers_count

ksdf_year['backers'] = pd.Series(backers_year)


# In[88]:


ksdf_year


# In[89]:


# Cross-year proposed projects
sns.set_style("whitegrid")
sns.barplot(ksdf_year['counts'].index, y= ksdf_year['counts'] ,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)


# In[90]:


# Cross-year success ratio
sns.set_style("whitegrid")
sns.barplot(ksdf_year['success_ratio'].index, y= ksdf_year['success_ratio'], data = ksdf_year,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)


# 

# In[91]:


sns.set_style("whitegrid")
sns.barplot(ksdf_year['backers'].index, y= ksdf_year['backers'] ,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)


# #### 1.4 Statistics of pledged amount
# - Descripitive statistics of pledged amount
# - Pledged amount comparison by state (successful/ failed/ others)

# In[92]:


sum_pledged = ksdf['usd_pledged_real'].sum()
print (sum_pledged)


# In[93]:


ksdf['usd_pledged_real'].describe()


# In[94]:


# Ratio of successful/ failed / others
success_pledged = ksdf[ksdf['state'] == "successful"]['usd_pledged_real'].sum()
fail_pledged = ksdf[ksdf['state'] == 'failed']['usd_pledged_real'].sum()
others_pledged = (ksdf[ksdf['state'] == 'canceled']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'undefined']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'live']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'suspended']['usd_pledged_real'].sum())

print (success_pledged, success_pledged/ sum_pledged * 100, '%')
print (fail_pledged, fail_pledged/ sum_pledged * 100, '%')
print (others_pledged, others_pledged/ sum_pledged * 100, '%')


# In[95]:


squarify.plot(sizes=[3036889045.99, 261108466.05, 132263736.79], 
              label=["Successful (88.53%)", "Failed (7.61%)", "Others (3.86%)",], color=["red","blue", "green"], alpha=.4 )
plt.title('Pledged Amount', fontsize = 20)
plt.axis('off')
plt.show()


# In[96]:


success_projects = ksdf[ksdf['state'] == 'successful']['state'].count()
fail_projects  = ksdf[ksdf['state'] == 'failed']['state'].count()
others_projects  = (
    ksdf[ksdf['state'] == 'canceled']['state'].count() +
    ksdf[ksdf['state'] == 'live']['state'].count() +
    ksdf[ksdf['state'] == 'undefined']['state'].count() +
    ksdf[ksdf['state'] == 'suspended']['state'].count())

print ("Average pledged amount per successful project = ",success_pledged/success_projects)
print ("Average pledged amount per failed project = ",fail_pledged/ fail_projects)
print ("Average pledged amount per other project = ",others_pledged/ others_projects)


# In[97]:


sns.set_style("whitegrid")
sns.barplot(["Successful", "Failed", "Others"],
            y= [22670.7952312, 1320.60381678, 2814.96055825],
            palette = "Set1",
            saturation = 0.5)
sns.despine(right = True, top = True)


# 

# ## 2. Region
# - 2.1 Projects proposed across regions

# In[98]:


ksdf['country'].unique()


# In[99]:


ksdf['country'].value_counts()


# In[100]:


sns.countplot(ksdf['country'], palette = 'Set1', order = ksdf['country'].value_counts().index)
sns.despine(bottom = True, left = True)


# In[101]:


us = ksdf[ksdf['country'] == "US"]['country'].count()
print (us/len(ksdf['country']) * 100, "%")


# 

# 
# ## 3. Caterogy
# * 3.1 Projects proposed across categories
# * 3.2 Pledged amount across categories
# * 3.3 Success rate across categories
# 
# We will go through this part by creating a new dataframe **cate_df** to record some information, including:
# - **pledged_sum** -> Sum of the pledged money for each categories
# - **count** -> Project counts for each categories
# - **average_amount** -> Average pledged amount for each categories 
# - **success_count** -> Successful projects counts for each categories
# - **success_rate** -> The ratio of success for each categories

# Adding **pledged_sum**

# In[102]:


pledged_sum = {}
for category in list(set(ksdf['main_category'])):
    amount = ksdf[ksdf['main_category'] == category]['usd_pledged_real'].sum()
    pledged_sum[category] = amount

# Create dataframe
cate_df = pd.Series(pledged_sum)
cate_df = pd.DataFrame(cate_df)
cate_df = cate_df.rename(columns = {0:"pledged_sum"})

cate_df.head()


# Adding **count**

# In[103]:


cate_count = {}
for category in list(set(ksdf['main_category'])):
    count = ksdf[ksdf['main_category'] == category]['main_category'].count()
    cate_count[category] = count
    
cate_df['count'] = pd.Series(cate_count)

cate_df.head()


# Adding **average_amount**

# In[104]:


cate_df['average_amount'] = cate_df['pledged_sum']/ cate_df['count']
cate_df.head()


# Adding **success_rate**

# In[105]:


success = {}
for category in list(set(ksdf['main_category'])):
    success_count = len(ksdf[(ksdf['main_category'] == category) & 
         (ksdf['state'] == "successful")])
    success[category] = success_count

cate_df["success_count"] = pd.Series(success)
cate_df.head()


# In[106]:


cate_df["success_rate"] = cate_df['success_count']/ cate_df['count']
cate_df.head()


# In[107]:


# pledged_sum plot
cate_df = cate_df.sort_values('pledged_sum',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['pledged_sum'].index, y= cate_df['pledged_sum'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)


# In[108]:


# avarage amount plot
cate_df = cate_df.sort_values('average_amount',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['average_amount'].index, y= cate_df['average_amount'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)


# In[109]:


# count plot
cate_df = cate_df.sort_values('count',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['count'].index, y= cate_df['count'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)


# In[110]:


# success rate plot
cate_df = cate_df.sort_values('success_rate',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['success_rate'].index, y= cate_df['success_rate'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)


# ## 4. Backers
# * 4.1 Discover categories that attract most backers
# * 4.2 The distribution of backers
# * 4.3 Relationship between backers & pledged amounts

# In[111]:


back_cate = {}

for category in set(ksdf['main_category']):
    backers = ksdf[ksdf['main_category'] == category]['backers'].sum()
    back_cate[category] = backers

backers = pd.Series(back_cate)
cate_df['backers'] = backers


# In[112]:


cate_df = cate_df.sort_values('backers',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['backers'].index, y= cate_df['backers'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)


# In[113]:


ksdf['backers'].quantile(list(np.arange(0,1,0.01))).plot(grid = 0, color = '#055968')


# In[114]:


sns.set_style("whitegrid")
sns.kdeplot(ksdf['backers'])
sns.despine(right = True, top = True)


# ### 5. Modeling
# * 5.1 Backers & pledged amount - `Linear Regression`
#     - Use **backers** to predict **usd_pledged_real**
#     - Retrieve projects that are not zero-pledged
#     - Model
#         - Input (X): **backers** (logarithm)
#         - Output (Y): **usd_pledged_real** (logarithm)
#         - Loss estimation: mean squared error
# * 5.2 Project state prediction - `Random Forest`
# - Data processing
#     - Encode **state** column to binary:
#         - successful -> 1
#         - failed -> 0
#         - else -> delete
#     - Concatenate new data set: `ksdf_rf`
#         - **main_category**
#         - **time** 
#         - **state**
#     - Data Split:
#         - train 70%
#         - test 30%
#     - Define X and Y
#         - X:
#             - category (one-hot)
#             -  main_category (one-hot)
#             - time
#         - Y:
#             - success/ failed (1/0)

# In[115]:


# Select not zero-pledged projects
non_zero = ksdf[ksdf['usd_pledged_real'] != 0]
print (non_zero.shape)


# In[116]:


# Define X and Y
X = ksdf[ksdf['usd_pledged_real'] != 0]['backers'].values
Y = ksdf[ksdf['usd_pledged_real'] != 0]['usd_pledged_real'].values

print (X.shape)
print (Y.shape)


# In[117]:


X = X.reshape(326134,1)
Y = Y.reshape(326134,1)


# In[118]:


# Model fitting and visualization
regr = linear_model.LinearRegression()
regr.fit(np.log(X+1), np.log(Y+1))

plt.scatter(np.log(X+1), np.log(Y+1))
plt.plot(np.log(X+1), regr.predict(np.log(X+1)), color='red', linewidth=3)
plt.show()


# In[119]:


# Results: error and parameters

Y_pred = regr.predict(np.log(X+1))
Y_true = np.log(Y+1)

print ("error = ", sklearn.metrics.mean_squared_error(Y_true, Y_pred))
print ("coefficient = ", regr.coef_)
print ("intercept = ", regr.intercept_)


# Encode **state** column to binary
# * successful -> 1
# * failed -> 0
# * else -> delete

# In[120]:


print (ksdf['state'].value_counts())
print ('')
print ("ksdf.shape = ", ksdf.shape)


# In[121]:


def state_change(cell_value):
    if cell_value == 'successful':
        return 1
    
    elif cell_value == 'failed':
        return 0
    
    else:
        return 'del'


# In[122]:


ksdf['state'] = ksdf['state'].apply(state_change)
print (ksdf[ksdf['state'] == 1].shape)
print (ksdf[ksdf['state'] == 0].shape)
print (ksdf[ksdf['state'] == 'del'].shape)
print (ksdf[ksdf['state'] == 1].shape[0] + ksdf[ksdf['state'] == 0].shape[0])


# Concatenate new DataFrame
# * **main_category**
# * **time** 
# * **state**

# In[123]:


ksdf_rf = ksdf.drop(ksdf[ksdf['state'] == 'del'].index)
print (ksdf_rf.shape)


# In[124]:


ksdf_rf = pd.concat([
                  ksdf_rf['main_category'],
                  ksdf_rf['time'],
                  ksdf_rf['state']], axis = 1
                 )

print (ksdf_rf.shape)


# Data Split:
# * Train: 70%
# * Test: 30%

# In[125]:


train, test = sklearn.model_selection.train_test_split(ksdf_rf, test_size = 0.3, random_state = 42)

print ("Train shape = ", train.shape, ",", len(train)/ len(ksdf_rf) * 100, "%")
print ("Test shape = ", test.shape, ",", len(test)/ len(ksdf_rf) * 100, "%")


# Define X and Y

# In[126]:


X_train = pd.concat(
    [
     pd.get_dummies(train['main_category'], prefix = 'main_category'),
     train["time"]
    ], axis=1)

Y_train = train['state']


# In[127]:


X_test = pd.concat(
    [
     pd.get_dummies(test['main_category'], prefix = 'main_category'),
     test["time"]
    ], axis=1)

Y_test = test['state']


# In[128]:


X_train = X_train.astype(int)
Y_train = Y_train.astype(int)
X_test = X_test.astype(int)
Y_test = Y_test.astype(int)

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)


# Random Forest

# In[129]:


for_record = {
    'baseline':{},
    'best_random1':{},
    'best_random2':{},
    'best_random3':{},
    'grid1':{},
    'grid2':{}
}


# In[130]:


start = time.time()
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, Y_train)
end = time.time()
sec = end - start
Y_pred = rf.predict(X_train)


# In[131]:


for_record['baseline']['params'] = rf.get_params()
for_record['baseline']['time'] = sec
for_record['baseline']["train_score"] = rf.score(X_train, Y_train)
for_record['baseline']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['baseline']['test_score'] = rf.score(X_test, Y_test)


# Randomized Search 1

# In[132]:


# Number of trees in random forest
n_estimators = [int(x) for x in range(2, 100, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(2,50,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_rand1 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random1 = RandomizedSearchCV(estimator = rf, param_distributions = param_rand1, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random1.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand1 = RandomForestClassifier(bootstrap = rf_random1.best_params_['bootstrap'],
                                 max_depth = rf_random1.best_params_['max_depth'],
                                 max_features = rf_random1.best_params_['max_features'],
                                 min_samples_leaf = rf_random1.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random1.best_params_['min_samples_split'],
                                 n_estimators = rf_random1.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand1.fit(X_train, Y_train)
end = time.time()
rand1_time = end - start
Y_pred = rf_rand1.predict(X_train)

for_record['best_random1']['params'] = rf_rand1.get_params()
for_record['best_random1']['time'] = rand1_time
for_record['best_random1']["train_score"] = rf_rand1.score(X_train, Y_train)
for_record['best_random1']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random1']['test_score'] = rf_rand1.score(X_test, Y_test)


# Grid Search 1

# In[133]:


# # Number of trees in random forest
# n_estimators = [int(x) for x in range(55, 65, 2)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in range(2, 20, 2)]
# # Minimum number of samples required to split a node
# min_samples_split = [2,5,10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1,2,4]
# # Method of selecting samples for training each tree
# bootstrap = [True,False]
# # Create the random grid
# param_grid1 = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_grid1 = GridSearchCV(estimator = rf, param_grid = param_grid1, cv = 3, verbose=2, n_jobs = -1)
# # Fit the random search model
# rf_grid1.fit(X_train, Y_train)

# # Use best random parameters to train a new model
# start = time.time()
# rf_grid1 = RandomForestClassifier(bootstrap = rf_grid1.best_params_['bootstrap'],
#                                  max_depth = rf_grid1.best_params_['max_depth'],
#                                  max_features = rf_grid1.best_params_['max_features'],
#                                  min_samples_leaf = rf_grid1.best_params_['min_samples_leaf'],
#                                  min_samples_split = rf_grid1.best_params_['min_samples_split'],
#                                  n_estimators = rf_grid1.best_params_['n_estimators'],
#                                  random_state = 42, n_jobs = -1)

# rf_grid1.fit(X_train, Y_train)
# end = time.time()
# grid1_time = end - start
# Y_pred = rf_grid1.predict(X_train)

# for_record['grid1']['params'] = rf_grid1.get_params()
# for_record['grid1']['time'] = grid1_time
# for_record['grid1']["train_score"] = rf_grid1.score(X_train, Y_train)
# for_record['grid1']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
# for_record['grid1']['test_score'] = rf_grid1.score(X_test, Y_test)


# Randomized Search 2

# In[134]:


# Number of trees in random forest
n_estimators = [int(x) for x in range(100, 200, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(50,100,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_random2 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random2 = RandomizedSearchCV(estimator = rf, param_distributions = param_random2, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random2.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand2 = RandomForestClassifier(bootstrap = rf_random2.best_params_['bootstrap'],
                                 max_depth = rf_random2.best_params_['max_depth'],
                                 max_features = rf_random2.best_params_['max_features'],
                                 min_samples_leaf = rf_random2.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random2.best_params_['min_samples_split'],
                                 n_estimators = rf_random2.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand2.fit(X_train, Y_train)
end = time.time()
rand2_time = end - start
Y_pred = rf_rand2.predict(X_train)
for_record['best_random2']['params'] = rf_rand2.get_params()
for_record['best_random2']['time'] = rand2_time
for_record['best_random2']["train_score"] = rf_rand2.score(X_train, Y_train)
for_record['best_random2']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random2']['test_score'] = rf_rand2.score(X_test, Y_test)


# Grid Search 2

# In[135]:


# # Number of trees in random forest
# n_estimators = [int(x) for x in range(100, 150, 2)]
# # Number of features to consider at every split
# max_features = 'auto'
# # Maximum number of levels in tree
# max_depth = [int(x) for x in range(2, 20, 2)]
# # Minimum number of samples required to split a node
# min_samples_split = 5
# # Minimum number of samples required at each leaf node
# min_samples_leaf = 2
# # Method of selecting samples for training each tree
# bootstrap = False
# # Create the random grid
# param_grid1 = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}


# Randomized Search 3

# In[136]:


# Number of trees in random forest
n_estimators = [int(x) for x in range(201, 300, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(100,150,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_random3 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random3 = RandomizedSearchCV(estimator = rf, param_distributions = param_random3, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random3.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand3 = RandomForestClassifier(bootstrap = rf_random3.best_params_['bootstrap'],
                                 max_depth = rf_random3.best_params_['max_depth'],
                                 max_features = rf_random3.best_params_['max_features'],
                                 min_samples_leaf = rf_random3.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random3.best_params_['min_samples_split'],
                                 n_estimators = rf_random3.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand3.fit(X_train, Y_train)
end = time.time()
rand3_time = end - start
Y_pred = rf_rand3.predict(X_train)
for_record['best_random3']['params'] = rf_rand3.get_params()
for_record['best_random3']['time'] = rand3_time
for_record['best_random3']["train_score"] = rf_rand3.score(X_train, Y_train)
for_record['best_random3']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random3']['test_score'] = rf_rand3.score(X_test, Y_test)


# In[141]:


print (for_record['best_random1']['test_score'])
print (for_record['best_random2']['test_score'])
print (for_record['best_random3']['test_score'])


# In[ ]:




