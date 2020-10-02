#!/usr/bin/env python
# coding: utf-8

# ## In this kernel we use xgboost to predict house prices.  
# 
# There are 5 parts to this kernel:
# 
# 1. Import the libraries and data 
# 1. Prepare the data 
# 1. Train the xgboost model 
# 1. Understand the model 
# 1. Create predictions

# ## 1a. Import the libraries we are going to use
# 
# We start by importing the various libraries we are going to use.

# In[ ]:


import numpy as np # mathematical library including linear algebra
import pandas as pd #data processing and CSV file input / output

import xgboost as xgb # this is the extreme gradient boosting library
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing 
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1b. Next we import the data
# 
# Now we read in the training and test data. We are using Pandas "read_csv" function for this.

# In[ ]:


df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])


# ##  2. Data preparation
# 
# - Create a vector containing the id's for our predictions
#  - Create a vector of the target variables in the training set
#  - Create joint train and test set to make data wrangling quicker and consistent on train and test
#  - Removing the id (could it be a useful source of leakage?)
#  - Feature engineering
#      - Convert the date into a number (of days since some point)
#      - Deal with categorical variables
#      - Deal with missing values

# In[ ]:



# Create a vector containing the id's for our predictions
id_test = df_test.id

#Create a vector of the target variables in the training set
# Transform target variable so that loss function is correct (ie we use RMSE on transormed to get RMLSE)
# ylog1p_train_whole will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
ylog1p_train_all = np.log1p(df_train['price_doc'].values)
df_train = df_train.drop(["price_doc"], axis=1)

# Create joint train and test set to make data wrangling quicker and consistent on train and test
df_train["trainOrTest"] = "train"
df_test["trainOrTest"] = "test"
num_train = len(df_train)

df_all = pd.concat([df_train, df_test])
del df_train
del df_test

# Removing the id (could it be a useful source of leakage?)
df_all = df_all.drop("id", axis=1)


# ### Feature engineering

# In[ ]:


# Convert the date into a number (of days since some point)
fromDate = min(df_all['timestamp'])
df_all['timedelta'] = (df_all['timestamp'] - fromDate).dt.days.astype(int)
print(df_all[['timestamp', 'timedelta']].head())


# In[ ]:


# Add month-year count - i.e. how many sales in the month 
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)


# ### Encoding categorical features
# We will take a naive approach and assign a numeric value to each categorical feature in our training and test sets. 
# Sklearn's preprocessing unit has a tool called LabelEncoder() which can do just that for us. 

# In[ ]:


for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))


# In[ ]:


# Alternative using the rather nice .select_dtypes
# df_numeric = df_all.select_dtypes(exclude=['object'])
# df_obj = df_all.select_dtypes(include=['object']).copy()

# for c in df_obj:
#    df_obj[c] = pd.factorize(df_obj[c])[0]

# df_all = pd.concat([df_numeric, df_obj], axis=1)


# ### Addressing problems with NaN in the data
# 
# As we saw from our EDA there were quite a lot of NaN in the data. Our model won't know what to do with these so we need to replace them with something sensible.
# 
# There are quite a few options we can use - the mean, median, most_frequent, or a numeric value like 0. Playing with these will give different results, for now I have it set to use the mean.
# 
# For the random forest we used the median.  Here we just replace na with -1.  Which is better?  Why?

# In[ ]:


# Fill missing values with  -1.  
df_all.fillna(-1, inplace = True)


# In[ ]:


# Convert to numpy values
X_all = df_all.values
print(X_all.shape)

# Create a validation set, with last 20% of data
num_val = int(num_train * 0.2)

#X_train_all  = X_all[:num_train]
X_train      = X_all[:num_train-num_val]
X_val        = X_all[num_train-num_val:num_train]
ylog1p_train = ylog1p_train_all[:-num_val]
ylog1p_val   = ylog1p_train_all[-num_val:]

X_test = X_all[num_train:]

df_columns = df_all.columns

del df_all

#print('X_train_all shape is', X_train_all.shape)
print('X_train shape is',     X_train.shape)
print('y_train shape is',     ylog1p_train.shape)
print('X_val shape is',       X_val.shape)
print('y_val shape is',       ylog1p_val.shape)
print('X_test shape is',      X_test.shape)


# Format the train and test sets we modified above for use in xgboost 
# (Dmatrix is the format required by the xgboost library)

# In[ ]:


##dtrain_all = xgb.DMatrix(X_train_all, ylog1p_train_all, feature_names = df_columns)
#dtrain     = xgb.DMatrix(X_train,     ylog1p_train,     feature_names = df_columns)
#dval       = xgb.DMatrix(X_val,       ylog1p_val,       feature_names = df_columns)
dtest      = xgb.DMatrix(X_test,                        feature_names = df_columns)


# ## 3. Train the xgboost model

# ####  Parameters for xgboost are as follows:
# 
# ### Booster parameters 
# These parameters are used to optimise the algorithm in terms of both accuracy and performance.
# 
# **eta / learning_rate:  This is similar to the learning rate (alpha) in gradient descent. 
# Makes the model more robust by shrinking the weights on each step. Typical final values range from 0.01-0.2.  
# 
# **max_depth:  It sets the maximum depth of a tree and is used to control over-fitting as higher depth allows the model to learn relations very specific to a particular sample. We tune it using cross-validation. Typical values range from 3-10
# 
# **subsample:  It denotes the fraction of obeservations to be randomly samples for each tree. Lower values make the algorithm conservative and prevent overfitting but too small and we may get under-fitting. Typical values range from 0.5-1 
# 
# **colsample_bytree:  It denotes the fraction of columns to be randomly samples for each tree. Typical values range from 0.5-1  
# 
# ### Learning Task Parameters
# These parameters are used to define the optimisation metric to be calculated at each step.
# 
# **'eval_metric': 'rmse'** sets our evaluation metric to root mean squared error
#     This  evaluation metric used to score submissions in this competition is the log root mean squared error, however this option is not available to us within xgboost so this is the closest match.
# 
# ### General parameters
# **booster** - left at default by not setting it, which means we are using a tree-based model. It can also be set to use linear models.
# 
# **silent: 1** - this defaults to 0 and is a binary switch. When set to 0 running messages will be printed which may help to understand the model. It can be set to 1 to suppress running messages.

# ### 3a. Instantiate the booster with the parameters we have chosen

# In[ ]:


# Choose values for the key parameters - keep the number of estimators low for now - not more than 200

model = xgb.XGBRegressor(    objective = 'reg:linear'
                           , n_estimators =  
                           , max_depth = 5
                           # , min_child_weight = min_child_weight
                           , subsample = 1.0
                           , colsample_bytree = 
                           , learning_rate = 
                           , silent = 1)


# ## 3b. Train the booster on our data
# 
# We do this in two stages.  First using partitioning the training data into training and validation data to find how many boosting rounds to carry out and then using all of the training data for the given number of rounds.

# In[ ]:


eval_set  = [( X_train, ylog1p_train), ( X_val, ylog1p_val)]

model.fit(X = X_train, 
          y = ylog1p_train,
          eval_set = eval_set, 
          eval_metric = "rmse", 
          early_stopping_rounds = 30,
          verbose = True)


# In[ ]:


num_boost_round = model.best_iteration 
num_boost_round

# Is num_boost_rounds one less than the n_estimators you chose above?  If it is 
# what does this tell you?  What should you do about it?


# In[ ]:


# Fill eta below with whatever you used for learning_rate above.
# Likewise for colsample_bytree

# Different syntax used here than above, due to issues with xgboost package (we can't get 
# variable importance the other way)

xgb_params = {
    'eta': ,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': ,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}

dtrain_all = xgb.DMatrix(np.vstack((X_train, X_val)), ylog1p_train_all, feature_names = df_columns)
model = xgb.train(xgb_params, dtrain_all, num_boost_round = num_boost_round)


# ## 4a. Variable importance

# In[ ]:


# Create a dataframe of the variable importances
dict_varImp = model.get_score(importance_type = 'weight')
df_ = pd.DataFrame(dict_varImp, index = ['varImp']).transpose().reset_index()
df_.columns = ['feature', 'fscore']


# In[ ]:


# Plot the relative importance of the top 10 features
df_['fscore'] = df_['fscore'] / df_['fscore'].max()
df_.sort_values('fscore', ascending = False, inplace = True)
df_ = df_[0:10]
df_.sort_values('fscore', ascending = True, inplace = True)
df_.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('xgboost feature importance', fontsize = 24)
plt.xlabel('')
plt.ylabel('')
plt.xticks([], [])
plt.yticks(fontsize=20)
plt.show()
#plt.gcf().savefig('feature_importance_xgb.png')


# In[ ]:


### 4b. Partial dependence plots


# In[ ]:


# from https://xiaoxiaowang87.github.io/monotonicity_constraint/
def partial_dependency(bst, X, y, feature_ids = [], f_id = -1):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """

    X_temp = X.copy()

    grid = np.linspace(np.percentile(X_temp[:, f_id], 0.1),
                       np.percentile(X_temp[:, f_id], 99.5),
                       50)
    y_pred = np.zeros(len(grid))

    if len(feature_ids) == 0 or f_id == -1:
        print ('Input error!')
        return
    else:
        for i, val in enumerate(grid):

            X_temp[:, f_id] = val
            data = xgb.DMatrix(X_temp, feature_names = df_columns)

            y_pred[i] = np.average(bst.predict(data))

    return grid, y_pred


# In[ ]:


lst_f = ['full_sq', 'timedelta', 'floor']
for f in lst_f:
    f_id = df_columns.tolist().index(f)


    feature_ids = range(X_train.shape[1])

    grid, y_pred = partial_dependency(model,
                                      X_train,
                                      ylog1p_train,
                                      feature_ids = feature_ids,
                                      f_id = f_id
                                      )

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.subplots_adjust(left = 0.17, right = 0.94, bottom = 0.15, top = 0.9)

    ax.plot(grid, y_pred, '-', color = 'red', linewidth = 2.5, label='fit')
    ax.plot(X_train[:, f_id], ylog1p_train, 'o', color = 'grey', alpha = 0.01)

    ax.set_xlim(min(grid), max(grid))
    ax.set_ylim(0.95 * min(y_pred), 1.05 * max(y_pred))

    ax.set_xlabel(f, fontsize = 24)
    ax.set_ylabel('Partial Dependence', fontsize = 24)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()


# In[ ]:


## 5. Create the predictions


# In[ ]:


# Create the predictions
ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1


# ### Output the data to CSV for submission

# In[ ]:


output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
output.to_csv('xgb_1.csv', index=False)

