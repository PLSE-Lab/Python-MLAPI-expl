#!/usr/bin/env python
# coding: utf-8

# # Clean Manual Feature Engineering
# 
# The purpose of this notebook is to clean up the manual feature engineering I had scattered over several other kernels. We will implement the complete manual feature engineering and then test the results.
# 
# Update August 7: __After some modifications, this can now run in a kernel!__ The features themselves are available at https://www.kaggle.com/willkoehrsen/home-credit-manual-engineered-features under `clean_manual.csv`. The feature importances for these features in a gradient boosting model are also available at the same link with the name `fi_clean_manual.csv`. 
# 
# ### Roadmap
# 
# Our plan of action is as follows.We have to be very careful about memory usage in the kernels, which affects the order of operations:
# 
# 1. Define functions:
#     * `agg_numeric`
#     * `agg_categorical`
#     * `agg_child` 
#     * `agg_grandchild`
#  2. Add in domain knowledge features to `app`
#  3. Work through the `bureau` and `bureau_balance` data
#      * Add in hand built features
#      * Aggregate both using the appropriate functions
#      * Merge with `app` and delete the dataframes
# 4. Work through `previous`, `installments`, `cash`, and `credit`
#     * Add in hand built features
#     * Aggregate using the appropriate functions
#     * Merge with `app` and delete the dataframes
# 5. Modeling using a Gradient Boosting Machine
#     * Train model on training data using best hyperparameters from random search notebook
#     * Make predictions and submit
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df):
    print(f'Original size of data: {return_size(df)} gb.')
    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    print(f'New size of data: {return_size(df)} gb.')
    return df


# In[ ]:


# Read in the datasets and replace the anomalous values
app_train = pd.read_csv('../input/application_train.csv').replace({365243: np.nan})
app_test = pd.read_csv('../input/application_test.csv').replace({365243: np.nan})
bureau = pd.read_csv('../input/bureau.csv').replace({365243: np.nan})
bureau_balance = pd.read_csv('../input/bureau_balance.csv').replace({365243: np.nan})

app_test['TARGET'] = np.nan
app = app_train.append(app_test, ignore_index = True, sort = True)

app = convert_types(app)
bureau = convert_types(bureau)
bureau_balance = convert_types(bureau_balance)

import gc
gc.enable()
del app_train, app_test
gc.collect()


# # Numeric Aggregation Function
# 
# The following function aggregates all the numeric variables in a child dataframe at the parent level. That is, for each parent, gather together (group) all of their children, and calculate the aggregations statistics across the children. The function also removes any columns that share the exact same values (which might happen using `count`). 

# In[ ]:


def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.
    
    Parameters
    --------
        df (dataframe): 
            the child dataframe to calculate the statistics on
        parent_var (string): 
            the parent variable used for grouping and aggregating
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated by the `parent_var` for 
            all numeric columns. Each observation of the parent variable will have 
            one row in the dataframe with the parent variable as the index. 
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed. 
    
    """
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg


# # Categorical Aggregation Function
# 
# Much like the numerical aggregation function, the `agg_categorical` function works on a child dataframe to aggregate statistics at the parent level. This can work with any child of `app` and might even be extensible to other problems with only minor changes in syntax.

# In[ ]:


def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical


# # Combined Aggregation Function
# 
# We can put these steps together into a function that will handle a child dataframe. The function will take care of both the numeric and categorical variables and will return the result of merging the two dataframes. 

# In[ ]:


import gc

def agg_child(df, parent_var, df_name):
    """Aggregate a child dataframe for each observation of the parent."""
    
    # Numeric and then categorical
    df_agg = agg_numeric(df, parent_var, df_name)
    df_agg_cat = agg_categorical(df, parent_var, df_name)
    
    # Merge on the parent variable
    df_info = df_agg.merge(df_agg_cat, on = parent_var, how = 'outer')
    
    # Remove any columns with duplicate values
    _, idx = np.unique(df_info, axis = 1, return_index = True)
    df_info = df_info.iloc[:, idx]
    
    # memory management
    gc.enable()
    del df_agg, df_agg_cat
    gc.collect()
    
    return df_info


# This function can be applied to both `bureau` and `previous` because these are direct children of `app`. For the children of the children, we will need to take an additional aggregation step. 

# # Aggregate Grandchild Data Tables
# 
# Several of the tables (`bureau_balance, cash, credit_card`, and `installments`) are children of the child dataframes. In other words, these are grandchildren of the main `app` data table. To aggregate these tables, they must first be aggregated at the parent level (which is on a per loan basis) and then at the grandparent level (which is on the client basis). For example, in the `bureau_balance` dataframe, there is monthly information on the loans in `bureau`. To get this data into the `app` dataframe will first require grouping the monthly information for each loan and then grouping the loans for each client. 
# 
# Hopefully, the nomenclature does not get too confusing, but here's a rounddown:
# 
# * __grandchild__: the child of a child data table, for instance, `bureau_balance`. For every row in the child table, there can be multiple rows in the grandchild. 
# * __parent__: the parent table of the grandchild that links the grandchild to the grandparent. For example, the `bureau` dataframe is the parent of the `bureau_balance` dataframe in this situation. `bureau` is in turn the child of the `app` dataframe. `bureau_balance` can be connected to `app` through `bureau`.
# * __grandparent__: the parent of the parent of the grandchild, in this problem the `app` dataframe. The end goal is to aggregate the information in the grandchild into the grandparent. This will be done in two stages: first at the parent (loan) level and then at the grandparent (client) level
# * __parent variable__: the variable linking the grandchild to the parent. For the `bureau` and `bureau_balance` data this is `SK_ID_BUREAU` which uniquely identifies each previous loan
# * __grandparent variable__: the variable linking the parent to the grandparent. This is `SK_ID_CURR` which uniquely identifies each client in `app`.
# 
# ### Aggregating Grandchildren Function
# 
# We can take the individual steps required for aggregating a grandchild dataframe at the grandparent level in a function. These are:
# 
# 1. Aggregate the numeric variables at the parent (the loan, `SK_ID_BUREAU` or `SK_ID_PREV`) level.
# 2. Merge with the parent of the grandchild to get the grandparent variable in the data (for example `SK_ID_CURR`)
# 3. Aggregate the numeric variables at the grandparent (the client, `SK_ID_CURR`) level. 
# 4. Aggregate the categorical variables at the parent level.
# 5. Merge the aggregated data with the parent to get the grandparent variable
# 6. Aggregate the categorical variables at the grandparent level
# 7. Merge the numeric and categorical dataframes on the grandparent varible
# 8. Remove the columns with all duplicated values.
# 9. The resulting dataframe should now have one row for every grandparent (client) observation
# 10. Merge with the main dataframe (`app`) on the grandparent variable (`SK_ID_CURR`). 
# 
# This function can be applied to __all 4 grandchildren__ without the need for hard-coding in specific variables. 

# In[ ]:


def agg_grandchild(df, parent_df, parent_var, grandparent_var, df_name):
    """
    Aggregate a grandchild dataframe at the grandparent level.
    
    Parameters
    --------
        df : dataframe
            Data with each row representing one observation
            
        parent_df : dataframe
            Parent table of df that must have the parent_var and 
            the grandparent_var. Used only to get the grandparent_var into
            the dataframe after aggregations
            
        parent_var : string
            Variable representing each unique observation in the parent.
            For example, `SK_ID_BUREAU` or `SK_ID_PREV`
            
        grandparent_var : string
            Variable representing each unique observation in the grandparent.
            For example, `SK_ID_CURR`. 
            
        df_name : string
            String for renaming the resulting columns.
            The columns are name with the `df_name` and with the 
            statistic calculated in the column
    
    Return
    --------
        df_info : dataframe
            A dataframe with one row for each observation of the grandparent variable.
            The grandparent variable forms the index, and the resulting dataframe
            can be merged with the grandparent to be used for training/testing. 
            Columns with all duplicate values are removed from the dataframe before returning.
    
    """
    
    # set the parent_var as the index of the parent_df for faster merges
    parent_df = parent_df[[parent_var, grandparent_var]].copy().set_index(parent_var)
    
    # Aggregate the numeric variables at the parent level
    df_agg = agg_numeric(df, parent_var, '%s_LOAN' % df_name)
    
    # Merge to get the grandparent variable in the data
    df_agg = df_agg.merge(parent_df, 
                          on = parent_var, how = 'left')
    
    # Aggregate the numeric variables at the grandparent level
    df_agg_client = agg_numeric(df_agg, grandparent_var, '%s_CLIENT' % df_name)
    
    # Can only apply one-hot encoding to categorical variables
    if any(df.dtypes == 'category'):
    
        # Aggregate the categorical variables at the parent level
        df_agg_cat = agg_categorical(df, parent_var, '%s_LOAN' % df_name)
        df_agg_cat = df_agg_cat.merge(parent_df,
                                      on = parent_var, how = 'left')

        # Aggregate the categorical variables at the grandparent level
        df_agg_cat_client = agg_numeric(df_agg_cat, grandparent_var, '%s_CLIENT' % df_name)
        df_info = df_agg_client.merge(df_agg_cat_client, on = grandparent_var, how = 'outer')
        
        gc.enable()
        del df_agg, df_agg_client, df_agg_cat, df_agg_cat_client
        gc.collect()
    
    # If there are no categorical variables, then we only need the numeric aggregations
    else:
        df_info = df_agg_client.copy()
    
        gc.enable()
        del df_agg, df_agg_client
        gc.collect()
    
    # Drop the columns with all duplicated values
    _, idx = np.unique(df_info, axis = 1, return_index=True)
    df_info = df_info.iloc[:, idx]
    
    return df_info


# # Putting it Together
# 
# Now that we have the individual pieces of semi-automated feature engineering, we need to put them together. There are two functions that can handle the children and the grandchildren data tables:
# 
# 1. `agg_child(df, parent_var, df_name)`: aggregate the numeric and categorical variables of a child dataframe at the parent level. For example, the `previous` dataframe is a child of the `app` dataframe that must be aggregated for each client. 
# 2. `agg_grandchild(df, parent_df, parent_var, grandparent_var, df_name)`: aggregate the numeric and categorical variables of a grandchild dataframe at the grandparent level. For example, the `bureau_balance` dataframe is the grandchild of the `app` dataframe with `bureau` as the parent. 
# 
# For each of the children dataframes of `app`, (`previous` and `bureau`), we will use the first function and merge the result into the `app` on the parent variable, `SK_ID_CURR`. For the four grandchild dataframes, we will use the second function, which returns a single dataframe that can then be merged into app on `SK_ID_CURR`. 

# ## Hand-Built Features
# 
# Along the way, we will add in hand-built features to the datasets. These have come from my own ideas (probably not very optimal) and from the community.
# 
# First we will add in "domain knowledge" features to the `app` dataframe. These were developed based on work done in other kernels (both from the community and my own work)

# In[ ]:


# Add domain features to base dataframe
app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT'] 
app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis = 1)
app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis = 1)


# ### Hand-Built Features for other Dataframes
# 
# We can also add in hand built features for the other dataframes. Since these are not the main dataframe, these features will end up being aggregated in different ways. These will be added as we go through the tables.

# #### Aggregate the bureau data
# 
# First add the loan rate for previous loans at other institutions.

# In[ ]:


bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']


# In[ ]:


bureau_info = agg_child(bureau, 'SK_ID_CURR', 'BUREAU')
bureau_info.head()


# In[ ]:


bureau_info.shape


# #### Aggregate the bureau balance
# 
# Now we turn to the `bureau_balance` dataframe. We will make a column indicating whether a loan was past due for the month or whether the payment was on time.

# In[ ]:


bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'


# In[ ]:


bureau_balance_info = agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
del bureau_balance, bureau
bureau_balance_info.head()


# In[ ]:


bureau_balance_info.shape


# ## Merge with the main dataframe
# 
# The individual dataframes can all be merged into the main `app` dataframe. Merging is much quicker if done on any index, so it's good practice to first set the index to the variable on which we will merge. In each case, we use a `left` join so that all the observations in `app` are kept even if they are not present in the other dataframes (which occurs because not every client has previous records at Home Bureau or other credit institutions). After each step of mergning, we remove the dataframe from memory in order to hopefully let the kernel continue to run.
# 
# The final result is one dataframe with a single row for each client that can be used for training a machine learning model. 

# In[ ]:


app = app.set_index('SK_ID_CURR')
app = app.merge(bureau_info, on = 'SK_ID_CURR', how = 'left')
del bureau_info
app.shape


# In[ ]:


app = app.merge(bureau_balance_info, on = 'SK_ID_CURR', how = 'left')
del bureau_balance_info
app.shape


# #### Aggregate previous loans at Home Credit
# 
# We will add in two domain features, first the loan rate and then the difference between the amount applied for and the amount awarded.

# In[ ]:


previous = pd.read_csv('../input/previous_application.csv').replace({365243: np.nan})
previous = convert_types(previous)
previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']


# `AMT_DIFFERENCE` is the difference between what was given to the client and what the client requested on previous loans at Home Credit.

# In[ ]:


previous_info = agg_child(previous, 'SK_ID_CURR', 'PREVIOUS')
previous_info.shape


# In[ ]:


app = app.merge(previous_info, on = 'SK_ID_CURR', how = 'left')
del previous_info
app.shape


# #### Aggregate Installments Data
# 
# The installments table has each installment (payment) for previous loans at Home Credit. We can create a column indicating whether or not a loan was late.

# In[ ]:


installments = pd.read_csv('../input/installments_payments.csv').replace({365243: np.nan})
installments = convert_types(installments)
installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']


# `LOW_PAYMENT` represents a payment that was less than the prescribed amount. 

# In[ ]:


installments_info = agg_grandchild(installments, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'IN')
del installments
installments_info.shape


# In[ ]:


app = app.merge(installments_info, on = 'SK_ID_CURR', how = 'left')
del installments_info
app.shape


# #### Aggregate Cash previous loans
# 
# The next dataframe is the `cash` which has monthly information on previous cash loans at Home Credit. We can create a column indicating if the loan was overdue for the month. 

# In[ ]:


cash = pd.read_csv('../input/POS_CASH_balance.csv').replace({365243: np.nan})
cash = convert_types(cash)
cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']


# `INSTALLMENTS_PAID` is meant to represent the number of already paid (or I guess missed) installments by subtracting the future installments from the total installments.

# In[ ]:


cash_info = agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
del cash
cash_info.shape


# In[ ]:


app = app.merge(cash_info, on = 'SK_ID_CURR', how = 'left')
del cash_info
app.shape


# #### Aggregate Credit previous loans
# 
# The last dataframe is `credit` which has previous credit card loans at Home Credit. We can make a column indicating whether the balance is greater than the credit limit, a column showing whether or not the balance was cleared (equal to 0), whether or not the payment was below the prescribed amount, and whether or not the payment was behind. Then we aggregate as with the other grandchildren.

# In[ ]:


credit = pd.read_csv('../input/credit_card_balance.csv').replace({365243: np.nan})
credit = convert_types(credit)
credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
credit['LATE'] = credit['SK_DPD'] > 0.0


# In[ ]:


credit_info = agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
del credit, previous
credit_info.shape


# In[ ]:


gc.collect()
gc.enable()


# __This is usually the point at which the kernel fails.__ To try and alleviate the problem, I have added a pause of 10 minutes.

# In[ ]:


import time
time.sleep(600)
app = app.merge(credit_info, on = 'SK_ID_CURR', how = 'left')
del credit_info
app.shape


# In[ ]:


print('After manual feature engineering, there are {} features.'.format(app.shape[1] - 2))


# In[ ]:


gc.enable()
gc.collect()


# In[ ]:


print(f'Final size of data {return_size(app)}')


# __Update August 7__: The kernel can now run!

# In[ ]:


# Check for columns with duplicated values
# _, idx = np.unique(app, axis = 1, return_index = True)
# print('There are {} columns with all duplicated values.'.format(app.shape[1] - len(idx)))


# In[ ]:


app.to_csv('clean_manual_features.csv', chunksize = 100)


# # Modeling
# 
# After all the hard work, now we get to test our features! We will use a model with the hyperparameters from random search that are documented in another notebook. 
# 
# The final model scores __0.792__ when uploaded to the competition.

# In[ ]:


app.reset_index(inplace = True)
train, test = app[app['TARGET'].notnull()].copy(), app[app['TARGET'].isnull()].copy()
gc.enable()
del app
gc.collect()


# In[ ]:


import lightgbm as lgb

params = {'is_unbalance': True, 
              'n_estimators': 2673, 
              'num_leaves': 77, 
              'learning_rate': 0.00764, 
              'min_child_samples': 460, 
              'boosting_type': 'gbdt', 
              'subsample_for_bin': 240000, 
              'reg_lambda': 0.20, 
              'reg_alpha': 0.88, 
              'subsample': 0.95, 
              'colsample_bytree': 0.7}


# In[ ]:


train_labels = np.array(train.pop('TARGET')).reshape((-1, ))

test_ids = list(test.pop('SK_ID_CURR'))
test = test.drop(columns = ['TARGET'])
train_ids = train.pop('SK_ID_CURR')

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)


# In[ ]:


model = lgb.LGBMClassifier(**params)
model.fit(train, train_labels)


# In[ ]:


preds = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_ids,
                           'TARGET': preds})

submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype(int)
submission['TARGET'] = submission['TARGET'].astype(float)
submission.to_csv('submission_manual.csv', index = False)


# ## Feature Importances
# 
# Now we can see if all that time was worth it! In the code below, we find the most important features and show them in a plot and dataframe.

# In[ ]:


features = list(train.columns)
fi = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_})


# In[ ]:


def plot_feature_importances(df, n = 15, threshold = None):
    """
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be "feature" and "importance"
    
    n : int, default = 15
        Number of most important features to plot
    
    threshold : float, default = None
        Threshold for cumulative importance plot. If not provided, no plot is made
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    Note
    --------
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
    
    """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', edgecolor = 'k', figsize = (12, 8),
                            legend = False)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.2, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 100 * threshold))
    
    return df


# In[ ]:


norm_fi = plot_feature_importances(fi, 25)
norm_fi.head(25)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    params = {'is_unbalance': True, 
              'n_estimators': 2673, 
              'num_leaves': 77, 
              'learning_rate': 0.00764, 
              'min_child_samples': 460, 
              'boosting_type': 'gbdt', 
              'subsample_for_bin': 240000, 
              'reg_lambda': 0.20, 
              'reg_alpha': 0.88, 
              'subsample': 0.95, 
              'colsample_bytree': 0.7}
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(**params, 
                                   n_estimators=10000, objective = 'binary', 
                                   n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


# In[ ]:


train['TARGET'] = list(train_labels)
train['SK_ID_CURR'] = list(train_ids)
test['SK_ID_CURR'] = list(test_ids)

submission, feature_importances, metrics = model(train, test)


# In[ ]:


submission.to_csv('submission_clean_manual2.csv')
norm_fi = plot_feature_importances(feature_importances)


# # Conclusions
# 
# This code is a little too much to run in the Kaggle kernels. However, the features themselves are available at https://www.kaggle.com/willkoehrsen/home-credit-manual-engineered-features under `clean_manual.csv`. The feature importances for these features in a gradient boosting model are also available at the same link with the name `fi_clean_manual.csv`. 
# 
# This notebook is meant to serve as a clean version of the manual feature engineering I had scattered across several other notebooks. We were able to build a complete set of __ features that scored 0.792 on the public leaderboard__. Further hyperparameter tuning might improve the performance. For additional feature engineering, we will probably want to turn to more technical operations such as treating this as a time-series problem. Since we have relative time information (relative to the current loan at Home Credit), it's possible to find the most recent information and also trends over time. These can be useful because changes in behavior might inform us as to whether or not a client will be able to repay a loan! 
# 
# Thanks for reading and as always, I welcome feedback and constructive criticism. I'll see you in the next notebook.
# 
# Best,
# 
# Will

# In[ ]:




