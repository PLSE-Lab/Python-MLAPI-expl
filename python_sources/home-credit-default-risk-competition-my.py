#!/usr/bin/env python
# coding: utf-8

# ### Manual Feature Engineering

# In[ ]:


# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')


# #### Counts of a client's previous loans

# In[ ]:


# Read in bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()


# In[ ]:


# groupby the client id (SK_ID_CURR), count the # of previous loans, and rename the col.
# every client has a # of prev. loans and we will count it and add them on main app DF
previous_loans_count = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU' : 'previous_loan_counts'})
previous_loans_count.head()


# In[ ]:


# join to the main df 
train = pd.read_csv('../input/application_train.csv')
train = train.merge(previous_loans_count, on = 'SK_ID_CURR', how='left')

# fill the missing values with 0
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()


# #### Assessing Usefulness of New Variable with r value

# ##### Kernel Density Estimate Plots

# In[ ]:


#  Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    
    # calc the correlation coeff. between the new var and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # calc the median for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # plot the dist for target == 0 and 1
    sns.kdeplot(df.ix[df['TARGET'] == 0 ,var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1 ,var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print corr
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)


# #### we can test this function using the EXT_SOURCE_3

# In[ ]:


kde_target('EXT_SOURCE_3', train)


# Now for the new variable we just made, the number of previous loans at other institutions.

# In[ ]:


kde_target('previous_loan_counts', train)


# From this it's difficult to tell if this variable will be important. The correlation coefficient is extremely weak and there is almost no noticeable difference in the distributions. 
# 
# Let's move on to make a few more variables from the bureau dataframe. We will take the mean, min, and max of every numeric column in the bureau dataframe.

# ### Aggregating Numeric Columns

# In[ ]:


# Group by the client id, calculate aggregation statistics
bureau_agg = bureau.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index=False).agg(['count', 'mean', 'max','min', 'sum']).reset_index()
bureau_agg.head()


# We need to create new names for each of these columns. The following code makes new names by appending the stat to the name. Here we have to deal with the fact that the dataframe has a multi-level index. I find these confusing and hard to work with, so I try to reduce to a single level index as quickly as possible.

# In[ ]:


# list col names
columns = ['SK_ID_CURR']

# iterate through the variable names
for var in bureau_agg.columns.levels[0]:
    # skip the id name
    if var != 'SK_ID_CURR':
        
        # iter through the stat names
        for stat in bureau_agg.columns.levels[1][:-1]:
            columns.append('bureau_%s_%s' %(var, stat))


# In[ ]:


# Assign the list of columns names as the dataframe column names
bureau_agg.columns = columns
bureau_agg.head()


# Now we simply merge with the training data

# In[ ]:


# Merge with the training data
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
train.head()


# ### Correlations of Aggregated Values with Target
# 
# We can calculate the correlation of all new values with the target. Again, we can use these as an approximation of the variables which may be important for modeling. 

# In[ ]:


# List of new correlations
new_corrs = []

# Iterate through the columns 
for col in columns:
    # Calculate correlation with the target
    corr = train['TARGET'].corr(train[col])
    
    # Append the list as a tuple

    new_corrs.append((col, corr))


# sort the correlations by the magnitude (absolute value)

# In[ ]:


#  Make sure to reverse to put the largest values at the front of list
new_corrs = sorted(new_corrs, key = lambda x:abs(x[1]), reverse = True)
new_corrs[:15]


# None of the new variables have a significant correlation with the TARGET. We can look at the KDE plot of the highest correlated variable, bureau_DAYS_CREDIT_mean, with the target in in terms of absolute magnitude correlation. 

# In[ ]:


kde_target('bureau_DAYS_CREDIT_mean', train)


# The definition of this column is: "How many days before current application did client apply for Credit Bureau credit". My interpretation is this is the number of days that the previous loan was applied for before the application for a loan at Home Credit. Therefore, a larger negative number indicates the loan was further before the current loan application. We see an extremely weak positive relationship between the average of this variable and the target meaning that clients who applied for loans further in the past potentially are more likely to repay loans at Home Credit. With a correlation this weak though, it is just as likely to be noise as a signal. 

# The definition of this column is: "How many days before current application did client apply for Credit Bureau credit". My interpretation is this is the number of days that the previous loan was applied for before the application for a loan at Home Credit. Therefore, a larger negative number indicates the loan was further before the current loan application. We see an extremely weak positive relationship between the average of this variable and the target meaning that clients who applied for loans further in the past potentially are more likely to repay loans at Home Credit. With a correlation this weak though, it is just as likely to be noise as a signal. 
# 
# #### The Multiple Comparisons Problem
# 
# When we have lots of variables, we expect some of them to be correlated just by pure chance, a [problem known as multiple comparisons](https://towardsdatascience.com/the-multiple-comparisons-problem-e5573e8b9578). We can make hundreds of features, and some will turn out to be corelated with the target simply because of random noise in the data. Then, when our model trains, it may overfit to these variables because it thinks they have a relationship with the target in the training set, but this does not necessarily generalize to the test set. There are many considerations that we have to take into account when making features! 

# ## Function for Numeric Aggregations
# Let's encapsulate all of the previous work into a function

# In[ ]:


def agg_numeric(df, group_var, df_name):
    """ Aggregation the numeric values in a dataframe. thi can be
    used to create features for each instance of the grouping variable.
    
    Parameters
    ---------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to griup df
        df_name (string):
            the variable used to rename the col
            
        return
        ----------
            agg (dataframe):
                a dataframe with the statistics aggregated for 
                all numeric columns. Each instance of the grouping variable will have 
                the statistics (mean, min, max, sum; currently supported) calculated. 
                The columns are also renamed to keep track of features created.
    """
    
    # remove id variable other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    
    # Need to create new column names
    columns = [group_var]
    
    # iter through var names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg


# In[ ]:


bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg_new.head()


# To make sure the function worked as intended, we should compare with the aggregated dataframe we constructed by hand. 

# In[ ]:


bureau_agg.head()


# ### Correlation Function
# 
# Before we move on, we can also make the code to calculate correlations with the target into a function.

# In[ ]:


# Function to calculate correlations with the target for a dataframe
def target_corrs(df):
    
    # list of corr
    corrs = []
    
    # Iterate through the columns 
    for col in df.columns:
        print(col)
        # skip the target col
        if col != 'TARGET' :
            corr = df['TARGET'].corr(df[col])
            
            # Append the list as a tuple
            corrs.append((col, corr))
            
    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs


# First we one-hot encode a dataframe with only the categorical columns (`dtype == 'object'`).

# In[ ]:


categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical.head()


# In[ ]:


categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])
categorical_grouped.head()


# The `sum` columns represent the count of that category for the associated client and the `mean` represents the normalized count. One-hot encoding makes the process of calculating these figures very easy!

# In[ ]:


categorical_grouped.columns.levels[0][:10]


# In[ ]:


categorical_grouped.columns.levels[1]


# In[ ]:


group_var = 'SK_ID_CURR'

# Need to create new column names
columns = []

# Iterate through the variables names
for var in categorical_grouped.columns.levels[0]:
    # Skip the grouping variable
    if var != group_var:
        # Iterate through the stat names
        for stat in ['count', 'count_norm']:
            # Make a new column name for the variable and stat
            columns.append('%s_%s' % (var, stat))

#  Rename the columns
categorical_grouped.columns = columns

categorical_grouped.head()


# We can merge this dataframe into the training data.

# In[ ]:


train = train.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
train.head()


# In[ ]:


train.shape


# In[ ]:


train.iloc[:10, 123:]


# ### Function to Handle Categorical Variables
# 

# In[ ]:


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical


# In[ ]:


bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()


# In[ ]:


bureau_counts.index


# ### Applying Operations to another dataframe

# We will now turn to the bureau balance dataframe. This dataframe has monthly information about each client's previous loan(s) with other financial institutions. Instead of grouping this dataframe by the `SK_ID_CURR` which is the client id, we will first group the dataframe by the `SK_ID_BUREAU` which is the id of the previous loan. This will give us one row of the dataframe for each loan. Then, we can group by the `SK_ID_CURR` and calculate the aggregations across the loans of each client. The final result will be a dataframe with one row for each client, with stats calculated for their loans.

# In[ ]:


# Read in bureau balance
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()


# First, we can calculate the value counts of each status for each loan. Fortunately, we already have a function that does this for us! 

# In[ ]:


# Counts of each type of status for each previous loan
bureau_balance_counts = count_categorical(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
bureau_balance_counts.head()


# In[ ]:


# Calculate value count statistics for each `SK_ID_CURR` 
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()


# In[ ]:


# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

bureau_by_loan.head()


# In[ ]:


bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
bureau_balance_by_client.head()


# # Putting the Functions Together

# In[ ]:


# Free up memory by deleting old objects
import gc
gc.enable()
del  bureau_agg_new, bureau_balance_agg, bureau_balance_counts, bureau_by_loan, bureau_balance_by_client, bureau_counts
gc.collect()


# In[ ]:


# Read in new copies of all the dataframes
train = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')


# ### Counts of Bureau Dataframe

# In[ ]:


bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()


# In[ ]:


# bureau_count


# ### Aggregated Stats of Bureau Dataframe

# In[ ]:


bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()


# ### Value counts of Bureau Balance dataframe by loan

# In[ ]:


bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()


# ### Aggregated stats of Bureau Balance dataframe by loan

# In[ ]:


bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()


# ### Aggregated Stats of Bureau Balance by Client

# In[ ]:


# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

# Aggregate the stats for each client
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')


# ### Insert Computed Features into Training Data

# In[ ]:


original_features = list(train.columns)
print('original # of Features: ', len(original_features))


# In[ ]:


# Merge with the value counts of bureau
train = train.merge(bureau_counts, on = 'SK_ID_CURR' ,how = 'left')

# Merge with the stats of bureau
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# Merge with the monthly information grouped by client
train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


new_features = list(train.columns)
print('Number of features using previous loans from other institutions data: ', len(new_features))


# ### Feature Selection

# Feature selection is the process of removing variables to help our model to learn and generalize better to the testing set. The objective is to remove useless/redundant variables while preserving those that are useful. There are a number of tools we can use for this process, but in this notebook we will stick to removing columns with a high percentage of missing values and variables that have a high correlation with one another. Later we can look at using the feature importances returned from models such as the Gradient Boosting Machine or Random Forest to perform feature selection.
# 

# ## Missing Values
# 
# An important consideration is the missing values in the dataframe. Columns with too many missing values might have to be dropped. 

# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


missing_train = missing_values_table(train)
missing_train.head(10)


# we will remove any columns in either the training or the testing data that have greater than 90% missing values.

# In[ ]:


missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])
len(missing_train_vars)


# ## Calculate Information for Testing Data

# In[ ]:


# Read in the test dataframe
test = pd.read_csv('../input/application_test.csv')

# Merge with the value counts of bureau
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# Merge with the stats of bureau
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# Merge with the value counts of bureau balance
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


print('Shape of Testing Data: ', test.shape)


# We need to align the testing and training dataframes, which means matching up the columns so they have the exact same columns. This shouldn't be an issue here, but when we one-hot encode variables, we need to align the dataframes to make sure they have the same columns.

# In[ ]:


train_labels = train['TARGET']

# Align the dataframes, this will remove the 'TARGET' column
train, test = train.align(test, join = 'inner', axis = 1)

train['TARGET'] = train_labels


# In[ ]:


print('Training Data Shape: ', train.shape)
print('Testing Data Shape: ', test.shape)


# The dataframes now have the same columns (with the exception of the `TARGET` column in the training data). This means we can use them in a machine learning model which needs to see the same columns in both the training and testing dataframes.
# 
# Let's now look at the percentage of missing values in the testing data so we can figure out the columns that should be dropped.

# In[ ]:


missing_test = missing_values_table(test)
missing_test.head(10)


# In[ ]:


missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
len(missing_test_vars)


# In[ ]:


missing_columns = list(set(missing_test_vars + missing_train_vars))
print('There are %d columns with more than 65%% missing in either the training or testing data.' % len(missing_columns))


# In[ ]:


# Drop the missing columns
train = train.drop(columns = missing_columns)
test = test.drop(columns = missing_columns)


# In[ ]:


train.to_csv('train_bureau_raw.csv', index = False)
test.to_csv('test_bureau_raw.csv', index = False)


# In[ ]:




