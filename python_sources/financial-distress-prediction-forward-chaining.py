#!/usr/bin/env python
# coding: utf-8

# # Financial Distress Prediction - Forward Chaining
# 
# Contrary to Cross-Sectional Data, Longitudinal Data has a time-component that implicitly orders each row. Therefore, it is not a good idea to use K-Fold cross validation to test model performance. The right approach is to use [Forward Chaining](https://robjhyndman.com/hyndsight/tscv/).
# 
# Python's scikit-learn module has a [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) function that can help run Forward Chaining. However, the function makes an assumption that the entire DataFrame is to be treated as one entity. Whereas, it is possible that the entire DataFrame actually be composed of groups of data (row-wise) each of which group should be treated to Forward Chaining separately. The Financial Distress Prediction Data Set is exactly this type of a data set.
# 
# Let me demonstrate what I mean:

# In[ ]:


# Load necessary modules
import numpy as np 
import pandas as pd
import os
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
print(os.listdir("../input"))


# In[ ]:


# Load the Data
# Read data frame
df = pd.read_csv('../input/Financial Distress.csv', index_col=False,
                 dtype={
                     'Company': np.uint16,
                     'Time': np.uint8,
                     'Financial Distress': np.double
                 })


# In[ ]:


# Look at the Data
print("Number of unique companies:", df.Company.unique().shape[0])  # 422 companies
print("Number of time periods per company:")
print(pd.crosstab(df.Company, df.Time.sum()))  # Some companies have < 5 time periods


# In[ ]:


# Take a look at the data based on Groups per Company
grouped_company = df.groupby('Company')

# Take first 5 groups
group_gen = ((name, group) for name, group in grouped_company)
for name, group in itertools.islice(group_gen, 5):
    # For each group, print and show the data
    print('-------------------------------------')
    print("Data of Company", name)
    print(group.head(15))


# ## Challenges with straightforward Time Series Split:
# 
# As can be seen based on the above code output, although we have 1 common time variable (Time), because we have multiple companies we may have multiple rows belonging to the same Time value (For example, 1 row for Time 1 + Company 1, another row for Time 1 + Company 2 etc).
# 
# This prevents us from using sklearn.model_selection.TimeSeriesSplit, as the assumption of that function is that each row represents a data point from a unique instance of time (and the rows are arranged as per increasing value of time).
# 
# We can still achieve our goal of implementing Forward Chaining by:
# 1. Split the Data Set into multiple groups - 1 group per Company
# 2. For each group, derive the indexes for Forward Chaining
# 3. Combine the list of indexes per group into 1 final index list
# 
# ## Dealing with Dummy Data:
# 
# As mentioned in the Data Dictionary, one of the features is actually a categorical feature. We will therefore create Dummy Columns:

# In[ ]:


# Dummy Variables
dummy_cols = pd.get_dummies(df[['x80']], prefix='dummy', columns=['x80'], drop_first=True)

print(dummy_cols.head())

# Combine dummy_cols back with original data set
x_cols = [col for col in df.columns if all([col.startswith('x'), col != 'x80'])]
df_transformed = pd.concat([df[['Company', 'Time', 'Financial Distress'] + x_cols].reset_index(drop=True),
                            pd.DataFrame(data=dummy_cols)], axis=1)

df_transformed.head()


# ## Creating Lagged Features:
# 
# With the above pre-processing step out of the way, we will now move on to the 2nd piece of Feature Engineering - Creating lagged features (again, lagged features per group):

# In[ ]:


# Helper function to create lagged features
def lagged_features(df_long, lag_features, window=2, lag_prefix='lag', lag_prefix_sep='_'):
    """
    Function calculates lagged features (only for columns mentioned in lag_features)
    based on time_feature column. The highest value of time_feature is retained as a row
    and the lower values of time_feature are added as lagged_features
    :param df_long: Data frame (longitudinal) to create lagged features on
    :param lag_features: A list of columns to be lagged
    :param window: How many lags to perform (0 means no lagged feature will be produced)
    :param lag_prefix: Prefix to name lagged columns.
    :param lag_prefix_sep: Separator to use while naming lagged columns
    :return: Data Frame with lagged features appended as columns
    """
    if not isinstance(lag_features, list):
        # So that while sub-setting DataFrame, we don't get a Series
        lag_features = [lag_features]

    if window <= 0:
        return df_long

    df_working = df_long[lag_features].copy()
    df_result = df_long.copy()
    for i in range(1, window+1):
        df_temp = df_working.shift(i)
        df_temp.columns = [lag_prefix + lag_prefix_sep + str(i) + lag_prefix_sep + x
                           for x in df_temp.columns]
        df_result = pd.concat([df_result.reset_index(drop=True),
                               df_temp.reset_index(drop=True)],
                               axis=1)

    return df_result


# Now split Data Set into groups (based on Company) and create lagged features for each group
grouped_company = df_transformed.groupby('Company')
cols_to_lag = [col for col in df_transformed.columns if col.startswith('x')]
df_cross = pd.DataFrame()

for name, group in grouped_company:
    # For each group, calculate lagged features and rbind to df_cross
    print('----------------------------------------------------')
    print('Working on group:', name, 'with shape', group.shape)
    df_cross = pd.concat([df_cross.reset_index(drop=True),
                          lagged_features(group, cols_to_lag).reset_index(drop=True)],
                         axis=0)
    print('Shape of df_cross', df_cross.shape)
    
# Remove rows with NAs
df_cross = df_cross.dropna()
df_cross.head()


# ## Time Series Splits per group
# 
# Next, we will write a helper function to create time series splits for forward chaining. The function will return a list of tuples. Each tuple will contain 2 values - The train index and the test index.
# 
# Here is how the return value will look like:
# 
#     > [(Int64Index([10, 11, 12], dtype='int64'), (Int64Index[13, 14], dtype='int64')), # For 1st iteration, train on row-index 10-12. test on row-index 13 and 14
#     > (Int64Index([10, 11, 12, 13, 14), Int64Index([15, 16])),                         # For 2nd iteration, train on row-index 10-14. test on row-index 15 and 16
#     > (Int64Index([10, 11, 12, 13, 14, 15, 16]), Int64Index([17, 18]))]                # For 3rd iteration, train on row-index 10-16. test on row-index 17 and 18

# In[ ]:


# Create Time-Series sampling function to draw train-test splits
def ts_sample(df_input, train_rows, test_rows):
    """
    Function to draw specified train_rows and test_rows in time-series rolling sampling format
    :param df_input: Input DataFrame
    :param train_rows: Number of rows to use as training set
    :param test_rows: Number of rows to use as test set
    :return: List of tuples. Each tuple contains 2 lists of indexes corresponding to train and test index
    """
    if df_input.shape[0] <= train_rows:
        return [(df_input.index, pd.Index([]))]

    i = 0
    train_lower, train_upper = 0, train_rows + test_rows*i
    test_lower, test_upper = train_upper, min(train_upper + test_rows, df_input.shape[0])

    result_list = []
    while train_upper < df_input.shape[0]:
        # Get indexes into result_list
        # result_list += [([df_input.index[train_lower], df_input.index[train_upper]],
        #                  [df_input.index[test_lower], df_input.index[test_upper]])]
        result_list += [(df_input.index[train_lower:train_upper],
                         df_input.index[test_lower:test_upper])]

        # Update counter and calculate new indexes
        i += 1
        train_upper = train_rows + test_rows*i
        test_lower, test_upper = train_upper, min(train_upper + test_rows, df_input.shape[0])

    return result_list


# ## Using ts_sample() per group
# 
# The next step is to use ts_sample **per group** of the data. This will give rise to 1 list of index tuples per group. 
# 
# Moreover, because the number of time periods per group is not the same, the size of these lengths will also vary. Therefore, we will need a way to **pad the shorter groups**. How this is done is described in the code comments below:

# In[ ]:


# For each group, apply function ts_sample
# Depending on size of group, the output size of ts_sample (which is a list of (train_index, test_index))
# tuples will vary. However, we want the size of each of these lists to be equal.
# To do that, we will augment the smaller lists by appending the last seen train_index and test_index
# For example:
# group 1 => [(Int64Index([1, 2, 3], dtype='int64'), (Int64Index[4, 5], dtype='int64)),
#             (Int64Index([1, 2, 3, 4, 5], dtype='int64'), (Int64Index([6], dtype='int64'))]
# group 2 => [(Int64Index([10, 11, 12], dtype='int64'), (Int64Index[13, 14], dtype='int64')),
#             (Int64Index([10, 11, 12, 13, 14), Int64Index([15, 16])),
#             (Int64Index([10, 11, 12, 13, 14, 15, 16]), Int64Index([17, 18]))]
# Above, group 2 has 3 folds whereas group 1 has 2. We will augment group 2 to also have 3 folds:
# group 1 => [(Int64Index([1, 2, 3], dtype='int64'), (Int64Index[4, 5], dtype='int64)),
#             (Int64Index([1, 2, 3, 4, 5], dtype='int64'), (Int64Index([6], dtype='int64')),
#             (Int64Index([1, 2, 3, 4, 5, 6]), Int64Index([]))]
grouped_company_cross = df_cross.groupby('Company')
acc = []
max_size = 0
for name, group in grouped_company_cross:
    # For each group, calculate ts_sample and also store largest ts_sample output size
    group_res = ts_sample(group, 4, 4)
    acc += [group_res]
    # print('Working on name:' + str(name))
    # print(acc)

    if len(group_res) > max_size:
        # Update the max_size that we have observed so far
        max_size = len(group_res)

        # All existing lists (apart from the one added latest)in acc need to be augmented
        # to match the new max_size by appending the last value in those list (combining train and test)
        for idx, list_i in enumerate(acc):
            if len(list_i) < max_size:
                last_train, last_test = list_i[-1][0], list_i[-1][1]
                list_i[len(list_i):max_size] = [(last_train.union(last_test),
                                                 pd.Index([]))] * (max_size - len(list_i))

                acc[idx] = list_i

    elif len(group_res) < max_size:
        # Only the last appended list (group_res) needs to be augmented
        last_train, last_test = acc[-1][-1][0], acc[-1][-1][1]
        acc[-1] = acc[-1] + [(last_train.union(last_test), pd.Index([]))] * (max_size - len(acc[-1]))


print(acc[0:2])


# In[ ]:


# acc now contains a list of lists, where each internal list contains tuples of train_index, test_index
# [[(group_1_train_index1, group_1_test_index1), (group_1_train_index2, group_1_test_index2)],
#  [(group_2_train_index1, group_2_test_index1), (group_2_train_index2, group_2_test_index2)],
#  [(group_3_train_index1, group_3_test_index1), (group_3_train_index2, group_3_test_index2)]]
#
# Our goal is to drill-down by removing group-divisions:
# [(train_index1, test_index1), (train_index2, test_index2)]
flat_acc = []
for idx, list_i in enumerate(acc):
    if len(flat_acc) == 0:
        flat_acc += list_i
        continue

    for inner_idx, tuple_i in enumerate(list_i):
        flat_acc[inner_idx] = (flat_acc[inner_idx][0].union(tuple_i[0]),
                               flat_acc[inner_idx][1].union(tuple_i[1]))


print(flat_acc[0:2])


# ## Modeling
# 
# Now that we have our lagged features as well as the indexes ready for Forward Chaining, we can proceed with modeling.
# 
# However, one decision that we will need to take is whether we want to treat this as a classification problem or a regression problem. The 'Financial Distress' column is real-valued, containing both positive and negative values. As per the Data Dictionary, we should consider the company financially distressed if the 'Financial Distress' column is <= -0.50. Accordingly, we will convert this problem into a classification problem by using that definition.

# In[ ]:


# Convert Financial Distress column into 0 or 1
df_model = df_cross.copy()
df_model['Financial Distress'] = ['0' if x > -0.50 else '1' for x in df_model['Financial Distress'].values]

df_model.head()


# In[ ]:


# For each entry in flat_acc, perform train and test and plot metrics
dependent_cols = [col for col in df_model.columns if col != 'Financial Distress']
independent_col = ['Financial Distress']
for idx, tuple_i in enumerate(flat_acc):
    print('---------------------------------------')
    X_train, X_test = df_model.loc[tuple_i[0]][dependent_cols], df_model.loc[tuple_i[1]][dependent_cols]
    y_train, y_test = df_model.loc[tuple_i[0]][independent_col], df_model.loc[tuple_i[1]][independent_col]
    
    # Fit logistic regression model to train data and test on test data
    lr_mod = LogisticRegression(C=0.01, penalty='l2')  # These should be determined by nested cv
    lr_mod.fit(X_train, y_train)
    
    y_pred_proba = lr_mod.predict_proba(X_test)
    y_pred = lr_mod.predict(X_test)
    
    # Print Confusion Matrix and ROC AUC score
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    print('ROC AUC score:')
    print(roc_auc_score(y_test['Financial Distress'].astype(int), y_pred_proba[:, 1]))


# ## Conclusion
# 
# The aim of this notebook was to show that forward chaining may not always be as straight forward as using TimeSeriesSplit function from sklearn. It is important to understand the structure of the data, and design the cross validation approach appropriately.
# 
# Of course, I did gloss over a lot of other aspects in this kernel. We should have also focused on the following:
# 1. **Dealing with imbalanced data:** Handling skewed data involves some flavor of undersampling of majority class + oversampling of minority class. We skipped over that here, but doing that is crucial in getting a good classifier. I plan on doing that as a later revision to this notebook.
# 2. **Nested Validation to choose hyper parameters (like C value in Logistic Regression):** In practice, we would use nested cross validation to also determine the hyper parameters that we have hard coded here.
