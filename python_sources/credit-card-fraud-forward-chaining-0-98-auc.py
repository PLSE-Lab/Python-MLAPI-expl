#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection - Forward Chaining
# 
# The Credit Card Fraud Dataset is a time-series data set. Accordingly, methods like K-Fold Cross Validation cannot be used because they end up destroying the temporal nature of the data set. Instead, the correct approach is to use what is known as [Forward Chaining](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection) in order to test the performance of the model while maintaining the temporal structure.
# 
# ### Agenda:
# 1. Data Preprocessing
# 2. Feature Engineering
# 3. Model Fitting
# 4. Model Evaluation via Forward Chaining

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
import os
print(os.listdir("../input"))


# ## Load Data Set

# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
data.head()


# ## Check various columns

# In[ ]:


# Check Time Column
data[['Time']].describe()


# Time column seems clean. Minimum value is 0 and maximum value is 172792

# In[ ]:


# How are the class labels spread
pd.value_counts(data['Class'])


# ## SMOTE + Undersampling
# As we knew already, the classes are highly imbalanced!
# 
# We will need to resolve this. There are various approaches, but here I will be using SMOTE + Undersampling of majority class.
# 
# SMOTE stands for Synthetic Minority Oversampling Technique. SMOTE seeks to create "synthetic minority data points" (that is, it creates artificial data points belonging to the minority class) based on randomized interpolation between a minority data point and 'n' randomly chosen other nearby minority data points.
# 
# [Technical Paper on SMOTE](https://arxiv.org/abs/1106.1813)
# 
# Here is a helpful visual explanation of how SMOTE works:
# 
# ![Helpful pictorial representation of how SMOTE works](https://www.researchgate.net/publication/282830682/figure/fig1/AS:324534361182217@1454386429561/Schematic-diagram-of-SMOTE-algorithm.png)
# 
# ## Feature Engineering
# Since the data provided is time series data, we can create lagged features. to give our classification model reference values into previous times. The important idea is to do this **before** doing the SMOTE + Undersampling, as the sampling will mess up the time series structure.
# 
# Another idea is to not rely on just 1 Time feature and instead break up the Time feature (currently in seconds) into 3 new features: Hours, Minutes, Seconds

# In[ ]:


# Convert Time-rows into columns (window method) T, T-1, T-2 (3 at a time)
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


# Augment data set with lagged features
lag_features = [col for col in data.columns if col not in ['Time', 'Amount', 'Class']]
data = lagged_features(data, lag_features)

print(data.shape)

data.head()


# In[ ]:


# Remove missing values that are introduced due to lagging
data = data.dropna()

# Restructure data set to keep 'Class' at end
col_order = [col for col in data.columns if col != 'Class'] + ['Class']
data = data[col_order]
data.head()


# In[ ]:


# SMOTE + Undersampling
# Split data into X and y
X, y = data[[col for col in data.columns if col != 'Class']], data[['Class']].astype(str)
print(X.head())
print(y.head())

# Undersampling of majority class
rus = RandomUnderSampler(sampling_strategy={'0': 8124, '1': 492}, 
                         replacement=True, 
                         random_state=123)
X_under, y_under = rus.fit_resample(X, y)
print(X_under.shape)
print(y_under.shape)

data_under = pd.DataFrame(np.hstack((X_under, y_under)))
data_under.columns = data.columns
print(data_under.shape)
print(data_under.head())

# Clean up temp variables
del X, y, X_under, y_under

X, y = data_under[[col for col in data_under.columns if col != 'Class']], data_under[['Class']]
print(y.head())

# SMOTE sampling of minority class
smote = SMOTE(sampling_strategy={'0': 8124, '1': 3444},
              k_neighbors=7,
              random_state=213)
X_smote, y_smote = smote.fit_resample(X, y)
print(X_smote.shape)
print(y_smote.shape)  # For some odd reason, the shape of y_smote is (nrow, ) instead of (nrow, 1). So we need to fix that.
y_smote = np.reshape(y_smote, (y_smote.shape[0], 1))

data_smote = pd.DataFrame(np.hstack((X_smote, y_smote)))
data_smote.columns = data.columns
print(data_smote.shape)
print(data_under.head())

# Look at class counts now
print(pd.value_counts(data_smote['Class']))
print(data_smote.columns)

data_sorted = data_smote.sort_values(['Time'], axis=0, ascending=[1]).reset_index(drop=True)

print('Final dataset glimpse...')
print(data_sorted.head())

# Clean up variables and objects not needed
del rus, smote, data_under, X, y, X_smote, y_smote, data_smote


# In[ ]:


# Augment data set with hours, minutes, seconds break up of 'Time'
data_time = data_sorted.Time.apply(lambda x: pd.Series({'Hours': np.floor(x/(60*60)), 
                                                        'Minutes': np.floor((x - np.floor(x/(60*60)) * (60*60))/60), 
                                                        'Seconds': x - np.floor(x/(60*60)) * (60*60) - np.floor((x - np.floor(x/(60*60)) * (60*60))/60) * 60
                                                       }))
data_aug = pd.concat([data_sorted.reset_index(drop=True), data_time.reset_index(drop=True)], axis=1)

# Reorder columns to keep time columns close together
col_order = ['Time', 'Hours', 'Minutes', 'Seconds'] + [col for col in data_aug.columns if col not in ['Time', 'Hours', 'Minutes', 'Seconds']]
data_aug = data_aug[col_order]

print(data_aug.shape)
data_aug.head()


# ## Data Pre-Processing
# 
# To pre-process the data, we will only need to Standardize the values in the Amount column. This is because the other features are already standardized via transformation through PCA.
# 

# In[ ]:


# Standardize Amount column
data_aug['Amount'] = StandardScaler().fit_transform(data_sorted['Amount'].values.reshape(-1, 1))
data_aug.head()


# ## Time Series Split + Logistic Regression
# 
# As mentioned earlier, we cannot use Cross Validation with Time Series Data. Instead, we will use Forward Chaining to ensure that our train and test sets are always using information available either in that moment, or from the past (never from the future).
# 
# We will use sklearn's ModelSelection sub-module for this. For each split, we will fit a logistic regression model on the split's train data and then test on the split's test data.

# In[ ]:


# Define object to handle time series splits
tscv = TimeSeriesSplit(n_splits=5)

# Loop over time series splits, fit model, and test on test data
dependent_cols = [col for col in data_aug.columns if col not in ['Class', 'Time']]  # Time information has been captured in Hours, Minutes, Seconds
independent_col = ['Class']
for train_index, test_index in tscv.split(data_aug):
    print('------------------------------------')
    X_train, X_test = data_aug.iloc[train_index][dependent_cols], data_aug.iloc[test_index][dependent_cols]
    y_train, y_test = data_aug.iloc[train_index][independent_col], data_aug.iloc[test_index][independent_col]
    
    # Fit logistic regression model to train data and test on test data
    lr_mod = LogisticRegression(C = 0.01, penalty='l2')  # The value of C should be determined by nested validation
    lr_mod.fit(X_train, y_train)
    
    y_pred_proba = lr_mod.predict_proba(X_test)
    y_pred = lr_mod.predict(X_test)
    
    # Print Confusion Matrix and ROC AUC score
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    print('ROC AUC score:')
    print(roc_auc_score(y_test.Class.astype(int), y_pred_proba[:, 1]))
    


# ## Conclusion
# 
# The primary purpose of this notebook was to show case how time series data set should be treated differently than cross-sectional data sets. The main take away for the reader should be with respect to feature engineering (some feature engineering should be done before sampling, some can be done after sampling), as well as how to use forward chaining in order to assess model performance.
# 
# Nevertheless, I did gloss over some approaches which should not be done in practise. Here are some ways that our approach could be even more robust:
# 1. **Nested Cross Validation to choose hyper-parameters (like SMOTE sampling and Undersampling parameters):** In our approach, we took a rough approach to oversample the minority class and undersample the majority class. In practice, we could choose these values by nested cross validation. In nested cross validation, we would create n_outer number of parallel datasets where each dataset was sampled somewhat differently. For instance, with n_outer = 5, we could have 5 parallel approaches where in the first approach we use SMOTE with k_neighbors = 3, in the 2nd approach we have k_neighbors = 5 etc.
# 2. **Standardization of columns should be done only for Train Data**: In this notebook, we used StandardScaler at one go, after which the data set is split into X_train and X_test. However, this leaks information about X_train to X_test. The correct approach should be to first split into X_train, X_test, then use StandardScaler on X_train, and then use the mean and sd values so obtained to transform X_test.
