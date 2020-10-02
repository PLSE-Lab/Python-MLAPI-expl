#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


class FeatureAggregator(object):

    """Feature aggregator - automated feature aggregation method.
    Two ways of usage, either selected aggregations can be applied onto
    numerical and categorical columns or specific combinations of aggregates
    can be set for each column.

    # Arguments:
        df: (pandas DataFrame), DataFrame to create features from.
        aggregates_cat: (list), list containing aggregates for
            categorical features
        aggregates_num: (list), list containing aggregates for
            numerical features.

    """

    def __init__(self,
                 df,
                 aggregates_cat=['mean', 'std'],
                 aggregates_num=['mean', 'std', 'sem', 'min', 'max']):

        self.df = df.copy()
        self.aggregates_cat = aggregates_cat
        self.aggregates_num = aggregates_num

    def process_features_batch(self,
                               categorical_columns=None,
                               categorical_int_columns=None,
                               numerical_columns=None,
                               to_group=['SK_ID_CURR'], prefix='BUREAU'):
        """Process, group features in batch.

        # Arguments:
            categorical_columns: (list), list of categorical columns, which need
            to be label-encoded (factorized).
            categorical_int_columns: (list), list of categorical columns, which
            are already of integer type.
            numerical_columns: (list), list of numerical columns.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_cat/df_num: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if categorical_columns is not None:
            assert len(categorical_columns) > 0, 'No columns to encode.'
            self.categorical_features_factorize(categorical_columns)
            df_cat = self.create_aggregates_set(
                columns=categorical_columns,
                aggregates=self.aggregates_cat,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_cat shape: {}'.format(df_cat.shape))
            return df_cat

        if categorical_int_columns is not None:
            assert len(categorical_int_columns) > 0, 'No columns to encode.'
            df_cat = self.create_aggregates_set(
                columns=categorical_int_columns,
                aggregates=self.aggregates_cat,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_cat int shape: {}'.format(df_cat.shape))
            return df_cat

        if numerical_columns is not None:
            assert len(numerical_columns) > 0, 'No columns to encode.'
            df_num = self.create_aggregates_set(
                columns=numerical_columns,
                aggregates=self.aggregates_num,
                to_group=to_group, prefix=prefix)
            print('\nAggregated df_num shape: {}'.format(df_num.shape))
            return df_num

        return

    def process_features_selected(self,
                                  aggregations,
                                  categorical_columns,
                                  to_group=['SK_ID_CURR'], prefix='BUREAU'):
        """Process, group features for selected combinations of aggregates
        and columns.

        # Arguments:
            categorical_columns: (list), list of categorical columns, which need
            to be label-encoded (factorized).
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_agg: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if categorical_columns:
            # Provide categorical_columns argument if some features need to be factorized.
            self.categorical_features_factorize(categorical_columns)

        df_agg = self.create_aggregates_set(
            aggregations=aggregations,
            to_group=to_group, prefix=prefix)

        print('\nAggregated df_agg shape: {}'.format(df_agg.shape))

        return df_agg

    def create_aggregates_set(self,
                              aggregations=None,
                              columns=None,
                              aggregates=None,
                              to_group=['SK_ID_CURR'],
                              prefix='BUREAU'):
        """Create selected aggregates.

        # Arguments:
            aggregations: (dict), dictionary specifying aggregates for selected columns.
            columns: (list), list of columns to group for batch aggregation.
            aggregates: (list), list of aggregates to apply on columns argument
            for batch aggregation.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for columns names.

        # Returns:
            df_agg: (pandas DataFrame), DataFrame with aggregated columns.

        """

        assert isinstance(
            to_group, list), 'Variable to group by must be of type list.'

        if aggregations is not None:
            print('Selected aggregations:\n{}\n.'.format(aggregations))
            df_agg = self.df.groupby(
                to_group).agg(aggregations)

        if columns is not None and aggregates is not None:
            print('Batch aggregations on columns:\n{}\n.'.format(columns))
            df_agg = self.df.groupby(
                to_group)[columns].agg(aggregates)

        df_agg.columns = pd.Index(['{}_{}_{}'.format(
            prefix, c[0], c[1].upper()) for c in df_agg.columns.tolist()])
        df_agg = df_agg.reset_index()

        return df_agg

    def get_column_types(self):
        """Select categorical (to be factorized), categorical integer and numerical
        columns based on their dtypes. This facilitates proper grouping and aggregates selection for
        different types of variables.
        Categorical columns needs to be factorized, if they are not of
        integer type.

        # Arguments:
            self.df: (pandas DataFrame), DataFrame to select variables from.

        # Returns:
            categorical_columns: (list), list of categorical columns which need factorization.
            categorical_columns_int: (list), list of categorical columns of integer dtype.
            numerical_columns: (list), list of numerical columns.
        """

        categorical_columns = [
            col for col in self.df.columns if self.df[col].dtype == 'object']
        categorical_columns_int = [
            col for col in self.df.columns if self.df[col].dtype == 'int']
        numerical_columns = [
            col for col in self.df.columns if self.df[col].dtype == 'float']

        categorical_columns = [
            x for x in categorical_columns if 'SK_ID' not in x]
        categorical_columns_int = [
            x for x in categorical_columns_int if 'SK_ID' not in x]

        print('DF contains:\n{} categorical object columns\n{} categorical int columns\n{} numerical columns.\n'.format(
            len(categorical_columns), len(categorical_columns_int), len(numerical_columns)))

        return categorical_columns, categorical_columns_int, numerical_columns

    def categorical_features_factorize(self, categorical_columns):
        """Factorize categorical columns, which are of non-number dtype.

        # Arguments:
            self.df: (pandas DataFrame), DataFrame to select variables from.
            Transformation is applied inplace.

        """

        print('\nCategorical features encoding: {}'.format(categorical_columns))

        for col in categorical_columns:
            self.df[col] = pd.factorize(self.df[col])[0]

        print('Categorical features encoded.\n')

        return

    def check_and_save_file(self, df, filename, dst='../input/'):
        """Utility function to check if there isn't a file with the same name already.

        # Arguments:
            df: (pandas DataFrame), DataFrame to save.
            filename: (string), filename to save DataFrame with.

        """

        filename = '{}{}.pkl'.format(dst, filename)
        if not os.path.isfile(filename):
            print('Saving: {}'.format(filename))
            df.to_pickle('{}'.format(filename))
        return


def feature_aggregator_on_df(df,
                             aggregates_cat,
                             aggregates_num,
                             to_group,
                             prefix,
                             suffix='basic',
                             save=False,
                             categorical_columns_override=None,
                             categorical_int_columns_override=None,
                             numerical_columns_override=None):
    """Wrapper for FeatureAggregator to process dataframe end-to-end using batch aggregation.
    It takes lists of aggregates for categorical and numerical features, which are created for
    selected column (to_group), by which data is grouped. In addition to that, prefix and suffix can
    be provided to facilitate column naming.
    _override arguments can be used if only selected subset of each type of columns should
    be aggregated. If those are not provided, FeatureAggregator processes all columns for each type.

        # Arguments:
            aggregates_cat: (list), list of aggregates to apply to categorical features.
            aggregates_num: (list), list of aggregates to apply to numerical features.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for column names.
            suffix: (string), suffix for filename.
            save: (boolean), whether to save processed DF.
            categorical_columns_override: (list), list of categorical columns
            to override default, inferred list.
            categorical_int_columns_override: (list), list of categorical integer
            columns to override default, inferred list.
            numerical_columns_override: (list), list of numerical columns
            to override default, inferred list.

        # Returns:
            to_return: (list of pandas DataFrames), DataFrames with aggregated columns,
            one for each type of column types. This is due to the fact that not every
            raw dataframe may contain all types of columns.

        """

    assert isinstance(aggregates_cat, list), 'Aggregates must be of type list.'
    assert isinstance(aggregates_num, list), 'Aggregates must be of type list.'

    t = time.time()
    to_return = []

    column_base = ''
    for i in to_group:
        column_base += '{}_'.format(i)

    feature_aggregator_df = FeatureAggregator(
        df=df,
        aggregates_cat=aggregates_cat,
        aggregates_num=aggregates_num)

    print('DF prefix: {}, suffix: {}'.format(prefix, suffix))
    print('Categorical aggregates - {}'.format(aggregates_cat))
    print('Numerical aggregates - {}'.format(aggregates_num))

    df_cat_cols, df_cat_int_cols, df_num_cols = feature_aggregator_df.get_column_types()

    if categorical_columns_override is not None:
        print('Overriding categorical_columns.')
        df_cat_cols = categorical_columns_override
    if categorical_columns_override is not None:
        print('Overriding categorical_int_columns.')
        df_cat_int_cols = categorical_int_columns_override
    if categorical_columns_override is not None:
        print('Overriding numerical_columns.')
        df_num_cols = numerical_columns_override

    if len(df_cat_cols) > 0:
        df_curr_cat = feature_aggregator_df.process_features_batch(
            categorical_columns=df_cat_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_cat, '{}_cat_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_cat)
        del df_curr_cat
        gc.collect()

    if len(df_cat_int_cols) > 0:
        df_curr_cat_int = feature_aggregator_df.process_features_batch(
            categorical_int_columns=df_cat_int_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_cat_int, '{}_cat_int_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_cat_int)
        del df_curr_cat_int
        gc.collect()

    if len(df_num_cols) > 0:
        df_curr_num = feature_aggregator_df.process_features_batch(
            numerical_columns=df_num_cols,
            to_group=to_group, prefix=prefix)
        if save:
            feature_aggregator_df.check_and_save_file(
                df_curr_num, '{}_num_{}_{}'.format(prefix, column_base, suffix))
        to_return.append(df_curr_num)
        del df_curr_num
        gc.collect()

    print('\nTime it took to create features on df: {:.3f}s'.format(
        time.time() - t))

    return to_return


def feature_aggregator_on_df_selected(df,
                                      aggregations,
                                      to_group,
                                      prefix,
                                      suffix='basic',
                                      save=False):
    """Wrapper for FeatureAggregator to process dataframe end-to-end using selected
    aggregates/columns combinations.
    It takes dictionary of aggregates/columns combination for selected features,
    which are created for selected column (to_group), by which data is grouped.
    In addition to that, prefix and suffix can be provided to facilitate column naming.

        # Arguments:
            aggregations: (dict), dictionary containing combination of columns/aggregates.
            to_group: (list), list of columns to group by.
            prefix: (string), prefix for column names.
            suffix: (string), suffix for filename.
            save: (boolean), whether to save processed DF.

        # Returns:
            to_return: (list of pandas DataFrames), DataFrames with aggregated columns,
            one for each type of column types. This is due to the fact that not every
            raw dataframe may contain all types of columns.

        """

    assert isinstance(
        to_group, list), 'Variable to group by must be of type list.'

    t = time.time()
    to_return = []

    column_base = ''
    for i in to_group:
        column_base += '{}_'.format(i)

    feature_aggregator_df = FeatureAggregator(df=df)

    print('DF prefix: {}, suffix: {}'.format(prefix, suffix))

    df_cat_cols, df_cat_int_cols, df_num_cols = feature_aggregator_df.get_column_types()

    if len(df_cat_cols) > 0:
        df_aggs = feature_aggregator_df.process_features_selected(
            aggregations=aggregations,
            categorical_columns=df_cat_cols,
            to_group=to_group,
            prefix=prefix)
    else:
        df_aggs = feature_aggregator_df.process_features_selected(
            aggregations=aggregations,
            to_group=to_group,
            prefix=prefix)

    if save:
        feature_aggregator_df.check_and_save_file(
            df_aggs, '{}_selected_{}_{}'.format(prefix, column_base, suffix))

    to_return.append(df_aggs)
    del df_aggs
    gc.collect()

    print('\nTime it took to create features on df: {:.3f}s'.format(
        time.time() - t))

    return to_return


# ## I. Idea
# 
# Group features, aggregates, are one of the most powerful way to capture relationships between variables in the dataset. Sometimes it is possible to group by target variable and thus provide model with direct information about it (although one should be careful when doing that in order not to introduce a leak, only training data subset can be grouped this way).
# For grouping of other variables, whole dataset can be used, as you are given both train and test data.
# **Important!** - this concerns Kaggle competitions, one should not do this in real-life ML, as you never know what exactly will the distribution of variables in test data be.
# In this kernel I try to create an end-to-end feature engineering solution based on groupby features.
# 
# Whole process can be divided into a few steps:
#   1. Selection of columns for each type:
#     1. `categorical` - categorical features which must be encoded (standard label encoding is used).
#     2. `categorical_int` - categorical features which are alredy in integer dtype, encoding is not needed.
#     3. `numerical` - numerical features
#   2. Factorization of `categorical` columns, if needed.
#   3. Creation of aggregates with columns renaming for each type of columns.
#   4. Resulting DataFrame is saved (if possible, will not work in Kaggle Kernels).
# 
# There are two ways of aggregated features creation:
#   1. Batch aggregation: all columns from DataFrame are processed, each column is appended to one of three types, `categorical` columns are factorized is there's a need and selected aggregates are applied to each column type. There is a distinction between aggregates for categorical columns and those for numerical, as each type requires a different approach.
#   2. Selected aggregation: combination of aggregates/columns should be provided in form of a dictionary, where for each column aggregates are specified in a list.
# 
# ## II. Setup
# 
# First, we need to choose aggregates, which will be used for grouping categorical and numerical variables.
# For numerical variables `aggs_num_basic` will be used, for categoricals - `aggs_cat_basic`.
# Those types of aggregations can be extended further, as is shown in `aggs1` list.
# 
# ```python
# aggs_num_basic = ['mean', 'min', 'max', 'std', 'sem', 'sum']
# aggs_cat_basic = ['mean', 'std', 'sum']
# aggs1 = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum', 'mad']
# ```

# In[ ]:


aggs1 = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum', 'mad']
aggs2 = ['mean', 'median', 'min', 'max', 'count', 'std', 'sem', 'sum']

aggs_num_basic = ['mean', 'min', 'max', 'std', 'sem', 'sum']
aggs_cat_basic = ['mean', 'std', 'sum']

aggs_num = aggs_num_basic
aggs_cat = aggs_cat_basic


# ## Loading the data:

# In[ ]:


train = pd.read_csv("../input/application_train.csv") 
test = pd.read_csv("../input/application_test.csv")

bureau = pd.read_csv("../input/bureau.csv")
bureau_bal = pd.read_csv('../input/bureau_balance.csv')


# ## IIIa. Bureau balance & Bureau, batch aggregation:
# 
# 
# Aggregates for Bureau Balance will be created at first.
# Here, `bureau_bal` DF is grouped by `SK_ID_BUREAU`, which will serve as a basis for further merge of this set to `bureau` data. `aggs_cat` and `aggs_num` are used for categorical and numerical features aggregations.
# `bureau_balance` is the prefix for column names and `basic` is the suffix for filename, if resulting DFs should be saved.
# 
# Second step is aggregating the `bureau` data. We begin with merging `bureau_bal` data into the Bureau set, as this will enable engineering of the features from Bureal Balance dataset in a way enabling their merge to main training DataFrame. `bureau` is grouped by `SK_ID_CURR`, as this is the ID column in `train`.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nbureau_bal_dfs = feature_aggregator_on_df(\n    bureau_bal, aggs_cat, aggs_num, ['SK_ID_BUREAU'], 'bureau_balance', 'basic', save=False)\n\nbureau_bal_dfs[0]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nbureau_ = bureau.merge(bureau_bal_dfs[0], how='left', on='SK_ID_BUREAU', copy=False)\nbureau_ = bureau_.merge(bureau_bal_dfs[1], how='left', on='SK_ID_BUREAU', copy=False)\n\nbureau_dfs = feature_aggregator_on_df(\n    bureau_, aggs_cat, aggs_num, ['SK_ID_CURR'], 'bureau', 'basic', save=False)\n\nbureau_dfs[1]")


# ## IIIb. Previous applications, selected aggregation:
# 
# The other way of aggregating features enabled by FeatureAggregator is __selected_aggregation__. Here, a dictionary is of selected combinations between columns and aggregates for them is specified.
# DF processed as example is `previous_application`. We start with replacing some of the values, which do not make sense with `np.nan` and create one new column, `APP_CREDIT_PERC` - those are steps made in one of the other kernels :).
# Afterwards, a dictionary specifying aggregations is created in `num_aggregations`, as this is one for numerical columns.
# DF is grouped according to this dictionary and is then grouped by `SK_ID_CURR` to facilitate merging onto the train table.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprev = pd.read_csv("../input/previous_application.csv")\nprev = prev.drop([\'SK_ID_PREV\'], axis=1)\n\nprev[\'DAYS_FIRST_DRAWING\'].replace(365243, np.nan, inplace= True)\nprev[\'DAYS_FIRST_DUE\'].replace(365243, np.nan, inplace= True)\nprev[\'DAYS_LAST_DUE_1ST_VERSION\'].replace(365243, np.nan, inplace= True)\nprev[\'DAYS_LAST_DUE\'].replace(365243, np.nan, inplace= True)\nprev[\'DAYS_TERMINATION\'].replace(365243, np.nan, inplace= True)\nprev[\'APP_CREDIT_PERC\'] = prev[\'AMT_APPLICATION\'] / prev[\'AMT_CREDIT\']\n\n\nnum_aggregations = {\n        \'AMT_ANNUITY\': [\'min\', \'max\', \'mean\'],\n        \'AMT_APPLICATION\': [\'min\', \'max\', \'mean\'],\n        \'AMT_CREDIT\': [\'min\', \'max\', \'mean\'],\n        \'APP_CREDIT_PERC\': [\'min\', \'max\', \'mean\', \'var\'],\n        \'AMT_DOWN_PAYMENT\': [\'min\', \'max\', \'mean\'],\n        \'AMT_GOODS_PRICE\': [\'min\', \'max\', \'mean\'],\n        \'HOUR_APPR_PROCESS_START\': [\'min\', \'max\', \'mean\'],\n        \'RATE_DOWN_PAYMENT\': [\'min\', \'max\', \'mean\'],\n        \'DAYS_DECISION\': [\'min\', \'max\', \'mean\'],\n        \'CNT_PAYMENT\': [\'mean\', \'sum\'],\n    }\n\nprev_selected_aggs = feature_aggregator_on_df_selected(\n    prev, num_aggregations, to_group=[\'SK_ID_CURR\'], prefix=\'prev\', suffix=\'basic_selected\', save=False)')


# ### and batch aggregations examples for the rest of the tables...

# In[ ]:


cred_card_bal = pd.read_csv("../input/credit_card_balance.csv")
cred_card_bal = cred_card_bal.drop(['SK_ID_PREV'], axis=1)

cred_card_bal_dfs = feature_aggregator_on_df(
    cred_card_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'cred_card_balance', 'basic', save=False)


# In[ ]:


pos_cash_bal = pd.read_csv("../input/POS_CASH_balance.csv")
pos_cash_bal = pos_cash_bal.drop(['SK_ID_PREV'], axis=1)

pos_cash_bal_dfs = feature_aggregator_on_df(
    pos_cash_bal, aggs_cat, aggs_num, ['SK_ID_CURR'], 'pos_cash_balance', 'basic', save=False)


# In[ ]:


ins = pd.read_csv('../input/installments_payments.csv')
ins = ins.drop(['SK_ID_PREV'], axis=1)

ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

ins_dfs = feature_aggregator_on_df(
    ins, aggs_cat, aggs_num, ['SK_ID_CURR'], 'installments', 'basic', save=False)


# In[ ]:




