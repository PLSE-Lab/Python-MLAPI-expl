# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def basic_data_preprocessing(data, nulls_percent=0.75, cardinality_percent=0.80, *cat_columns, correlation_threshold=0.80, test_size=0.25):
    '''
    Parameters

    data : DataFrame (default = None)
    Allows a DataFrame, on which the the operations are to be performed.

    nulls_percent : float or None (default = 0.75)
    Threshold percentage of nulls in a column to remove from DataFrame.

    cardinality_percent: float or None (default = 0.80)
    Threshold percentage a value can occupy in the column. Any column with more then cardinality_percentage will be removed from the DataFrame

    *cat_columns: array-like or None (default = None)
    Allows a list of column names(categorical)

    correlation_threshold: float, int or None (default = 0.80)
    Threshold percentage of correlation between columns to remove them from DataFrame

    test_size : float or None (default : 0.25)
    Represent the proportion of the dataset to include in the test split.
'''
    # initializing a deletable columns list - a empty list
    del_columns = []

    # initializing a list with columns with nulls greater than the threshold value - a empty list
    high_null_columns = []
    for col in data.columns:
        if (data[col].isnull().sum()) >= (len(data)*nulls_percent):
            high_null_columns.append(col)

    # initializing a list with columns with a column value greater than the threshold value - a empty list
    high_cardinal_columns = []
    for col in data.columns:
        if (max((data[col].value_counts().values/len(data))*100) > cardinality_percent):
            high_cardinal_columns.append(col)

    # a list with columns with correlation greater than the threshold value - a empty list
    corr_matrix = data.corr().abs()
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    high_corr_columns = [column for column in upper_matrix.columns if any(upper_matrix[column] > 0.95)]

    # print all the lists for user reference
    print ('Columns with high amount of nulls :' + str(high_null_columns))
    print ('Columns with high percentage of single column value :' + str(high_cardinal_columns))
    print ('Columns with high correlation :' + str(high_corr_columns))

    # combine all the lists and remove the deletable list from the dataframe
    del_columns = high_null_columns + high_cardinal_columns + high_corr_columns
    data.drop(del_columns, axis = 1, inplace = True)

    # dealing with categorical columns
    if (len(cat_columns)>0):
        data = pd.get_dummies(columns = cat_columns, data = data)
    return data