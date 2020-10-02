# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
iowa_liquor_sales = pd.read_csv('../input/iowa-liquor-sales/Iowa_Liquor_Sales.csv')

# Printing number of rows and columns of entire data frame
iowa_liquor_sales.shape

# Prnting data frame columns
print(iowa_liquor_sales.columns.values)

# Getting new column 'year'
iowa_liquor_sales['year'] = pd.DatetimeIndex(iowa_liquor_sales['Date']).year

# Checking for unique 'year' column
iowa_liquor_sales['year'].unique()

# Getting new column 'month'
iowa_liquor_sales['month'] = pd.DatetimeIndex(iowa_liquor_sales['Date']).month

# Checking for unique 'month' column
iowa_liquor_sales['month'].unique()

# Filtering 2017 year data last six months
liquor_sales_2017 = iowa_liquor_sales[(iowa_liquor_sales['year'] == 2017) & (iowa_liquor_sales['month'].isin([7,8,9,10,11,12]))]

# Checking for number of rows and columns in 2017 data
liquor_sales_2017.shape

# Writing the csv file 
liquor_sales_2017.to_csv('liquor_sales_2017_last_six_months.csv', index = False)