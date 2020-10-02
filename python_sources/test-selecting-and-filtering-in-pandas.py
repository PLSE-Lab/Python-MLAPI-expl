# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Practicing pandas
# Learning from example on "https://www.kaggle.com/dansbecker/selecting-and-filtering-in-pandas"

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

melbourne_file_path = '../input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)

# Any results you write to the current directory are saved as output.

# By convention, the names should also be converted to lower case. Pandas is case sensitive, so future calls to all of the columns will need to be updated.

melbourne_data.columns = [col.lower() for col in melbourne_data.columns]
print(melbourne_data.columns)

# Store the series of address separately as melbourne_address_data
melbourne_address_data = melbourne_data.address
# the head command returns the top few lines of data (Default is 5 lines)
print(melbourne_address_data.head())

# Select multiple columns by providing a list of column names inside brackets
columns_of_interest = ['lattitude','longtitude']
two_columns_of_data = melbourne_data[columns_of_interest]

# verify we got the columns we need with the describe() method
two_columns_of_data.describe()
