# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #machine learning library
import scipy as sp, scipy.stats #fundamental library for scientific computing


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/goodreadsbooks/books.csv"))
print(os.listdir("../input/groceries/groceries.csv"))

#Importing Dataset
pd.options.mode.chained_assignment = None
dataset = r'../Original-Dataset/Online Retail.csv'
Data = pd.read_csv(dataset, dtype= {'CustomerID': 'Int64'})
Data.head()

# Any results you write to the current directory are saved as output.