# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt

train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])
test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])
ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])

df_train = pd.merge(train, ppl, on='people_id')
df_test = pd.merge(test, ppl, on='people_id')
del train, test, ppl

print("Training Data Set: ",df_train.shape)

print("Test Data Set: ",df_test.shape)

print("Column names: ",df_train.columns)

#View first 5 records from dataset
print(df_train.head(5))

#People_id and activity_id together form the primary key

#Find the number of valid entries in each column
print(df_train.count())

#Summary of Numerical columns
print(df_train.describe())
#Summary of all columns
print(df_train.describe(include =  'all'))

print(df_train.groupby('outcome').size())
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

act_stack = df_train.groupby(['outcome','activity_category'])['outcome'].count().unstack()

act_stack.plot(kind='bar',stacked=True,rot=1)
# Any results you write to the current directory are saved as output.