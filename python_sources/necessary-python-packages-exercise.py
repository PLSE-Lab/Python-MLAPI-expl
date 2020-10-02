# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#"""importing necessary packages for exercise"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib as mt
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#"""reading the ASHRAE data: efforts"""

#del building_meta, building_train, chunk, my_list, weather_test, weather_train
#del my_list, building_meta

#building_train = pd.read_csv("D:/Work/ashrae-energy-prediction/train.csv", low_memory=False, error_bad_lines=False)
#my_list = []
#for chunk in pd.read_csv("D:/Work/ashrae-energy-prediction/train.csv",chunksize=15000):
#    my_list.append(chunk)
#building_train = pd.concat(my_list,axis=0)
#train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
#ny_data_df=pd.read_csv('../input/AB_NYC_2019.csv')

#building_train_ch = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', iterator=True, chunksize=1000)
#building_train_df = pd.concat(building_train_ch, ignore_index=True)

#building_meta_df = pd.read_csv("D:/Work/ashrae-energy-prediction/building_metadata.csv", low_memory=False, error_bad_lines=False)
#my_list = []
#for chunk in pd.read_csv("D:/Work/ashrae-energy-prediction/building_metadata.csv",chunksize=10000):
#    my_list.append(chunk)
#building_meta = pd.concat(my_list,axis=0)
#del my_list
#building_meta_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

#building_test = pd.read_csv("D:/Work/ashrae-energy-prediction/test.csv", low_memory=False, error_bad_lines=False)
#my_list = []
#for chunk in pd.read_csv("D:/Work/ashrae-energy-prediction/test.csv",chunksize=1000):
#    my_list.append(chunk)
#building_test = pd.concat(my_list,axis=0)
#del my_list
#building_test_ch = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',iterator=True,chunksize=1000)
#building_test_df = pd.concat(building_test_ch, ignore_index=True)

#weather_train = pd.read_csv("D:/Work/ashrae-energy-prediction/weather_train.csv", low_memory=False, error_bad_lines=False)
#weather_train_ch = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',iterator=True,chunksize=1000)
#weather_train_df = pd.concat(weather_train_ch, ignore_index=True)

#weather_test = pd.read_csv("D:/Work/ashrae-energy-prediction/weather_test.csv", low_memory=False, error_bad_lines=False)
#weather_test_ch = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv',iterator=True,chunksize=1000)
#weather_test_df = pd.concat(weather_test_ch, ignore_index=True)

#sample_submit = pd.read_csv("D:/Work/ashrae-energy-prediction/sample_submission.csv", low_memory=False, error_bad_lines=False)
#sample_submit_ch = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv', iterator=True,chunksize=1000)
#sample_submit_df = pd.concat(sample_submit_ch, ignore_index=True)

#building_train.head()
#del building_meta_df
#building_meta_df.head()
#building_train_df.head()

#"""Pandas exercise"""
s = pd.Series(['apple','banana',1,6,100,np.nan,78])
s
s1 = pd.Series(['apple','melon','avocado'])
s1
building_train_df.head()
s2 = pd.Series([1,5,8,np.nan,78])
s2
s3 = pd.date_range('20190601',periods=6)
s3
df = pd.DataFrame(np.random.randn(6, 4), index=s3, columns=list('ABCD'))
df
df = pd.DataFrame(np.random.randn(6, 4), index=s3, columns=list(['n1','n2','n3','n4']))
df
s4 = list(['n1','n2','n3','n4'])
s4
s5 = ['n1','n2','n3','n4']
s5
s6 = np.array(['n1','n2','n3','n4'])
s6
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20190601'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4,dtype='int32'),
                    'E': pd.Categorical(['test','train','test','train']),
                    'F': 'foo'})
df2
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20190601'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3,5,10,7],dtype='int32'),
                    'E': pd.Categorical(['test','train','test','train']),
                    'F': 'foo'})
df2
df2.dtypes
df.head()
df.tail()
df.index
df.to_numpy()
df2.to_numpy()
df.describe()
df.T
df
df.sort_index(axis=1,ascending=False)
df.sort_values(by='n2')
df['n2']
df[0:3]
df['20190604':'20190606']
df.loc[s3[0]]
df.loc[s3[1]]
df.loc[:,['n1','n2']]
df.loc['20190604':'20190606',list(['n2','n1'])]
df.loc['20190604',list(['n2','n1'])]
df.loc[s3[3],list(['n2'])]
df.at[s3[3],'n2']
df.iloc[0]
df.iloc[3:5,0:2]
df.iloc[list([1,2,4]),list([0,2])]
df.iloc[1:3, :]
df.iloc[:, 1:3]
df.iloc[1, 1]
df.iat[1, 1]
df[df.n1 > 0.20]
df[df > 0]
df
df2 = df.copy()
df2['n5'] = ['one','one','two','three','four','three']
df2
df2[df2['n5'].isin(list(['two','four']))]
df2[df2.n5.isin(list(['two','four']))]
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
s1
df.loc[s3[0], 'A'] = 0
df.iloc[0, 1] = 0
df.loc[:, 'n4'] = np.array([5] * len(df))
df
df2 = df.copy()
df2[df2 > 0] = -df2
df2
df1 = df.reindex(index=s3[0:4], columns=list(df.columns) + ['E'])
df1.loc[s3[0]:s3[1], 'E'] = 1
df1
df1.dropna(how='any')
df1.fillna(value=5)
pd.isna(df1)
df.mean()
df.mean(1)
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=s3).shift(2)
s
df.sub(s, axis='index')
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
s = pd.Series(np.random.randint(0, 7, size=10))
s
s.value_counts()
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()
df = pd.DataFrame(np.random.randn(10, 4))
df
pieces = [df[:3], df[3:7], df[7:]]
pieces
pd.concat(pieces)
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
df
s = df.iloc[3]
df.append(s, ignore_index=True)
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
df.groupby('A').sum()
df.groupby(['A', 'B']).sum()
tuples = list(zip(*[['bar','bar','baz','baz','foo','foo','qux','qux'],
                    ['one','two','one','two','one','two','one','two']]))
tuples
index = pd.MultiIndex.from_tuples(tuples, names=list(['first','second']))
df = pd.DataFrame(np.random.randn(8,2),index=index,columns=['A','B'])
df
df2 = df[:4]
df2
stacked = df2.stack()
stacked
stacked.unstack()
stacked.unstack(1)
stacked.unstack(2)
df = pd.DataFrame({'A':['one','one','two','three']*3,
                   'B':['A','B','C']*4,
                   'C':['foo','foo','foo','bar','bar','bar']*2,
                   'D':np.random.randn(12),
                   'E':np.random.randn(12)})
df
pd.pivot_table(df, values='D', index=list(['A','B']), columns=list(['C']))






