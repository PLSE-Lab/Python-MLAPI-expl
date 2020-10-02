# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

df=pd.read_csv(r'../input/BreadBasket_DMS.csv')

df['datetime'] = pd.to_datetime(df['Date']+" "+df['Time'])
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Weekday'] = df['datetime'].dt.weekday
df['Hours'] = df['datetime'].dt.hour

df1=df[['Transaction', 'Month', 'Year', 'Weekday','Hours']]

df1.drop_duplicates(inplace=True)

# we can see from here is that maximum transaction happened on weekday 5 i.e. Saturday. We can pick this day to launch new product or increase our stokc or offer a discount plan
sns.countplot(x='Weekday',data=df1)


# Similarly we can view the month with maximum transactions. 11 i.e. November. We can pick this day to launch new product or increase our stokc or offer a discount plan
sns.countplot(x='Month',data=df1)

#similarly for year
sns.countplot(x='Year',data=df1)

# we have more sales during day between 9 am to 3pm.
sns.countplot(x='Hours',data=df1)



















