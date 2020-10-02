#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


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


# In[ ]:


df=pd.read_excel('/kaggle/input/junglee-mukeshk/101 Pool Game Case Study - Data Set_withAnalysis.xlsx',sheet_name='Sheet_1_crosstab (1)')


# In[ ]:


df.head(3)


# In[ ]:


df.columns


# In[ ]:


del(df[143546424.25])

df.columns


# In[ ]:


list(df.columns.values)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


len(df['Date'].unique())


# In[ ]:


df


# In[ ]:



df.count(axis=0)


# columns is equal to "0"

# 

# In[ ]:


df.isna().sum()


# In[ ]:


df


# In[ ]:


df['full_count'] = df.apply(lambda x: x.count(), axis=1)
df


# In[ ]:


df['full_count'].unique()


# In[ ]:


df.isnull().sum(axis=0)


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df['Date'].head()


# In[ ]:


min(df['Date'])


# In[ ]:


max(df['Date'])


# In[ ]:


uCount = len(df['Date'].unique())

fullCount = len(df['Date'])

remainder = fullCount % uCount

print(remainder)


# In[ ]:


df.groupby(['Date']).count()


# In[ ]:


df['Date'].unique()


# In[ ]:


df['Date'].value_counts().describe()


# In[ ]:


date_str = 'June 1, 2018'
date_str_array = date_str.split()
date_str_array[1] = date_str_array[1].replace(',','')

print(date_str)
print(date_str_array[0])
print(date_str_array[1])
print(date_str_array[2])


# In[ ]:


df['Month'] = df['Date'].apply(lambda x:x.split(' ')[0] + ' ' + x.split(' ')[2])
df


# In[ ]:


df['Month'].value_counts().sort_values(ascending=False)


# In[ ]:


df['Month'] = df['Date'].apply(lambda x:x.split(' ')[0])
df


# In[ ]:


df['Day'] = df['Date'].apply(lambda x:x.split(' ')[1].replace(',',''))
df


# In[ ]:


df['Year'] = df['Date'].apply(lambda x:x.split(' ')[2])
df


# In[ ]:





# In[ ]:


df['Seat'].head()


# In[ ]:


df['Seat'].value_counts()


# In[ ]:


values=df['Seat'].unique()
print(values)


# In[ ]:


df['Seat'].isna().value_counts()


# In[ ]:


min(df['Seat'])


# In[ ]:


max(df['Seat'])


# In[ ]:


df['Seat'].describe()


# In[ ]:


pd.value_counts(df['Seat']).plot.bar()


# In[ ]:


pd.value_counts(df['Seat']).plot.pie()


# In[ ]:


df['Seat'].plot()


# In[ ]:


df['Seat'].hist(bins=7)


# In[ ]:


df.head()


# In[ ]:


columns=['Month','full_count']
df = df.drop(columns,axis=1)
df.head()      


# In[ ]:


df.head()


# In[ ]:


df['# Users'].count()


# 

# In[ ]:


max(df["# Users"])


# In[ ]:


min(df["# Users"])


# In[ ]:


df['# Users'].describe()


# In[ ]:


len(df['# Users'].unique())


# In[ ]:


s=df.groupby('# Users')
s.first()


# In[ ]:


s=df["# Users"].duplicated(keep='last').head()
s


# In[ ]:


df["Duplicated"]=df.duplicated()
df


# In[ ]:


df['santhosh']="a"
df


# In[ ]:


df['Composition'].head()


# In[ ]:


max(df['Composition'])


# In[ ]:


min(df['Composition'])


# In[ ]:


df['Composition'].describe()


# In[ ]:


df['Composition'].value_counts()


# In[ ]:


df['Composition']=='2'


# In[ ]:


df['CONFIGURATION'].describe()


# In[ ]:


df['CONFIGURATION'].max()


# In[ ]:


df['CONFIGURATION'].value_counts()


# In[ ]:


seat_mean= df.Seat.mean()
print("seat mean : " ,seat_mean)


# In[ ]:


df.groupby(['CONFIGURATION', 'Wager']).size()


# In[ ]:


df.groupby('CONFIGURATION',as_index=False).agg({"Wager": "sum"})


# In[ ]:


df.groupby('CONFIGURATION', as_index=False).agg({"Wager": "sum"}).max()


# In[ ]:


df.groupby('CONFIGURATION', as_index=False).agg({"Wager": "sum"}).min()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].sum().reset_index().max()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].sum().reset_index()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].sum()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].sum().reset_index().min()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].count().reset_index()


# In[ ]:


df.groupby(['CONFIGURATION'])['Wager'].count()


# In[ ]:


df.size


# In[ ]:


df.shape


# In[ ]:


df


# In[ ]:


df.dropna(how='all')


# In[ ]:


sum=df.groupby(['Date'])['Rake (Profit)'].sum()
sum


# In[ ]:


df.groupby(['Date'])['Rake (Profit)'].max()


# In[ ]:


date_str = 'June 1, 2018'
date_str_array = date_str.split()
date_str_array[1] = date_str_array[1].replace(',','')


# In[ ]:


df


# In[ ]:


print datetime.strptime("%B %d,%Y","July 1, 2018").strftime("%m/%d/%y")


# In[ ]:


s = "July 1"
print(datetime.datetime.strptime("{0} 2018".format(s), "%B %d %Y").strftime("%d-%m-%Y"))


# In[ ]:


df['day'] = pd.to_datetime(df.Date, format='%m/%d/%Y', errors='coerce')
df


# In[ ]:


del df['Day']


# In[ ]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




