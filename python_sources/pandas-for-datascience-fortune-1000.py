#!/usr/bin/env python
# coding: utf-8

# This dataset Cotains the data of 1000 big Corporations in the world published annualy by Fortune Magazine.This is a kernel in Process and I will be updating the kernel in coming days.

# In[ ]:


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


# In[ ]:


import matplotlib.pyplot as plt


# ### Importing data with Pandas

# In[ ]:


df=pd.read_csv('../input/fortune1000/fortune1000.csv',index_col='Rank')
df.head()


# We have used rank column as our index

# ### Summary of Dataset

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# ### Attributes

# In[ ]:


df['Sector'].values


# In[ ]:


df.index


# We have used rank column as our index

# In[ ]:


df['Sector'].dtype


# ### Method

# In[ ]:


df['Revenue'].sum()


# In[ ]:


df['Employees'].mean()


# In[ ]:


df.head(2)


# In[ ]:


df.tail(2)


# ### Parameters and Arguments

# ### Filter

# In[ ]:


df[df['Sector']== 'Industrials']


# Here we have filtered all the rows based on the Sector which are all industrials

# In[ ]:


df[df['Sector']!= 'Industrials']


# We have used not equal to operator to get info of all the sectors other than Industrials

# In[ ]:


df[df['Revenue']>50000]


# Now we have got the details of the companies whos revenue is greater than 50000.

# ### Groupby

# In[ ]:


sectors=df.groupby("Sector")
sectors


# ### Type

# In[ ]:


type(sectors)


# In[ ]:


sectors


# ### Number of Groupings

# In[ ]:


len(sectors)


# There are 21 industry sectors in the dataset

# ### Other method to count the number of Sectors

# In[ ]:


df.nunique()


# In[ ]:


df['Sector'].nunique()


# ### Sizing the Grouping

# In[ ]:


sectors.size().sort_values(ascending=False).plot(kind='bar');


# So here we have go the information of number of companies in each sector.For companies are 139 are there in financial sector and least 15 nos being in apparel sector.

# ### Extrating first row of every Sector

# In[ ]:


sectors.first()


# ### Getting the information of Last row of all sectors

# In[ ]:


sectors.last()


# Now we have information on the last row of all the sectors

# ### Groups on Groupby

# In[ ]:


sectors.groups


# In[ ]:


df.loc[24]


# Here we have got the index number of all the rows in each group.We can see that the index 2 falls in the Aerospace and Defense.

# ### Get Group

# In[ ]:


sectors.get_group('Energy')


# Using the get_group command we are able to get the rows values in a particular sectors.

# ### Max

# In[ ]:


sectors.max()


# We can see that we have got the company arranged alphabetically.W is max alphebatically so we can see company like Woodward getting reported.

# ### Min

# In[ ]:


sectors.min()


# We can see that we have got the company arranged alphabetically.A is min alphebatically so we can see company like B/E Aerospace getting reported.

# ### Sum

# In[ ]:


sectors.sum()


# So for numerical columns we have got the sum of the Revenue,Profits and Employees for each sector

# ### Mean

# In[ ]:


sectors.mean()


# So for numerical columns we have got the mean of the Revenue,Profits and Employees for each sector

# In[ ]:


sectors['Employees'].sum()


# In[ ]:


sectors[['Profits','Employees']].sum()


# ### Grouping by Multiple Columns

# In[ ]:


sector=df.groupby(['Sector','Industry'])
sector.size()


# Here in addition to sector our data is grouped by industry too.

# ### Sum of Multi Groupby

# In[ ]:


sector.sum()


# In[ ]:


sector['Revenue'].sum()


# In[ ]:


sector['Employees'].mean()


# We can do the sum,max,min,mean etc operation on a list of columns as shown above.

# ### Add Method

# In[ ]:


sectors.agg({'Revenue':'sum','Profits':'sum','Employees':'mean'})


# By using a dictionary on groupy by we can have prefered operation on each column usiing agg method

# ### Multiple operations to Multiple Columns

# In[ ]:


sectors.agg(['size','sum','mean'])


# So using a list we have applied multiple operations on each column

# ### Iterating through Groups

# In[ ]:


df1=pd.DataFrame(columns=df.columns)
df1


# In[ ]:


for sector, data in sectors:
    highest_revenue_company_in_group=data.nlargest(1,'Revenue')
    df1=df1.append(highest_revenue_company_in_group)
df1    


# So with the above code we are able to get the details of the company with the highest revenue in each sector.

# In[ ]:


cities=df.groupby('Location')
df2=pd.DataFrame(columns=df.columns)
df2


# In[ ]:


for city, data in cities:
    highest_revenue_company_in_group=data.nlargest(1,'Revenue')
    df2=df2.append(highest_revenue_company_in_group)
df2


# Here we are able to get the highest revenue company in each city.

# ### Between Method

# In[ ]:


df[df['Revenue'].between(25000,50000)]


# So with Between command we were able to list down the companies which have revenue between 25000 t0 50000.Between command can also be used for datatime object.

# ### Precision

# In[ ]:


pd.get_option('precision')


# In[ ]:


pd.get_option('precision',2)


# In[ ]:


pd.reset_option('precision')


# In[ ]:


pd.get_option('precision')


# We have changed the precision to two decimels and then reset the decimel to six.

# ### Pandas Options with Attributes and Dot Syntax

# In[ ]:


pd.options.display.max_rows


# When we request a display of DataFrame 60 rows will be getting displayed

# In[ ]:


pd.options.display.max_rows=4


# In[ ]:


df


# We have reduced the maximum number of rows on display to be 4

# ### Changing Pandas Options with Methods

# In[ ]:


pd.get_option('max_rows')


# In[ ]:


pd.get_option('max_columns')


# In[ ]:


pd.set_option('max_rows',6)
df


# In[ ]:


pd.set_option('max_columns',3)
df


# In[ ]:


pd.pandas.set_option('display.max_columns',None)
df


# **TO BE CONTINUED**

# In[ ]:




