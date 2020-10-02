#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Any results you write to the current directory are saved as output.


# In[ ]:


dt_app_data = pd.read_csv('../input/googleplaystore.csv')
dt_user_review_data = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# In[ ]:


dt_app_data.shape


# In[ ]:


dt_app_data.head()


# In[ ]:


dt_app_data.info()


# In[ ]:


dt_app_data.isnull().sum()


# In[ ]:


dt_app_data.isnull().sum()


# **Category**

# In[ ]:


dt_app_data.Category.value_counts()


# In[ ]:



# Delete the row with Categroy value as 1.9  , The complete row has mis managed data so we can remove it

# Get names of indexes for which column Age has value 30
indexNames = dt_app_data[ dt_app_data['Category'] == '1.9' ].index
dt_app_data.loc[indexNames]
# Delete these row indexes from dataFrame
dt_app_data.drop(indexNames , inplace=True)


# In[ ]:


dt_app_data.Category.nunique()


# In[ ]:


labels = dt_app_data.Category.value_counts().index
values = dt_app_data.Category.value_counts().values
trace = go.Pie(labels=labels, values=values)
#fig = go.Figure(data=[visData],layout=layout)
iplot([trace])


# **Rating**

# Given there are 33 Unique categories that exist for this column no data cleaning is required

# In[ ]:


dt_app_data.Rating.value_counts()


# In[ ]:


print ("There are %d null values out of  %d total values " % (dt_app_data.Rating.isna().sum(), len(dt_app_data)))

print( (dt_app_data.Rating.isna().sum()/ len(dt_app_data)) *100 )


# In[ ]:


# Fill 13% Null values with mean rating values
dt_app_data['Rating'] =  dt_app_data['Rating'].fillna(value=dt_app_data['Rating'].mean())


# **Size**

# In[ ]:


dt_app_data.Size.value_counts()


# In[ ]:


#Contains M &k to signify MB & kB 
dt_app_data.Size = dt_app_data.Size.apply(lambda x: x.replace('M', '000') if 'M' in x else x)
dt_app_data.Size = dt_app_data.Size.apply(lambda x: x.replace('k','') if 'k' in str(x) else x)
dt_app_data.Size = dt_app_data.Size.apply(lambda x: x.replace('Varies with device','0') if 'Varies with device' in str(x) else x)
#dt_app_data.Size = dt_app_data.Size.apply(lambda x: float(x))


# In[ ]:


#Find the row which has non numeric value
dt_app_data[~dt_app_data.Size.str.isnumeric()]


# In[ ]:


dt_app_data.Size=pd.to_numeric(dt_app_data.Size)


# **Install**

# In[ ]:


dt_app_data.Installs.value_counts()
# + & , needs to removed and Free should be moved to Nan


# In[ ]:


# + & , needs to removed and Free should be moved to Nan
dt_app_data.Installs = dt_app_data.Installs.apply(lambda x:x.strip('+'))
dt_app_data.Installs = dt_app_data.Installs.apply(lambda x:x.replace(',',''))
dt_app_data.Installs = dt_app_data.Installs.replace('Free','0')


# In[ ]:


#Convert to Numeric 
dt_app_data.Installs = pd.to_numeric(dt_app_data.Installs)


# **Review**

# In[ ]:


#Check non numeric column
dt_app_data.Reviews.str.isnumeric().sum()


# In[ ]:


#Find the row which has non numeric value
dt_app_data[~dt_app_data.Reviews.str.isnumeric()]


# In[ ]:


#Convert to Numeric 
dt_app_data.Reviews = pd.to_numeric(dt_app_data.Reviews)


# **Type**

# In[ ]:


dt_app_data.Type.value_counts()


# The data is clean for this field so no pre processing required

# **Price**

# In[ ]:


dt_app_data.Price.value_counts()


# In[ ]:


# The Price data is clean , only $ sign needs to be removed before converting it to numeric
dt_app_data.Price = dt_app_data.Price.apply(lambda x:x.strip('$'))
dt_app_data.Price = pd.to_numeric(dt_app_data.Price)


# **Content Rating**

# In[ ]:


dt_app_data['Content Rating'].value_counts()


# In[ ]:


dt_app_data[dt_app_data['Content Rating'].isna()]


# No changes are required in this column as no repeation exist

# **Genres**

# In[ ]:


dt_app_data.Genres.value_counts()


# In[ ]:


dt_app_data.Genres.nunique()


# In[ ]:


dt_app_data['Primary_Genres'] = dt_app_data.Genres.apply(lambda x: x.split(';')[0] )
dt_app_data['Secondary_Genres'] = dt_app_data.Genres.apply(lambda x: x.split(';')[-1])


# **Last Updated**

# In[ ]:


from datetime import date,datetime
dt_app_data['Last Updated'] = pd.to_datetime(dt_app_data['Last Updated'])
dt_app_data['Last_Updated_Since'] =  dt_app_data['Last Updated'].apply(lambda x: date.today() - datetime.date(x))


# **Android Version**

# In[ ]:


dt_app_data['Android Ver'].value_counts()


# In[ ]:


x='4.1 and up'
y='4.0.3 - 7.1.1'
z='4.4W and up'

a=x.split('and')
b=y.split('-')

a = [i.strip(' ') for i in a]
a[0]


# In[ ]:



dt_app_data['Android_Base_Ver']=dt_app_data['Android Ver'].apply(lambda x:str(x).split(' and ')[0].split(' - ')[0])
dt_app_data['Android_Base_Ver']=dt_app_data['Android_Base_Ver'].replace('4.4W','4.4')
dt_app_data['Android_Last_Ver']=dt_app_data['Android Ver'].apply(lambda x:str(x).split(' and ')[-1].split(' - ')[-1])


# In[ ]:


dt_app_data['Android_Base_Ver']


# **Current Version**

# In[ ]:


dt_app_data['Current Ver'].value_counts()


# In[ ]:


dt_app_data['Current Ver'].isna().sum()


# **Visualization**

# In[ ]:


import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
# plotly
init_notebook_mode(connected=True)


# In[ ]:


labels = dt_app_data.Category.value_counts().index
values = dt_app_data.Category.value_counts().values
trace = go.Pie(labels=labels, values=values)
#fig = go.Figure(data=[visData],layout=layout)
iplot([trace])


# **Category distribution w.r.t Rating**

# In[ ]:


group_category_rating = dt_app_data.groupby('Category')['Rating'].mean()
#group_category_rating.sort_values()[-10:].plot(kind='boxplot');
sns.boxplot(x = group_category_rating.values, y = group_category_rating.index )


# In[ ]:


trace = [go.Histogram(x = dt_app_data.Rating, xbins = {'start': 1, 'size': 0.1, 'end' :5})]
iplot(trace, filename='overall_rating_distribution')


# In[ ]:


sns.jointplot(data = dt_app_data , x= 'Rating' , y= 'Price');


# In[ ]:


dt_app_data.info()


# In[ ]:




