#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")


# Top 5 data of our table .

# In[ ]:


data .head()


# the last 5 data of our table .

# In[ ]:


data .tail()


# Let's examine a few details about the table now .

# In[ ]:


data .columns


# data types of the headings of our table .

# In[ ]:


data .info()


# numerical data

# In[ ]:


data.describe()


# movies and tv show numbers

# In[ ]:


import matplotlib.pyplot as plt
print(data['type'].value_counts(dropna =False))


# the number of filled data in the table.

# In[ ]:


data.count ()


# blank data count.

# In[ ]:


data.isnull (). sum ()


# In[ ]:


monitoring_studies = [4265,1969]
headlines = ['Movie','TV Show']
colors = ['g','y']

plt.pie(monitoring_studies,
labels=headlines,
colors=colors,
  startangle=90,
  shadow= True,
  explode=(0,0),
  autopct='%1.1f%%')
 
plt.title('percentiles')
plt.show()


# Let's look at the names that are most and least directed.

# In[ ]:


print(data['director'].value_counts(dropna =False) )


# Let's look at the countries now.
# 
# NOTE = Let's do this time (dropna = True) in order not to see the parts whose country is not written.

# In[ ]:


data['country'].value_counts(dropna =True)


# Now let's graph our data.

# In[ ]:


objects = ("United States","India","United Kingdom","Japan","Canada")
y_pos = np.arange(len(objects))
performance =[2032,777,348,176,141]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('country')
plt.ylabel('number of movies and tv show')
plt.title('countries with the most movies and tv show')

plt.show()


# 
# If you want, we can sort it as (movie name-data type-movie type)

# In[ ]:


data_new = data.head() 
#Although this method of use is not preferred ...

melted = pd.melt(frame=data_new,id_vars = 'title', value_vars= ['type'])

melted


# 
# movie actors director ranking

# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'title', value_vars= ['director','cast'])

melted.pivot(index = 'title', columns = 'variable',values='value')


# first 5 and last 5 columns.

# In[ ]:


data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)
conc_data_row

