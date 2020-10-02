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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **The Issue**
# 
# Child Labour is a major issue in India. The root of the problem lies in inequal wealth distribution between different classes of the country's citizens. India has been able to lift a significant percentage of it's population out of poverty but despite all the efforts more than 20% of its population still live in extreme poverty. When a family is living in extreme poverty, when all members of the family need to work to just to make the ends meet, children working for wages seems like a non issue to its members. They can't afford to realise that they are striping away their children's shot at a better life by not letting them have an education. This vicious cycle of poverty will continue for generations until the familes stop sending their children to work.

# **About data**
# 
# Sectorail distribution of child labour in India. It has 9 columns and has 21 rows. 20 rows for different states of India and the last row is an All India distribution. The first two columns of the data give us an about the various states and the categories they belong to namely 'Non Special Category' and 'Special Category'. The rest of the columns are various sectors where child labour is prevalant. This data has to be read row wise meaning, take a state and read through the data from different columns in it's row to understand the prevalancy of child labour in different sectors for that particalular state.
# 
# 

# **Exploratory data analysis**

# In[ ]:


df = pd.read_csv('/kaggle/input/child-labour-in-inida/Child Labour in India.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


print('The number of rows are {}' '\n' 'The number of columns are {}'.format(df.shape[0], df.shape[1]) )


# In[ ]:


df.drop('Total', axis = 1, inplace = True)


# In[ ]:


All_india = df[df['States'] == 'All India']
All_india


# In[ ]:


df.drop(20, axis = 0, inplace = True)


# In[ ]:


df.columns = ['Hospitality' if 'Restaurants' in col_name else 'Services' if 'Services' in col_name else col_name for col_name in df.columns]


# In[ ]:


df['Category of States'].value_counts().plot(kind = 'bar')


# In[ ]:


df['States'].unique()


# In[ ]:


df['States'].nunique()


# In[ ]:


df['Manufacturing'] = [9.9 if x == '9. 9' else float(x) for x in df['Manufacturing']]


# In[ ]:


num_cols = np.array([col for col in df.columns if df[col].dtype == 'float64'])


# In[ ]:


fig, ax = plt.subplots(3,2, figsize = (45, 24))
for x in range(3):
    for y in range(2):
        sns.barplot(x = 'States', y = num_cols.reshape(3,2)[x][y], data = df, ax = ax[x][y])
        plt.setp(ax[x][y].get_xticklabels(), rotation = 'vertical', fontsize=11)
        ax[x][y].set_ylabel(num_cols.reshape(3, 2)[x][y], fontsize = 15)


# In[ ]:


States = df['States'].values.reshape(4,5)


# In[ ]:


df.tail()


# In[ ]:


fig, ax = plt.subplots(4, 5, figsize = (35, 25))
explode = [0, 0.2, 0, 0, 0.2, 0]
for x in range(States.shape[0]):
    for y in range(States.shape[1]):
        ax[x][y].pie(x = df[df['States'] == States[x][y]][num_cols].values.tolist()[0] , labels = df[df['States'] == States[x][y]][num_cols].columns
                     , explode=explode, autopct ='%.0f%%', wedgeprops = {'linewidth': 2.0}, pctdistance = 0.8, shadow = True, startangle = 90.5)
        ax[x][y].set_xlabel(States.reshape(4,5)[x][y], fontsize = 14)
        fig.show()


# In[ ]:


fig, ax = plt.subplots(nrows = len(num_cols), figsize = (25, 35))
for col in range(len(num_cols)):
    dat = df.sort_values(by = num_cols[col], ascending = False)
    sns.barplot(x = 'States', y = num_cols[col], data = dat, ax = ax[col])
    plt.setp(ax[col].get_xticklabels(), rotation = 'vertical')


# In[ ]:


fig, ax = plt.subplots(ncols = len(num_cols), figsize = (30, 3))
sns.set_style('darkgrid')
colors = ['blue', 'red', 'yellow', 'green', 'orange', 'gold']
for col in range(len(num_cols)):
    sns.distplot(df[num_cols[col]], ax = ax[col], color = colors[col], kde = False, bins = 31)


# In[ ]:


df.head()


# In[ ]:


Mean_of_all = [round(df[x].mean()) for x in num_cols]


# In[ ]:


Mean_of_all_df = pd.DataFrame(Mean_of_all, index=num_cols)


# In[ ]:


Mean_of_all_df.plot.bar(figsize = (12, 5))
plt.title("Mean of Child Labour in various sectors across all states")


# This Kernel is a work in progress I will continue to imporve it with time. Any suggestions or criticisim will be appricated.
# 
# Music I listened to:
# 
# Coast Modern Album - Coast Modern

# In[ ]:




