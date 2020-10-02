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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#read the csv file and make the data frame 
zomato_df = pd.read_csv('../input/zomato.csv')


# In[ ]:


#display the data frame
zomato_df


# In[ ]:


#display the first 5 rows
zomato_df.head()


# In[ ]:


#display the last five rows
zomato_df.tail()


# In[ ]:


#display the how many rows and columns are there
print("the data frame has {} rows and {} columns".format(zomato_df.shape[0],zomato_df.shape[1]))


# In[ ]:


#display the data types of columns
zomato_df.dtypes


# In[ ]:


#display the size of data frame
print("the size of data frame is {}".format(zomato_df.size))


# In[ ]:


#dispaly the information of data frame
zomato_df.info()


# so from above we can see that some columns have null values

# In[ ]:


#display null values for each column
zomato_df.apply(lambda x:sum(x.isnull()))


# so from above we can see dish_liked has more null values

# In[ ]:


#display null values for each column graphically
sns.heatmap(zomato_df.isnull(),cbar=0)
plt.show()


# so we can see from above heatmap that dish_liked has more null values

# In[ ]:


#make the copy of data frame and apply changes to the new data frame
new_zomato_df = zomato_df.copy()


# In[ ]:


#so approx_cost is object type we have to make it integer type and remove ','
import re
def align_proper(x):
    if (re.search(',',x)):
        return(x.replace(',',''))
    else:
        return x
new_zomato_df['approx_cost(for two people)'] = new_zomato_df[new_zomato_df['approx_cost(for two people)'].notnull()]['approx_cost(for two people)'].apply(align_proper).astype('int')


# In[ ]:


#fill null values of approx_cost with mean value of approx_cost
new_zomato_df['approx_cost(for two people)'] = new_zomato_df['approx_cost(for two people)'].fillna(value=np.mean(new_zomato_df['approx_cost(for two people)']))


# In[ ]:


new_zomato_df['approx_cost(for two people)'] = new_zomato_df['approx_cost(for two people)'].astype('int')


# In[ ]:


#display the list of restaurant whose approx_cost(for two people) is minimum
new_zomato_df[new_zomato_df['approx_cost(for two people)']==new_zomato_df['approx_cost(for two people)'].min()]


# In[ ]:


#display the list of restaurant whose approx_cost(for two people) is maximum
new_zomato_df[new_zomato_df['approx_cost(for two people)']==new_zomato_df['approx_cost(for two people)'].max()]


# In[ ]:


#display 5 number summary of approx_cost(for two people)
new_zomato_df['approx_cost(for two people)'].describe()


# In[ ]:


#display graphically how approx_cost(for two people) distributed
sns.distplot(new_zomato_df['approx_cost(for two people)'],kde=0)
plt.show()


# In[ ]:


#display which location has more restaurants
new_zomato_df['listed_in(city)'].value_counts()


# so from above we can see BTM has maximum number of restaurants

# In[ ]:


#display which type of restaurants are more
new_zomato_df['listed_in(type)'].value_counts()


# In[ ]:


chart =sns.countplot(new_zomato_df['listed_in(type)'])
chart.set_xticklabels(chart.get_xticklabels(),rotation=30)


# so from above we can see delivery type of restaurants are more and pubs and bars are least

# In[ ]:


#display how many restaurants with online order 
new_zomato_df['online_order'].value_counts()


# so from above we can see 30444 restaurants with online order

# In[ ]:


#display which type of restaurant having online order option or not
chart =sns.countplot(new_zomato_df['listed_in(type)'],hue=new_zomato_df['online_order'])
chart.set_xticklabels(chart.get_xticklabels(),rotation=30)


# In[ ]:


pd.crosstab(new_zomato_df['listed_in(type)'],new_zomato_df['online_order'])


# so from above we can see that 354 buffet type restaurants having online order facility while 528 buffet type restaurant not having online order facility
