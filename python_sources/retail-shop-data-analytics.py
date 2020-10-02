#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data Cleansing 

# In[ ]:


Transactions = pd.read_csv('..//input/retail-shop-case-study-dataset/Transactions.csv')
Transactions.head()


# In[ ]:


Transactions.dtypes


# In[ ]:


Customer = pd.read_csv('..//input/retail-shop-case-study-dataset/Customer.csv')
Customer.head()


# In[ ]:


Cat = pd.read_csv('..//input/retail-shop-case-study-dataset/prod_cat_info.csv')
Cat.head(50)


# There is a clear need to create a new table prod_sub_cat_code table as well as a new cat table

# In[ ]:


Sub_Cat = Cat[['prod_sub_cat_code', 'prod_subcat']]
Sub_Cat.head()


# In[ ]:


Cat_New = Cat[['prod_cat_code', 'prod_cat']]
Cat_New.head()



#df.drop(df.columns[[0,1,3]], axis=1, inplace=True)


# In[ ]:


Trans = pd.merge(Transactions, Cat_New, how = 'outer')

Trans.head()


# In[ ]:


Trans = pd.merge(Transactions, Sub_Cat, how = 'outer', left_on="prod_subcat_code", right_on="prod_sub_cat_code")
Trans.head()


# In[ ]:


Trans.head()


# In[ ]:


Trans = pd.merge(Transactions, Customer,  left_on="cust_id", right_on="customer_Id")
Trans.head()


# In[ ]:



Trans = Trans.drop(['prod_subcat_code', 'prod_cat_code','customer_Id'] , 1) #drop cust_id column
Trans.head()


# If you have a look at the data, you will noticed that the data in Qty, Rate and total_amt columns are negatives. We need to transform them to positive values using the abs () function.

# In[ ]:


Trans.Qty = Trans.Qty.abs()
Trans.Rate = Trans.Rate.abs()
Trans.total_amt = Trans.total_amt.abs()
Trans.head()


# We have the customer's DOB, let's make any attempt to get their Age

# In[ ]:


def calculate_age(born):
    born = datetime.strptime(born, "%d-%m-%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

Trans['Age'] = Trans['DOB'].apply(calculate_age)
Trans.head()


# In[ ]:


Trans.info()


# ## Data Analysis and Visualization 
# 

# Let's look for unique values in the dimensions. This will help our analysis

# In[ ]:


print("# unique values in STore Type: {0}".format(len(Trans['Store_type'].unique().tolist())))
print("# unique values in City: {0}".format(len(Trans['city_code'].unique().tolist())))
print("# unique values in Customer : {0}".format(len(Trans['cust_id'].unique().tolist())))
print("# unique values in Age: {0}".format(len(Trans['Age'].unique().tolist())))


# Let's see the Store Types people tends to use to purchase products 

# In[ ]:


labels = Trans['Store_type'].value_counts().index
sizes = Trans['Store_type'].value_counts().values
explode = (0, 0.1, 0.2, 0.1)  # only "explode" the 2nd and 3rd slices (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title("Retail Shop Store Types")

plt.show()


# Does this mean that e-shop will have the highest revenue among other channels in this dataset? Let's have a look

# In[ ]:



Trans.groupby('Store_type').sum()['total_amt'].plot.bar()
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Let's see which of the city has the most customers as well as their spending power
# 

# In[ ]:


sns.set(style="darkgrid")       #style the plot background to become a grid
sns.countplot(y = 'city_code', data=Trans, hue = 'Gender', order = Trans['city_code'].value_counts().index )


# Let's see the top 5 customers who made the highest number of purchase in the data set

# In[ ]:


top_customer = Trans.cust_id.value_counts()
top_customer[:5].plot(kind='barh')
#top_customer.sort()top_customer[-5:].plot(kind='barh')


# In[ ]:




