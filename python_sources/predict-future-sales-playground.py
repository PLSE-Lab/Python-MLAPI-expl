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


# > ### Descrition of this competition:
# 
# |Data               |fields                                                                     |
# |-------------------|---------------------------------------------------------------------------|
# |ID                 |an Id that represents a (Shop, Item) tuple within the test set             |
# |shop_id            |unique identifier of a shop                                                |
# |item_id            |unique identifier of a product                                             |
# |item_category_id   |unique identifier of item category                                         |
# |item_cnt_day       |number of products sold.You are predicting a monthly amount of this measure|
# |item_price         |current price of an item                                                   |
# |date               |date in format dd/mm/yyyy                                                  |
# |date_block_num     |a consecutive month number, used for convenience. January 2013 is 0,       |
# |                   | February 2013 is 1,..., October 2015 is 33                                |
# |item_name          |name of item                                                               |
# |shop_name          |name of shop                                                               |
# |item_category_name |name of item category                                                      |

# <h2 style="color:blue" align="left"> 1. Importing Libraries </h2>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
import gc


# <h2 style="color:blue" align="left"> 2. Import Datasets </h2>

# In[ ]:


train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
item_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
Shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")


# In[ ]:


display(train.head())
display(test.head())
display(items.head())
display(item_cat.head())
display(Shops.head())
display(submission.head())


# In[ ]:


train = pd.merge(train, items, on='item_id', how='inner')
train = pd.merge(train, item_cat, on='item_category_id', how='inner')
train = pd.merge(train, Shops, on='shop_id', how='inner')

test = pd.merge(test, items, on='item_id', how='inner')
test = pd.merge(test, item_cat, on='item_category_id', how='inner')
test = pd.merge(test, Shops, on='shop_id', how='inner')


# In[ ]:


display(train.head())
display(test.head())


# In[ ]:


display(train.shape)
display(test.shape)


# In[ ]:


display(train.info())
display(test.info())


# In[ ]:


display(train.dtypes)
display(test.dtypes)


# In[ ]:


display(train.count())
display(test.count())


# In[ ]:


print('Number of duplicates in train:', len(train[train.duplicated()]))
print('Number of duplicates in test:', len(test[test.duplicated()]))


# In[ ]:


train.describe()


# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(train.corr(), annot=True, cbar=False, cmap='coolwarm')


# ### 1. Items

# In[ ]:


items.head()


# In[ ]:


items.shape


# In[ ]:


items.dtypes


# In[ ]:


items.count()


# In[ ]:


print('Number of Duplicates in item:', len(items[items.duplicated()]))


# In[ ]:


print('Unique item names:', len(items['item_name'].unique()))


# In[ ]:


items.item_id.nunique()


# In[ ]:


items.item_category_id.nunique()


# In[ ]:


plt.figure(figsize=(18,18))
items.groupby('item_category_id')['item_id'].size().plot.barh(rot=0)
plt.title('Number of items related to different categories')
plt.xlabel('Categories')
plt.ylabel('Number of items');


# In[ ]:


items.groupby('item_category_id')['item_id'].size().mean()


# In[ ]:


items.groupby('item_category_id')['item_id'].size().max()


# In[ ]:


items.groupby('item_category_id')['item_id'].size().min()


# In[ ]:


item_cat[item_cat['item_category_id'].isin(items.groupby('item_category_id')['item_id'].size().nlargest(5).index)]


# In[ ]:


item_cat[item_cat['item_category_id']                .isin((items.groupby('item_category_id')['item_id'].size()[items.groupby('item_category_id')['item_id'].size()==1])                      .index)]


# In[ ]:


plt.figure(figsize=(10, 5))
sns.distplot(train['item_id'], color="red");


# In[ ]:


plt.figure(figsize=(10, 5))
sns.distplot(train['item_price'], color="red");


# In[ ]:


plt.figure(figsize=(10, 5))
sns.distplot(np.log(train['item_price']), color="red");


# #### 2. Item Category

# In[ ]:


item_cat.head()


# In[ ]:


item_cat.shape


# In[ ]:


item_cat.dtypes


# In[ ]:


item_cat.count()


# In[ ]:


print('Number of Duplicates in item_cat:', len(item_cat[item_cat.duplicated()]))


# In[ ]:


print('Unique item names:', len(item_cat['item_category_id'].unique()))


# In[ ]:


item_cat['item_category_id'].nunique()


# In[ ]:


item_cat['item_category_id'].values


# ### 3. Shop

# In[ ]:


Shops.head()


# In[ ]:


Shops.shape


# In[ ]:


Shops.dtypes


# In[ ]:


Shops.count()


# In[ ]:


color = sns.color_palette("hls", 8)
sns.set(style="darkgrid")
plt.figure(figsize=(15, 5))
sns.countplot(x=train['shop_id'], data=train, palette=color)


# In[ ]:


print("Percentual of total sold by each Shop")
print((train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25) / train.groupby('shop_name')['item_cnt_day'].sum().sum() * 100)[:5])

train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25).iplot(kind='bar',
                                                                    title='TOP 25 Shop Name by Total Amount Sold',
                                                    xTitle='Shop Names', 
                                                                       yTitle='Total Sold')


# ### Missing Data

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


items.isnull().sum()


# In[ ]:


item_cat.isnull().sum()


# In[ ]:


Shops.isnull().sum()


# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(x='item_price', data=train)


# In[ ]:


plt.hist(x='item_price')


# <h2 style="color:green" align="left"> 5. Data Visualization </h2>
# 
# - Used below **visualisation libraries**
# 
#      1. Matplotlib
#      2. Seaborn (statistical data visualization)
#      
#      
# ### 1. Univariate Analysis
# 
# - Univariate Analysis : data consists of **only one variable (only x value)**.

# In[ ]:


train.item_cnt_day.plot()
plt.title("Number of products sold per day");


# In[ ]:


train.item_price.hist()
plt.title("Item Price Distribution");


# ### 2. Bivariate Analysis
# 
# - **Bivariate Analysis** : data involves **two different variables**.

# ### 3. Multivariate Analysis
# 
# - 1. Pair Plot

# In[ ]:


sns.pairplot(train)


# ### Outliers

# In[ ]:


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(train['Age'])

def hist_plots(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df)
    plt.title("Histogram Plot")
    plt.show()
hist_plots(train['Age'])

def scatter_plots(df1,df2):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df1,df2)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    plt.title("Scatter Plot")
    plt.show()
scatter_plots(train['Age'],train['Fare'])

def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(train['Fare'])

def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df,line='s')
    plt.title("Normal QQPlot")
    plt.show()
qq_plots(train['Fare'])


# <h2 style="color:blue" align="left"> 6. Data Preprocessing </h2>

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

