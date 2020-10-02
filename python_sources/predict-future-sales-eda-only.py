#!/usr/bin/env python
# coding: utf-8

# Predict Future Sales - EDA

# ## Libraries

# The first step is to load all the required libraries and load raw data files into memory.

# ### Scientific Libraries

# In[ ]:


import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv, Series, DataFrame, to_datetime

import pickle

import warnings
warnings.filterwarnings('ignore')


# ### Machine Learning Libraries

# In[ ]:


import sklearn as sklearn
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# ### Visualization Library

# #### `Matplotlib`

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, plot, scatter, show, title, xlabel, ylabel, xticks
get_ipython().run_line_magic('matplotlib', 'inline')


# #### `Pandas` Options

# In[ ]:


pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
pd.plotting.backend='hvplot'


# #### `Seaborn` Options

# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(20, 5)})
from seaborn import set_context, barplot, boxplot


# #### `palettable`

# In[ ]:


from palettable.colorbrewer.qualitative import Pastel1_7, Pastel1_8, Pastel1_9


# #### `plotly`

# In[ ]:


from plotly.graph_objects import Figure, Pie


# ### Version Control

# In[ ]:


get_ipython().system('pip install googletrans')


# In[ ]:


for p in [mpl, np, pd, sklearn, lgb, sns]:
    print (p.__name__, p.__version__)


# ## Functions

# We import the functions we have developed for this assignment.

# In[ ]:


def translate_to_english(word):

    from googletrans import Translator
    translator = Translator()
    translated = translator.translate(word)
    
    return translated.text


# ## Load raw data

# Let's load the data from the hard drive first.

# In[ ]:


raw_dir = '../input/competitive-data-science-predict-future-sales/' 
sales = read_csv(raw_dir+'sales_train.csv')
shops = read_csv(raw_dir+'shops.csv')
items = read_csv(raw_dir+'items.csv')
item_cats = read_csv(raw_dir+'item_categories.csv')
test = read_csv(raw_dir+'test.csv')


# # Exploratory Data Analysis (EDA)

# ## Sales
# 
# This DataFrame consists of 
# - `date`, 
# - `date_block_num`, 
# - `shop_id`, 
# - `item_id`, 
# - `item_price`	and 
# - `item_cnt_day`

# In[ ]:


sales.sample(8).sort_index()


# In[ ]:


from datetime import datetime as dt
try:
    sales.date = sales.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y'))
except:
    pass


# ### Proxy column (helper variable)
# #### `year-month` 

# In[ ]:


sales['year_month'] = to_datetime(sales['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month= '{:02d}'.format(x.month)))


# #### `date_block_num`

# In[ ]:


sales.sample(8).sort_index().sort_values('date_block_num')


# `date_block_num` is nothing more than the order number of `year_month` combination:
# 
# - starting from `0` at `2013-01` 
# - to `33` at `2015-10`

# ### Descriptive Statistics

# #### How's the date range of the sales data?

# In[ ]:


sales.date.min(), sales.date.max()


# 3 years minus 2 months duration = 34 months
# 
# 2 years 10 months

# In[ ]:


time_range = sales.date.max() - sales.date.min() 
time_range


# In[ ]:


sales.describe()


# #### Was/ were there something unusual about the data?

# There were something unusual visible from the description above:
# 
# - negative `item_price`
# - maximum `item_price` in the order of $(exp)^{+05}$ while the mean were in the order of $(exp)^{+02}$
# 
# - negative `item_cnt_day`
# - maximum `item_cnt_day` in the order of $(exp)^{+03}$ while the mean were in the order of $(exp)^{+00}$

# #### Was there negative value in `item_price`?

# ##### Negative `item_price` removal

# In[ ]:


sales[sales.item_price < 0]


# In[ ]:


sales = sales[sales.item_price > 0]


# We preserve only those sane price which were positive.

# #### What to do with maximum value in `item_price`?

# ##### Maximum `item_price`

# In[ ]:


sales[sales.item_price == sales.item_price.max()]


# In[ ]:


boxplot(x=sales.item_price)
show()


# We have an outlier in `item_price`, it can be seen from the boxplot where the price of more than 300,000. This outlier must be removed for sanity sake. We must limit sales.item_price to be lower than the outlier price of 300,000.

# In[ ]:


sales = sales[(sales.item_price < 307980.0) & (sales.item_price > 0)]


# In[ ]:


boxplot(x=sales.item_price)
show()


# #### Was there negative values in `item_cnt_day`?

# In[ ]:


sales[sales.item_cnt_day < 0]


# We preserve only those `item_cnt_day` which were positive. There were 7356 rows of data and it's too many to delete safely so we just replace those `-1.0` with `0`

# In[ ]:


sales.item_cnt_day.mask(sales.item_cnt_day <0, 0, inplace=True)


# In[ ]:


boxplot(x=sales.item_cnt_day)
show()


# #### What to do with maximum value in `item_price`?

# we have an outlier in the `item_cnt_day`

# In[ ]:


sales.item_cnt_day.max()


# In[ ]:


sales = sales[sales["item_cnt_day"] < 2000]


# after removal of the outlier

# In[ ]:


boxplot(x=sales.item_cnt_day)
show()


# #### How many unique Shop IDs were in the data?

# In[ ]:


len(sales['shop_id'].unique())


# There are 60 unique shop ids in the data of which are:

# In[ ]:


unique_shops = array(sorted(sales['shop_id'].unique()))
unique_shops, len(unique_shops)


# In[ ]:


pie(unique_shops,
    labels=unique_shops,
    labeldistance=1.0,
    colors=Pastel1_9.hex_colors,
    textprops={'fontsize': 8},
    rotatelabels = True,
    startangle=-90
   )
title('60 SHOP IDs')
show()


# #### How many Shop IDs selling for each `block_date_num`?
# 
# Counting unique number of shops selling in each block date.

# In[ ]:


_ = DataFrame(sales.groupby('date_block_num')['shop_id'].nunique())
_.sample(8).sort_index()


# In[ ]:


set_context("talk", font_scale=1.1)
barplot(
    data = _,
    x = _.index,
    y = _.shop_id
);
plot(_.shop_id)

title('\nNUMBER SHOPS SELLING per DATE BLOCK\n')
xlabel('\nDATE BLOCK')
ylabel('n of SHOPs-SELLING\n')
xticks(rotation=75, fontsize='xx-small')

_.plot()

del _
show()


# ## Sales Fluctuation

# ### How was the sales for each Date Block?
# 
# Items Sold per date block

# In[ ]:


_ = DataFrame(sales.groupby(['date_block_num']).sum().item_cnt_day).reset_index()
_.head(8)


# In[ ]:


set_context("talk", font_scale=1.0)
barplot(
    data = _,
    x = 'date_block_num', 
    y = 'item_cnt_day'
);
plot(_.item_cnt_day)
title('\nNUMBER OF ITEMS SOLD per DATE BLOCK\n')
xlabel('\nDATE BLOCK')
ylabel('ITEMS SOLD\n')
xticks(rotation=75, fontsize='x-small' )
_.plot()
title('\nNUMBER OF ITEMS SOLD per DATE BLOCK\n')
xlabel('\nDATE BLOCK')
ylabel('ITEMS SOLD\n')
xticks(rotation=75, fontsize='x-small' )
del _
show()


# ### How was the sales for each weekday?

# In[ ]:


sales["weekday"] = sales.date.apply(lambda x: x.weekday())
sales.groupby("date")["item_cnt_day"].sum().plot(figsize=(15,7))
show()


# In[ ]:


_ = DataFrame(sales.groupby("weekday")["item_cnt_day"].sum().sort_index())
week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_.index = _.index.to_series().apply(lambda x: week[x])
_


# In[ ]:


title('\nSALES PER WEEKDAY\n')
xlabel('\nWeekday')
ylabel('Items Sold\n')
xticks(rotation=75, fontsize='x-small' )

barplot(
    data = _,
    x = _.index,
    y = _.item_cnt_day
)
_.plot()

title('\nSALES PER WEEKDAY\n')
xticks(rotation=75, fontsize='x-small' )
#xlabel('\nWeekday')
ylabel('Items Sold\n')

show()


# In[ ]:


pie(_.item_cnt_day,
    labels=_.index, 
    colors=Pastel1_7.hex_colors)
del _
show()


# Items were sold mostly on Saturday and least on Monday.

# ### How was the sales for each month?
# 

# In[ ]:


_ = DataFrame(sales.groupby('year_month')['item_cnt_day'].sum())
_


# In[ ]:


title('\nSALES PER MONTH\n')
xticks(rotation=75, fontsize='x-small' )
xlabel('\nMONTH of YEAR')
ylabel('Items Sold\n')

barplot(
    data = _,
    x = _.index,
    y = _.item_cnt_day
       )
plot(_.item_cnt_day)

_.plot()

title('\nSALES PER MONTH\n')
xticks(rotation=75, fontsize='x-small' )
xlabel('\nMONTH of YEAR')
ylabel('Items Sold\n')
del _
show()


# It can be seen from the plot that the number of sold items peaked every December of each year.

# ### How was the sales for each Shop ID?

# In[ ]:


_ = DataFrame(sales.groupby(['shop_id']).sum()['item_cnt_day'])
_ = _.sort_values('item_cnt_day', ascending=False)
_.head(8)


# In[ ]:


set_context("talk", font_scale=1.1)
chart = barplot(
    data = _,
    x = _.index, 
    y = _.item_cnt_day
);

title('\nITEMS SOLD per SHOP IDs\n')
xlabel('\nSHOP ID #')
ylabel('ITEMS SOLD\n')
xticks(rotation=60, fontsize='xx-small' )

_.sort_index().plot()
title('\nITEMS SOLD per SHOP IDs\n')
xlabel('\nSHOP ID #')
ylabel('ITEMS SOLD\n')
del _
xticks(rotation=60, fontsize='xx-small' )
show()


# ### How was the sales for each item ID?

# In[ ]:


_ = DataFrame(sales.groupby(['item_id']).sum()['item_cnt_day'])
_ = _.sort_values('item_cnt_day', ascending=False)
_.head(8)


# In[ ]:


set_context("talk", font_scale=1.4)
_.sort_index().plot();
xlabel('item id');
ylabel('sales');
show()


# #### What item was the most sold? 

# In[ ]:


max_item_id = _['item_cnt_day'].idxmax()
max_item_id


# ##### REMOVE OUTLIER

# From the visualization it seems that we have found an outlier or anomaly with the most sold items with id 20949. We will remove this product ID from `item_cnt` and `sales`.

# In[ ]:


_ = _[_.index != max_item_id]
sales = sales[sales.item_id !=max_item_id]
del max_item_id


# ##### AFTER OUTLIER REMOVAL

# In[ ]:


_.sort_index().plot()
title('ITEMS SOLD FOR EACH ITEM ID\n')
xlabel('\nITEM ID#')
ylabel('ITEMS SOLD\n')
show()


# #### What were 10 of the most sold items?

# In[ ]:


sales.groupby(['item_id']).sum().sort_values(['item_cnt_day'], ascending=False).head(10)[['item_cnt_day']]


# #### What were the least sold items?

# In[ ]:


_ = sales.groupby(['item_id']).sum()
_ = _.sort_values(['item_cnt_day'], ascending=True)[['item_cnt_day']]
_[_.item_cnt_day==1.0]


# #### When was the most recent items sold?

# In[ ]:


sales.date.max()


# #### What was/ were the most recent items sold?

# In[ ]:


DataFrame(sales[sales.date == sales.date.max()]['item_id'].unique()).head(10)


# ## Shops
# This DataFrame consists of 
# - `shop_name` and 
# - `shop_id`.

# In[ ]:


shops.sample(8).sort_index()


# ### How Many Unique Shop IDs?

# In[ ]:


unique_shops = shops['shop_id'].unique()
unique_shop_names = shops['shop_name'].unique()

if len(unique_shops) != len(unique_shop_names):
    print("There are different shop ids with the same name!")
else:
    print("There are "+str(len(unique_shops))+" unique shops and none of them have the same name!")


# ### How Many Shops Had Certain Name Lengths?

# In[ ]:


shop_name_lengths = {}
for r in shops['shop_name']:
    shop_name_lengths[r] = len(r)
shop_names = Series(shop_name_lengths).sort_values()


# In[ ]:


shop_names = DataFrame(shop_names.reset_index())
shop_names.columns=['name','name_length']
shop_names.sample(8).sort_index()


# In[ ]:


set_context('talk', font_scale=1.2)
data = shop_names.groupby(['name_length']).count()
barplot(data=data, x=data.index, y=data.name)
title('\nNUMBER OF SHOPS\nwith CERTAIN NAME LENGTH\n')
xlabel('\nNAME LENGTH (number of characters)\n')
ylabel('SHOPS\n')
show()


# In[ ]:


shop_names.hist(bins=shop_names['name_length'].nunique()+1)
title('NUMBER OF SHOPS\nwith CERTAIN NAME LENGTH\n')
xlabel('\nNAME LENGTH (number of characters)\n')
ylabel('Number of Shops\n')
show()


# Most shops have 22 characters length in their name. The longest shop name has 47 characters.

# ### What is the Longest Shop Name?

# In[ ]:


_ = shop_names.iloc[shop_names['name_length'].idxmax()]
length = int(_.name_length)
print("The longest shop name is:\n"+ _['name'] + "\n" +str(length) + " characters long")
del _


# ### Where is the City for Each Shops?

# In[ ]:


shops.shop_name.str.split().apply(lambda x: x[0]).unique()


# In[ ]:


shops['city'] = shops.shop_name.str.split().apply(lambda x: x[0].strip())
shops['city_en'] = shops.city.apply(lambda w: translate_to_english(w))


# In[ ]:


shops.sample(8).sort_index()


# so we have a `city` column and a simplified `shop_name`

# In[ ]:


shops['shop_name_en'] = shops.shop_name.apply(lambda w: translate_to_english(w))
shops.head(10)


# ### How many cities do we have?

# In[ ]:


shops.city_en.nunique()


# ### Which city has the most shop?

# In[ ]:


shops.groupby('city_en').count().sort_values('shop_id', ascending=False)


# Moscow can be seen as the city with the most shop

#  ###  How many shop ids were most of the cities had?

# In[ ]:


_ = shops.groupby('city_en').count()
_ = _.sort_values('shop_id', ascending=False).reset_index()
_ = _.groupby('shop_id').count()[['shop_name']]


# Most of the cities had only one `shop_id`

# ### Which city has the most sales?

# In[ ]:


_ = sales.merge(shops, how='left', on='shop_id')
_ = _.groupby('city_en').sum()[['item_cnt_day']]
_ = _.sort_values('item_cnt_day', ascending=False)
_.head(10)


# ## Items
# 
# This DataFrame consists of 
# - `item_name`, 
# - `item_id` and 
# - `item_category_id`

# In[ ]:


items.sample(8).sort_index()


# ### How Many Unique Item Categories in `items`?

# In[ ]:


items.item_category_id.nunique()


# ### How Many Unique Item IDs in `items`?

# In[ ]:


n_items = items.item_id.nunique()
n_items


# ### What was the most recent items sold?

# The most recent sales date was:

# In[ ]:


sales.date.max()


# ##### The Most Recent Item IDs 

# In[ ]:


_ = sales[sales.date == sales.date.max()][['item_id']]
_ = _.sort_values('item_id').groupby('item_id').count()
_


# In[ ]:


items[items.item_id.isin(_.index.to_list())]


# ### How Many unique item ids was sold recently?

# In[ ]:


len(_)


# In[ ]:


percentage = len(_)/n_items * 100
print('percentage: ' + str('{:.3}'.format(percentage)) + " % items were sold recently.")


# ### What were 10 of the most pricey items sold at the first date of sales?

# In[ ]:


sales.date.min()


# In[ ]:


sales[sales.date == sales.date.min()][['item_price','item_id']].sort_values(['item_price'], ascending=False).head(10)


# In[ ]:


_ = sales[sales.date == sales.date.min()][['item_price','item_id']]
_ = _.sort_values(['item_price'], ascending=False)
_ = _.head(10)['item_id'].to_list()
_


# In[ ]:


items[items.item_id.isin(_)][['item_name']]


# ### What were 10 of the most frequent items sold during all period?

# In[ ]:


sales.groupby(['item_id']).sum().sort_values(['item_cnt_day'], ascending=False).head(10)[['item_cnt_day']]


# taking just the index:

# In[ ]:


_ = sales.groupby(['item_id']).sum()[['item_cnt_day']]
_ = _.sort_values(['item_cnt_day'], ascending=False)
_ = _.head(10).index.to_list()
_


# In[ ]:


items[items.item_id.isin(_)][['item_name']]


# ### What were 10 of the most sold items at the beginning of 2013?

# In[ ]:


sales.date.min()


# In[ ]:


sales[sales.date == sales.date.min()].groupby(['item_id']).sum()[['item_cnt_day']].sort_values(['item_cnt_day'], ascending=False).head(10)


# In[ ]:


_ = sales[sales.date == sales.date.min()]
_ = _.groupby(['item_id']).sum()[['item_cnt_day']]
_ = _.sort_values(['item_cnt_day'], ascending=False)
_ = _.head(10).index.to_list()
_


# In[ ]:


items[items.item_id.isin(_)][['item_name']]


# ## Item Cats
# This DataFrame consists of 
# - `item_category_name` and
# - `item_category_id`

# In[ ]:


item_cats.sample(8).sort_index()


# In[ ]:


item_cats['item_category_name_en'] = item_cats['item_category_name'].apply(lambda w: translate_to_english(w))
item_cats.head(10)


# ### How Many Unique *`item_category_id`* in `item_cats`?

# In[ ]:


item_cats.item_category_id.nunique()


# ### How Many Unique *`item_category_name`* in `item_cats`?

# In[ ]:


item_cats.item_category_name.nunique()


# ### How Many Item Sold for Each Item Categories?

# In[ ]:


_ = sales.merge(items,how='left', on='item_id')
_ = _.groupby('item_category_id').item_cnt_day.sum()
_ = DataFrame(_)
_.sample(8).sort_index()


# In[ ]:


set_context("talk", font_scale=1.5)

barplot(
    data = _,
    x = _.index, 
    y = 'item_cnt_day'
);

title('\nITEM CATEGORY SOLD\n')
xlabel('\nItem Category ID')
ylabel('Items Sold\n')
xticks(rotation=85, fontsize='xx-small' )
show()

_.plot()

title('\nITEM CATEGORY SOLD\n')
xlabel('\nItem Category ID')
ylabel('Items Sold\n')
xticks(rotation=85, fontsize='xx-small' )

del _
show()


# ### Can We Split Categories into *`item_group`* and *`item_subgroup`*?

# In[ ]:


item_cats.sample(5).sort_index()


# we can create item category group and item category subgroup

# In[ ]:


cat_split = item_cats.item_category_name.str.split(" - ")

item_cats["item_group"] = cat_split.apply(lambda x:  x[0].strip())
item_cats['item_group_en'] = item_cats['item_group'].apply(lambda w: translate_to_english(w))

item_cats["item_subgroup"] = cat_split.apply(lambda x:  x[1].strip() if len(x) == 2 else x[0].strip())
item_cats['item_subgroup_en'] = item_cats['item_subgroup'].apply(lambda w: translate_to_english(w))

item_cats.sample(8).sort_index()[['item_category_id', 'item_category_name_en','item_group_en','item_subgroup_en']]


# ### How Many Category Groups Do We Have?

# In[ ]:


item_cats["item_group"].nunique()


# In[ ]:


set(item_cats["item_group_en"].values)


# ### How Many Category Subgroups Do We Have?

# In[ ]:


item_cats["item_subgroup"].nunique()


# ## All train data in one DataFrame

# In[ ]:


all_train_data = sales.merge(items, how='left', on='item_id')
all_train_data = all_train_data.merge(item_cats, how='left', on='item_category_id')
all_train_data = all_train_data.merge(shops, how='left', on='shop_id')
all_train_data.head(1)


# ##### export to csv.gz

# In[ ]:


get_ipython().system('mkdir ../working/processed')


# In[ ]:


necessary_columns = ['date','date_block_num','shop_id','item_id','item_price','item_cnt_day','year_month','weekday','item_category_id','item_group_en','item_subgroup_en','city_en']
all_train_data[necessary_columns].to_csv('../working/processed/all_train_data.csv.gz', compression='gzip')


# In[ ]:


get_ipython().system('ls -al ../working/processed/')


# ### What were 10 of the most popular groups items sold at all period?

# In[ ]:


_ = all_train_data.groupby('item_group_en').sum()[['item_cnt_day']]
_ = _.sort_values('item_cnt_day', ascending=False)
_.head(10)


# ### From the most popular item group, which shop sold the most items?

# In[ ]:


_ = all_train_data[all_train_data.item_group_en =='Movie']
_ = _.groupby('shop_id').sum()[['item_cnt_day']]
_ = _.sort_values('item_cnt_day', ascending=False).head(10)
_


# ### Which city has all item categories?

# In[ ]:


_ = all_train_data.groupby('city_en')[['item_category_id']].nunique()
_ = _.sort_values('item_category_id', ascending=False)
_


# ### How many item groups for each city?

# In[ ]:


_ = all_train_data.groupby('city_en')[['item_group']].nunique()
_ = _.sort_values('item_group', ascending=False)
_


# In[ ]:


def func(val, _):
    return _.item_group.ix[val]


# In[ ]:


labels = _.index
values = _.item_group
fig = Figure(data=[Pie(labels=labels, values=values, hole=.6)])
fig.update_layout(
    title={
        'text': "Number of Item Groups for Each City",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(hoverinfo='label+value', textinfo='label+value', textfont_size=8)
fig.show()


# ### What *`item_group`* exists in all cities?

# In[ ]:


groups = item_cats['item_group_en'].unique()
cities = shops['city_en'].unique()
print(len(groups), len(cities))


# We have 32 cities and 20 `item_group`s which means there are 32 x 20 combination of `city-item_group` to check

# In[ ]:


_ = all_train_data.groupby(['city_en','item_group_en']).count()[['item_cnt_day']]
_ = DataFrame(_.index.to_list(), columns =['city_en', 'item_group_en'])
_.sample(10).sort_index()


# In[ ]:


_ = _.groupby('item_group_en').agg({'city_en':'count'}).sort_values('city_en', ascending = False)
_


# In[ ]:


labels = _.index
values = _.city_en
fig = Figure(data=[Pie(labels=labels, values=values, hole=.6)])
fig.update_layout(
    title={
        'text': "Number of City<br>for Each Item Groups",
        'y':0.9,
        'x':0.3,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(hoverinfo='label+value', textinfo='label+value', textfont_size=8)
fig.show()


# In[ ]:


_[_.city_en == 32]


# From the above result: 
# - payment cards, 
# - books, 
# - PC Games and 
# - Programs
# 
# are the item groups that exist in all cities.

# ### What *`item_group`* exists in only one city?

# In[ ]:


_[_.city_en == 1]


# In[ ]:


for city in all_train_data[all_train_data.item_group_en.isin(_[_.city_en == 1].index.to_list())]['city_en'].unique():
    print(city)


# From the above result:
#  - Delivery of goods
#  - MAC games
#  - Android Games
#     
# are the item groups that exist only in 1 city that is: 
#  - Shop Online
#  - Digital

# ### How was the sales fluctuate for each item group?

# In[ ]:


_ = all_train_data[['year_month','item_group_en','item_cnt_day']]
_ = _.groupby(['year_month','item_group_en']).sum()
_.head(17)


# In[ ]:


plot(_.item_cnt_day.unstack())
#plt.legend(all_train_data.item_group_en.unique())
title('\nSALES for EACH ITEM GROUP\n')
xlabel('\nDATE BLOCK')
ylabel('SALES\n')
xticks(rotation=75, fontsize='xx-small')
show()


# In[ ]:




