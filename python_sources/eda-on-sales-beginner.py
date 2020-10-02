#!/usr/bin/env python
# coding: utf-8

# This is my first project for eda using the product sales records. The report will focus a lot more on discussing the type of the test data (the relationship between test and train), and take on some detail looks into some examples. Here's the outline: 
# 
# [Input data information](#Input-data-information)
# * [Original features](#Original-features)
# * [Data cleaning](#Data-cleaning)
# 
# [EDA on features](#EDA-on-training-data)
# * [Category](#Category)
# * [Shops](#Shops)
# * [Price](#Price)
# * [Sales information / item based](#Sales-information-/-item-based)
# 
# [Test data (depending on training data)](#Train-data-and-Test-data-overlapped?)
# * [Items in test data](# items-in-shops)
# * [Old items new launched in shop items](#Old-items-new-launched-in-shop-items)
# * [New items new launched in shop items](#New-items-new-launched-in-shop-items)
# * [Discussion](#Discussion)
# 
# [Special case](#Input-data-information)
# * [Negative price](#Negative-price)
# * [Negative sales](#Negative-sales)
# 
# [Conclusion](#Input-data-information)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# # Input data information

# ## Original features

# In[ ]:


shops = pd.read_csv("../input/shops.csv")  # shop_name, shop_id
item_categories = pd.read_csv("../input/item_categories.csv")  # item_category_name, item_category_id
sales_train = pd.read_csv("../input/sales_train.csv")  # date, date_block_num, shop_id, item_id, item_price, item_cnt_day
items = pd.read_csv("../input/items.csv")  # item_name, item_id, item_category_id

test = pd.read_csv("../input/test.csv")  # ID, (shop_id, item_id)
sample_submission = pd.read_csv("../input/sample_submission.csv")  # ID, item_cnt_month


# In[ ]:


print('\n# shops\n')
shops.info()
print('\n# item_categories\n')
item_categories.info()
print('\n# sales_train\n')
sales_train.info()
print('\n# items\n')
items.info()
print('\n# test\n')
test.info()


# In[ ]:


sales_train = pd.merge(sales_train, items, on=['item_id'])
sales_train = pd.merge(sales_train, item_categories, on=['item_category_id'])
sales_train = pd.merge(sales_train, shops, on=['shop_id'])
sales_train.head()


# ## Data cleaning 
# check duplicated, uniqness, null

# In[ ]:


shops.drop_duplicates(keep = 'first', inplace = True)
item_categories.drop_duplicates(keep = 'first', inplace = True)
sales_train.drop_duplicates(keep = 'first', inplace = True)
items.drop_duplicates(keep = 'first', inplace = True)


# In[ ]:


pd.isnull(sales_train).sum()


# In[ ]:


sales_train.info()


# # EDA on training data

# In[ ]:


sales_train_copy = sales_train.copy()


# ## Category
# 
# There are 84 categories from the csv file.
# 
# Relationship to total items in each category, total / average sales in each category, price range in each category

# In[ ]:


# how many items in each category in training data

fig, axs = plt.subplots(2,2,figsize=(15,12))

# Total sales variation

records_category = pd.concat([items.groupby('item_category_id')['item_id'].count(),sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum()],axis=1).rename(columns={'item_id':'item_numbers','item_cnt_day':'total_sales'})  
records_category.sort_index()
records_category['average_sales'] = sales_train_copy.groupby(['item_category_id','item_id'])['item_cnt_day'].sum().groupby('item_category_id').mean().sort_index()  
records_category['sales_std'] = sales_train_copy.groupby(['item_category_id','item_id'])['item_cnt_day'].sum().groupby('item_category_id').std().sort_index() 

# Month variations

# Plot

items.groupby('item_category_id')['item_id'].count().plot(kind='bar',title='item counts in the category (training data)', ax=axs[0,0])     
sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum().plot(kind='bar',title='sales of all items in the category', ax = axs[0,1]) 
records_category['average_sales'].plot(kind='bar',title='average sales of the item in the category', ax = axs[1,0]) 
records_category['sales_std'].plot(kind='bar',title='sales std of the items in the category', ax = axs[1,1]) 

print('Corelation', items.groupby('item_category_id')['item_id'].count().corr(sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum())) 


# In[ ]:


records_category.sort_values(by='average_sales',ascending=False).head()


# For the maximum sales among each category, cate 40 (Cinema - DVD) has 634K total item sales, which also has the most item variations (5025) in all the categories. The item sales has correlation of 0.82 with the item variations in the category. With higher sales in the category, there are more different kinds of items in it. But there are also some exceptions in it.
# 
# There are also some categories that have very high average sales per item. Cate 71 (Gifts - Bags, Albums, Mouse pads) and Cate 79 (Service) have average item sales of 31K and 16K, with only 6 items and 1 item in each category. 

# In[ ]:


# Category sales through time

sales_train_copy_fplot = sales_train_copy.groupby(['date_block_num','item_category_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('item_category_id').loc[:]
plot_df.plot(legend=False, kind = 'line')


# Note that from the figure, we can see the sales trend of well-sold category. Cate 40 (Cinema -DVD) and cate 30 (PC games - standard edition) have the most sales throughout all the months, with the decreasing trends.
# 
# There are also lots of time series trend with monthly total sales lesser than 5000.

# ### items in a category
# Take Cate 40 as examples, take some examples and find if those items has launching information (time) through item_id.

# In[ ]:


items[items['item_category_id']==40].head()


# In[ ]:


cate40_items_ex = [0,10,24,37,22149,22156,22160,22163]
#items in cate40_items_ex
items[items['item_id'].isin(cate40_items_ex) == True]


# Film release date of the items
# (You can search the film if the site https://www.kinopoisk.ru/)
# 
# - item 0: 2000
# - item 10: 2001
# - item 24: 2000
# - item 37: 2011
# - item 22149: 2011
# - item 22156: 2001
# - item 22160: 2004
# - item 22163: 2014
# 
# The item_id in category 40 doesn't really carry the launching information of the items in the 8 examples.

# ## Shops

# In[ ]:


sales_train_copy.groupby('shop_id').sum()['item_cnt_day'].plot(kind='bar', title='total sales of each shop')


# Shop 31 has the most item sales in 2 years.

# In[ ]:


a = sales_train_copy.groupby(['shop_id','item_category_id']).sum()['item_cnt_day'].reset_index()
fig, axs = plt.subplots(1,3,figsize=(15,5))
a[a.groupby(['shop_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index()['item_category_id'].plot(kind='bar',ax=axs[0],title='Best sellinf cates in each shop')
a[a.groupby(['shop_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index().groupby('item_category_id').count()['shop_id'].plot(kind='pie', title='Ratio of Best Selling cates in 60 shops', ax=axs[1]).set_ylabel('') 
a[a.groupby(['item_category_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index().groupby('shop_id').count()['item_category_id'].plot(kind='pie', title='Ratio of Best Selling shops in 84 shops', ax=axs[2]).set_ylabel('') 


# For each shop, the best selling categories varies, but there are category 40 and category 30 are having around 75% being the best category sellers in the 60 shops. 
# 
# For each category, the sales are different in each shops. While category 31, 55 and 25 are containing more of the best selling conditions of the category.

# ## Price

# In[ ]:


plt.figure(figsize=(10,4)) # sales of all samples
sns.boxplot(x=sales_train_copy.item_cnt_day)

plt.figure(figsize=(10,4)) # price of all samples
sns.boxplot(x=sales_train_copy.item_price)


# In[ ]:


sales_train_copy.loc[sales_train_copy['item_cnt_day'].idxmax()]


# In[ ]:


sales_train_copy.loc[sales_train_copy['item_cnt_day'].idxmin()]


# In[ ]:


sales_train_copy.loc[sales_train_copy['item_price'].idxmax()]


# In[ ]:


sales_train_copy.loc[sales_train_copy['item_price'].idxmin()]


# ## Sales information / item based 

# In[ ]:


sales_train_copy.groupby(['item_id']).sum()['item_cnt_day'].plot()


# item 20949 is having extremely high sales compared to other items.

# In[ ]:


bestsale = sales_train_copy.groupby(['item_id']).get_group(sales_train_copy.groupby(['item_id']).sum()['item_cnt_day'].idxmax())
bestsale.sort_values(by='item_cnt_day', ascending=False).head()


# # Train data and Test data overlapped?
# 

# ### items in shops

# old items already lauched in shops

# In[ ]:


train_keys_shop_item = sales_train_copy.groupby(['item_id','shop_id']).groups.keys()
test_keys_shop_item = test.groupby(['item_id','shop_id']).groups.keys()
print('# train_keys and test_keys size', len(list(train_keys_shop_item)), len(list(test_keys_shop_item)))
print('# intersection', len(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item))))


# old items already launched

# In[ ]:


train_keys = sales_train_copy.groupby(['item_id']).groups.keys()
test_keys = test.groupby(['item_id']).groups.keys()
print('# train_keys and test_keys size', len(list(train_keys)), len(list(test_keys)))
print('# intersection', len(set(list(train_keys)) & set(list(test_keys))))


# test items not in train data (False) = new lauched items

# In[ ]:


test.isin({'item_id': list(train_keys)}).groupby('item_id').size()


# plot 3 types of data

# In[ ]:


test_length = len(test)
olditems_alreadylaunched = len(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item)))
newitems_newlaunched = len(test.isin({'item_id': list(train_keys)}).groupby('item_id').get_group(False))
olditems_newlaunched = test_length - olditems_alreadylaunched - newitems_newlaunched

# Data to plot

labels = 'old items already launched', 'old items new launched', 'new items new launched'
sizes = [olditems_alreadylaunched, olditems_newlaunched, newitems_newlaunched]

plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# There are three conditions of data for the test data.
# - old items already luanched are the test data that have history data in training.
# - old items new luanched are the test data that have item history data 'in other shops' in training.
# - new items new luanched are items in test data that don't have any history data in training.
# Here, we'll take a closer look into 'old items new launched' and 'new items new launched'.

# In[ ]:


olditemsshopsintest = pd.DataFrame(np.asarray(list(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item)))), columns=['item_id','shop_id']) 
olditemsshopsIDintest = pd.merge(test, olditemsshopsintest)['ID']

test1 = pd.concat([test, test.isin({'item_id': list(train_keys)})['item_id'].rename('item_in_train') ], axis=1)
newitemsIDintest = test1.groupby('item_in_train').get_group(False)['ID']

olditemsnewshopIDintest = test.drop(pd.concat([olditemsshopsIDintest,newitemsIDintest]))['ID']

# Index data

olditemsnewshopintest_data = test.loc[test['ID'].isin(olditemsnewshopIDintest)]
olditemsshopsintest_data = test.loc[test['ID'].isin(olditemsshopsIDintest)]
newitemsintest_data = test.loc[test['ID'].isin(newitemsIDintest)]


# ## Old items new launched in shop items
# 
# For old items new launched in shop items, the real condition can be
# - new launched in the shop 
# - launched with 0 selling records
# - launched with missing selling records
# 
# Few questions to ask for this kind of condition:
# - How's the item saling in other shops? (hotness of the item)
# - How's the category saling in the shop? in other shops? 
# 
# We'll use eda techniques to help decide how to deal with this pipe of condition.

# In[ ]:


# Group by items

toplot = olditemsnewshopintest_data[['shop_id','item_id']].groupby('item_id').count().rename(columns = {'shop_id':'shop_numbers_intest'}).reset_index()   

itemsoldinshops_train = sales_train_copy.groupby(['item_id','shop_id']).count().reset_index()[['item_id','shop_id']].groupby('item_id').count().rename(columns = {'shop_id':'shop_numbers_intrain'}).reset_index()[['item_id','shop_numbers_intrain']]  
toplot = pd.merge(toplot, itemsoldinshops_train)
toplot['test_train_shopsratio'] = toplot['shop_numbers_intest'] / toplot['shop_numbers_intrain']

fig,axs = plt.subplots(1,3,figsize=(15,5))
toplot.sort_values(by='shop_numbers_intrain').reset_index()['shop_numbers_intrain'].plot(ax=axs[0])
toplot.sort_values(by='shop_numbers_intest').reset_index()['shop_numbers_intest'].plot(ax=axs[1])
toplot.sort_values(by='test_train_shopsratio').reset_index()['test_train_shopsratio'].plot(ax=axs[2])
axs[0].title.set_text('shop_numbers_in_train')
axs[1].title.set_text('shop_numbers_in_test')
axs[2].title.set_text('test_train_shopsratio')


# For the kind of data, there are around 87K shop+item combinations, with around 4.7K items in the test data. For each item, the number of other shops having sales records in training data can range to around 60 shops, while the number of shops not having sales records in testing data can also range to around 45 shops. 
# 
# Most of the inputs in this kind of data (old items new launched) have the phenomenon with testing data less than training data, with some having very extreme ratio (test num / train num more than 40). Therefore, we have to predict the sales of the items with most of the conditions that the items have lesser data of other shops in traning data compared to testing data.
# 
# We'll then take a detail looks to some of the examples in the case. From the perspective of items, we'll take 3 items and do further analysis from the view of item and category.
# - item_id: 11286  shop_numbers_intest: 41  shop_numbers_intrain: 1  
# - item_id: 13818  shop_numbers_intest: 18  shop_numbers_intrain: 25
# - item_id: 4240  shop_numbers_intest: 2  shop_numbers_intrain: 57
# 

# In[ ]:


# The category ratio of this task
toplot = pd.merge(toplot, items)
toplot.groupby('item_category_id').count()['item_id'].rename(columns={'item_id':'item_count'}).plot()   


# The figure shows the item counts in group of category. In this kind of task, there are lots of items between the category 14-75.

# In[ ]:


toplot[toplot['item_id'].isin([11286,13818,4240])]


# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_id']==4240].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_id']==4240].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_id']==4240].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of item 4240')
axs[1].set_title('Monthly sales of each shop of item 4240')
axs[2].set_title('Monthly average sales of the shops having sales')


# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_category_id']==23].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==23].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_category_id']==23].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of cate 23')
axs[1].set_title('Monthly sales of each shop of cate 23')
axs[2].set_title('Monthly average cate sales of the shops having sales')


# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_id']==13818].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0],kind='bar')      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_id']==13818].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_id']==13818].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of item 13818')
axs[1].set_title('Monthly sales of each shop of item 13818')
axs[2].set_title('Monthly average sales of the shops having sales')


# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_category_id']==37].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==37].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_category_id']==37].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of cate 37')
axs[1].set_title('Monthly sales of each shop of cate 37')
axs[2].set_title('Monthly average cate sales of the shops having sales')


# In[ ]:


sales_train_copy[sales_train_copy['item_id']==11286].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(title='Monthly sales of all shops of item 11286 (only one shop)')      


# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_category_id']==31].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(kind='bar', ax=axs[0])      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==31].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_category_id']==31].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of cate 31')
axs[1].set_title('Monthly sales of each shop of cate 31')
axs[2].set_title('Monthly average cate sales of the shops having sales')


# - item_id: 4240  shop_numbers_intest: 2  shop_numbers_intrain: 57
# 
# The item 4240 (Kinect Dance Central 3 (only for MS Kinect) [Xbox 360]), in category 23 (Games - Xbox360) has the sales peak around the 23rd date_block_num. Cate 23 is also having the similar trend with the item 4240.
# 
# - item_id: 13818  shop_numbers_intest: 18  shop_numbers_intrain: 25
# 
# The item 13818 (LEGENDS OF NIGHT GUARDS WB (BD)), in category 37 (Games - Xbox360) start having better sales on the 30th date_block_num. The trend of cate 37 isn't similar to the item's.
# 
# - item_id: 11286  shop_numbers_intest: 41  shop_numbers_intrain: 1  
# 
# The item 11286 (Truckers 3: The Conquest of America + Great Race [PC, Digital Version]), in category 37 (PC Games - Digital) have sales ranging from 2 to 16 through months, starting from month 20. Cate 37 is also having more sales from month 20 to month 30.
# 
# - similar trends
# 
# For each shop, they follow similar trend of total sales, while there are also some with 0 saling records or missing points.
# In addition, the average sales of each shop having sales are also following some of the trends of total sales.
# To find out the previous records of not-having-sales-record-in-shop items, we can use the average of the item sales among the shops having saling records. 

# ## New items new launched in shop items
# 
# There are around 7% of test data in the category.
# 
# For new items new launched in shop items, the real conditions can be 
# - new launched 
# - launched with 0 selling records
# - launched with missing selling records
# 
# Question to ask for this kind of condition:
# - How's the category saling in the shop? in other shops? 
# 
# After some observation, we'll find out how to deal with the kind of condition.

# In[ ]:


newitemsintest_data = pd.merge(newitemsintest_data,items[['item_id','item_category_id']],on='item_id')

fig, axs = plt.subplots(1,2,figsize=(15,8))
newitemsintest_data.groupby('item_category_id').count()['item_id'].plot(kind='bar', ax=axs[0])
newitemsintest_data.groupby('item_id').count()['shop_id'].plot(kind='bar',ax=axs[1]) # all the same

axs[0].set_title('test input counted in item_category')
axs[1].set_title('test input counted in items [based on shop]')


# We look at the kind of data by using the history data of category sales in the shop.
# There are around 15246 kinds of inputs, 16K groups when considering shops and category, distributed in 39 category, 42 shops, 363 items. (For each item, we're predicting the sales in the 42 shops.)
# Since we don't have the item sales record, for each input, we can take a look at the category sales in the shop and over all sales in all shop. Take cate 72 for example.
# 
# 

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(15,5))

sales_train_copy[sales_train_copy['item_category_id']==72].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      

sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==72].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1])

sales_train_copy[sales_train_copy['item_category_id']==72].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      

axs[0].set_title('Monthly sales of all shops of cate 72')
axs[1].set_title('Monthly sales of each shop of cate 72')
axs[2].set_title('Monthly average cate sales of the shops having sales')


# We can apply the history data of monthly item average sales of the category for the kind of data.

# ## Discussion
# 
# From the above analysis, we've found that there are mainly three kinds of test data. In addition, depends on the kinds, we can think of the training data set presented to us in different ways. Here are some questions we can ask:
# 
# - Whether the training data is really 'the complete' history data ?
# - Whether non-record item is really having no sales? or is its history missing? or is it new launched?
# 
# Here are some directions that we can think of based on the kind of test data:
# 
# - Old items already launched (52%)
#     - Believe all the training data are the total selling records in history
#     - Training data are missing some selling records
# - Old items new launched in shops (42%)
#     - 0 selling records in the shop
#     - Missing selling records of the shop
#     - New launched in the shop at the month to be predicted
# - New items new launched (7%)
#     - 0 selling records in the shop
#     - Missing selling records of the shop
#     - New launched in the shop at the month to be predicted
# 
# Depending on the way we look at the training data, we will have different methods on dealing with these kinds of data:
# 
# - Training data not missing history (0 sales if not having records)
# When predicting the sales, directly assign 0 to the time that don't have records and then predict the monthly sales.
# - Training data missing history (items launched in the shop)
# On the analysis above, we found out that giving the average monthly sales can be a kind of solution, such as using average monthly sales of the item in all shop (Old items new launched) or using average monthly item sales of the category in the shop (New items new launched).
# - New launched at the month to be predicted
# Need to find out the new launched items in training data, both as new launched in a shop or all shops, and use the kind of data to create new set of training data and then predict. This is different then two types above, since that they're using all the history data in training data. The method here needs to distinguish the start-selling day, and the predicted results will depend on their selling records afterwards, with other features like category, shops, etc included.
# 
# Note that since 'Old items new launched' has around 40%, the way we deal with the kinds of data can also play an essential role on having a better predicting results.

# # Special Case - negative values of price and sales in train data

# ## Negative price
# 
# - We first find about the items with negative price (one case)
# - Further analysis on their features (shop, price, sales)

# In[ ]:


sales_train_copy.groupby('item_price').size()


# In[ ]:


sales_train_copy.groupby('item_price').get_group(-1)


# There's one case with negative price, check the prices of the item sold in the shop.

# In[ ]:


sales_train_copy[(sales_train_copy['item_id']==2973) & (sales_train_copy['shop_id']==32)].head()


# From the data above, we can assume that maybe it's wrong keyed in, we can replace the the price with the same month price = 1249. Next, we take a little closer look and do some analysis on the item.

# ##### About the negative-price item - sales, price, shop

# In[ ]:


sale2973 = sales_train_copy[sales_train_copy['item_id']==2973]
sale2973.head()


# monthly total sales of the item in each shop and all shops

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))

sale2973.groupby('date_block_num')['item_cnt_day'].sum().plot(kind=' bar', ax=axs[0], title='monthly total sales of the item in all shops') # monthly sales of the item 

sale2973_fplot = sale2973.groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
plot_df = sale2973_fplot.unstack('shop_id').loc[:]
plot_df.plot(legend=False, kind = 'line', ax=axs[1], title='monthly total sales of the item in each shops')


# monthly average price  and std of all shops of the item

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))

sale2973.groupby(['date_block_num'])['item_price'].mean().plot(kind = 'bar', ax=axs[0], title='monthly average price of all shops of the item')
pd.Series(sale2973.groupby(['date_block_num', 'shop_id'])['item_price'].mean().groupby('date_block_num').std()).plot(kind = 'bar', ax=axs[1], title='monthly price std between all the shops')     


# ## Negative sales
# - 3511 / 22K items have returns
# - 7259 kinds of varations on items+month+shops
# - distribution in each category

# In[ ]:


print(sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_id']).size())
print(sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_id','date_block_num', 'shop_id']).size())


# In[ ]:


# Category with sales return 

fig, axs = plt.subplots(1,3,figsize=(15,5))

# item-based negative-sold counts in each category
sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_category_id'])['item_id'].agg(['count']).plot(kind='bar',ax = axs[0])                               
# item negative-sold total sales in each category
sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_category_id'])['item_cnt_day'].agg(['sum']).abs().plot(kind='bar', ax = axs[1])                           
# item-based
negative_totalsales = sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby('item_id')['item_cnt_day'].agg('sum').abs()
negative_totalsales.plot(kind='line', ax=axs[2], title='Counts of returns of each item')

axs[0].title.set_text('Counts of items with return record in the category')
axs[1].title.set_text('Ruturn sales of items in the category')


# Similar trend for fig1 and fig2 (no big difference on item_id or item_sales), since that no sales return isn't that big for each item (max 60)

# #####  About the item with the most return sales
# * item_id: 2331
# * sales return and sold: 60/542
# * category: 20 (Game-PS4)
# 
#     * 100 negative-sales items
#     * 157 items with record in training data
#     * 175 items in total

# In[ ]:


sales_2331 = sales_train_copy[sales_train_copy['item_id']==2331]
print('positive sales', sales_2331[sales_2331['item_cnt_day']>0]['item_cnt_day'].sum())
print('negative sales', sales_2331[sales_2331['item_cnt_day']<0]['item_cnt_day'].sum())


# In[ ]:


sales_cate20 = sales_train_copy[sales_train_copy['item_category_id']==20]

# 157 items in cate 20 in training data 
fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Total sales of the 157 items in category 20 (training data)')
sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].sort_values().plot(kind='bar', ax=axs[0])
sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].sort_values().plot(kind='pie', ax=axs[1])


# In[ ]:


# negative sales of the category - 100/157 items with negative sales in this category
fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Total returns of the 100 items in category 20 (training data)')
sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].abs().sort_values().plot(kind='bar', ax=axs[0])
sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].abs().sort_values().plot(kind='pie', ax=axs[1])


# In[ ]:


## Among these negative-sale items in cate20 - 100 items in 157/175 items in category 20

# negative sales
sales20_negtotal = sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].reset_index()
# positive sales
sales20_postotal = sales_cate20[sales_cate20['item_cnt_day']>0].groupby('item_id').agg('sum')['item_cnt_day'].reset_index()
# total sales
sales20_total = sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].reset_index()

sales20_compare = pd.merge(sales20_postotal, sales20_negtotal.abs(), on='item_id')
sales20_compare.rename(columns={"item_cnt_day_x": "sales_sold", "item_cnt_day_y": "sales_return"}, inplace=True)
sales20_compare = pd.merge(sales20_compare, sales20_total, on='item_id')
sales20_compare.rename(columns={"item_cnt_day": "sales_total"}, inplace=True)
sales20_compare.fillna(0)
sales20_compare['return_ratio'] = sales20_compare['sales_return'] / sales20_compare['sales_sold']

fig, axs = plt.subplots(2,2,figsize=(15,10))
fig.suptitle('Sale records of the 100 items with returns in category 20 (training data)')
sales20_compare[['item_id','sales_sold']].plot(x='item_id',y='sales_sold', kind='bar', ax=axs[0,0])
sales20_compare[['item_id','sales_return']].plot(x='item_id',y='sales_return', kind='bar', ax=axs[0,1])
sales20_compare[['item_id','sales_total']].plot(x='item_id',y='sales_total', kind='bar', ax=axs[1,0])
sales20_compare[['item_id','return_ratio']].plot(x='item_id',y='return_ratio', kind='bar', ax=axs[1,1])


# In the 100 items that are sold with return
# - The sales sold ranges from a few to 6000 items
# - The maximum of sales return is around 60 items
# - Most of the returns are smaller than 10 items
# - The most return items isn't the best seller in this category

# # Conclusion
# 
# Several analysis on training data are implemented. 
# 
# The types of test data (item, shop) have 3 kinds of condition
# - having history data of the item in the shop
# - not having data of the item in shop but in other shops
# - not having any history data. 
# 
# Depending on the way we view the kind of data, the way we design the model varies. Note that there are 41% of data that are not having data of the item in shop but in other shops, and how to deal with them can also play an essential roles on predicting. 
# 
# In addition, special cases like negative price and sales are discussed. 
# 
# More analysis techniques and topics can be implemented in the future.
