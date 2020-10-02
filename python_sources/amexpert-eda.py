#!/usr/bin/env python
# coding: utf-8

# # AmExpert EDA and Insights.
# 
# **Problem Statement: **
# The goal here in this problem is to successfully predict which customers will redeem which coupons assigned to them in a given campaign across various channels.
# 
# The data is divided into multiple csv files each holding unique information about the part of the problem we are trying to solve.
# 
# The dataset consist of 8 files namely:
# 1. campaign_data.csv : Information about the campaign held by the company. (campaign_id)
# 2. coupon_item_mapping.csv : Mapping the coupon data with the item data. (coupon_id, item_id)
# 3. customer_demographics.csv : Demographics of the customers. (customer_id)
# 4. customer_transaction_data.csv : Transactions of the customers during the campaign period. (customer_id)
# 5. item_data.csv : information about the items sold in the campaign. (item_id)
# 6. train.csv and test.csv : Training and Testing data consisting of some features and our target variable(redemption_status).
# 7. sample_submission.csv : sample submission file for final prediction. - 0.5 AUC without creating any model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

#sns color plalette.
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colour = sns.color_palette(flatui)

#Path for the directory
PATH = '../input/amexpert/'


# In[ ]:


#Reading the dataset.
train = pd.read_csv(f'{PATH}train.csv')
test = pd.read_csv(f'{PATH}test_QyjYwdj.csv')
campaign_data = pd.read_csv(f'{PATH}campaign_data.csv')
coupon_item = pd.read_csv(f'{PATH}coupon_item_mapping.csv')
customer_trans = pd.read_csv(f'{PATH}customer_transaction_data.csv')
item_data = pd.read_csv(f'{PATH}item_data.csv')
customer_demo = pd.read_csv(f'{PATH}customer_demographics.csv')


# # Exploring the dataset.
# 
# The goal here will be to explore each dataset carefully and then deduce some insights from it and understand the data better which will help us in our later part of feature engineering and feature selection.

# Let us first explore the train dataset.

# In[ ]:


#train first 5 observations
train.head()


# This dataset consist of 5 columns. The id column is a unique id for each entry or observation in the dataset.
# 1. campaign_id : Ids of Campaign held by the company. 18 campaigns data. 
# 2. coupon_id : coupons offered during campaigns. 
# 3. customer_id : Ids representing the custmoers.
# 4. redemption_status : Target variable, whether the coupon was redeemed by the user/customer.

# In[ ]:


#shape of the dataset.
print(f'Train shape: {train.shape}')


# > Consists of 78369 observations.

# In[ ]:


#function to get the number of unique items in a feature.
def get_unique(df, col=None):
    assert col == None or type(col) == list, 'Either None or List.'
    if col:
        for i in col:
            print(f'Unique items in {i} are {df[i].unique().shape[0]}')
    else:
        for col in df.columns:
            print(f'Unique items in {col} are {df[col].unique().shape[0]}')


# In[ ]:


#getting unique items in the data.
get_unique(train, col=['id', 'campaign_id', 'coupon_id', 'customer_id'])


# Here are the unique items in a particular columns. As we can see the campaign_id consists of only 18 campaigns as said in the data description.
# > Based on previous transaction & performance data from the last 18 campaigns, predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?

# In[ ]:


#Distribution of the redemption_status.
plt.figure(figsize=(7,5))
sns.countplot(train.redemption_status, color=colour[0])


# We have a highly imbalanced dataset. The ratio of imbalancess is not clear from the graph. Lets see some hard numbers.

# In[ ]:


#value count
train.redemption_status.value_counts()


# We have a very few couponn redemption. Heavily imbalanced dataset.

# In[ ]:


#imbalance ratio:
print('*'*70)
print(train.redemption_status.value_counts(normalize=True))
print('*'*70, '\n')
print(f'Imbalance ratio: {train.redemption_status.value_counts()[1] / train.redemption_status.value_counts()[0]}')


# The imbalance ratio is 0.009. The percent of each value in the column is shown above with 0 have 99% distribution and 1 with only 0.009%.

# In[ ]:


#Campaign data.
campaign_data.head()


# Will talk about features not mentioned previously.
# 1. campaign_type: Type  of the campaign X or Y. The exact meaning not known and is not needed.
# 2. start_date : Start date of the campaign.
# 3. end_date: End date of the campaign.
# 
# From the first five observation we can see that the campaign last for 1 or 2 months.

# In[ ]:


get_unique(campaign_data)


# The unique campaign held are 28 which seems right as there are 18 campaigns in the train data and 10 campaigns in the test data. This means that we have data for all the campaigns held in the dataset.

# In[ ]:


#Campaign type
plt.figure(figsize=(7,5))
sns.countplot(campaign_data.campaign_type, color=colour[1])


# In[ ]:


#Start date.
campaign_data.start_date.describe()


# In[ ]:


#End date.
campaign_data.end_date.describe()


# The start date and end date here are of type string. The top index shows us the most number of occurence of a value in the feature and the feq gives us the frequency of occurence of that value.

# We can create new features from this dataset such as number of monhs the campaign lasted, number of days the campaign lasted, Day of week when the campaign started/ended, month of campaign start/end so on and so forth.

# In[ ]:


#shape of the data.
campaign_data.shape


# In[ ]:


#coupon item mapping
coupon_item.head()


# Mappings of items with the coupons.

# In[ ]:


#unique data
get_unique(coupon_item)


# there are 1116 coupons and 36289 items in this dataset.

# In[ ]:


#items data
item_data.head()


# This set contains more information on the items which were present in the coupon item mapping dataset.
# 1. brand : brand to which an item belong.
# 2. brand_type: type/status of brand.
# 3. category : category to which the brand belongs.

# In[ ]:


#shape:
item_data.shape


# In[ ]:


#getting unique items from this data.
get_unique(item_data)


# We can see that there are 74066 total items sold by the store and not all the items are present in the coupon item mapping dataset.
# This is made clear from the data description page that the coupons can be availed for some items only.

# In[ ]:


#plotting brand_type
plt.figure(figsize=(7,5))
sns.countplot(item_data.brand_type, color=colour[2])


# In[ ]:


#brand category.
plt.figure(figsize=(17,5))
sns.countplot(item_data.category, color=colour[3])
plt.xticks(rotation=90);


# Most of the item belongs to the Grocery and Pharmaceuticals category.

# In[ ]:


#lets merge the items data with coupon_item mapping.
coupon_item = coupon_item.merge(item_data, on='item_id', how='left')
coupon_item.head()


# In[ ]:


#shape
coupon_item.shape


# Lets check for any nul values. Presence of null values will indicate that the information for the particular item present in the coupon item mapping dataset is not present in the items dataset.

# In[ ]:


#null values.
coupon_item.isnull().sum()


# We have information for all the items present in the coupon item mapping data.

# Lets us now check which of the items for which the coupon is given belongs to what category.

# In[ ]:


#category on which most coupons are given.
group1 = coupon_item.groupby('category')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)
plt.figure(figsize=(17,5))
sns.barplot(group1.category, group1.count_coupons, color=colour[4])
plt.xticks(rotation=90);


# The coupons are given the most for the grocery items followed by pharmaceuticals and natural products.<br>
# We yet do not know how many of these coupons are redeemed by the customers which we will explore later.

# In[ ]:


#brand on which most coupons are given.
get_ipython().run_line_magic('time', "group2 = coupon_item.groupby('brand')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)")
plt.figure(figsize=(17,5))
sns.barplot('brand', 'count_coupons',
            data=group2.head(10), #considering only top 10
            color=colour[5])
plt.xticks(rotation=90);


# Most of the coupons are given on brand 56. Comparitvely the highest. There is a vast difference in the coupon applied on brand 56 and the rest of the brands.<br>
# Again we do not know how many of these coupons are redeemed by the customer. We wil look into that later. I promise!

# In[ ]:


#items on which most coupons are given.
get_ipython().run_line_magic('time', "group3 = coupon_item.groupby('item_id')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)")
plt.figure(figsize=(17,5))
sns.barplot('item_id', 'count_coupons', 
            data=group3.head(10), #considering only top 10
            color=colour[0])
plt.xticks(rotation=90);


# Item 56523 is present in most of the coupons. The rest of the 9 items distribution is almost the same.

# In[ ]:


#brand_type on which most coupons are given.
get_ipython().run_line_magic('time', "group4 = coupon_item.groupby('brand_type')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)")
plt.figure(figsize=(7,5))
sns.barplot('brand_type', 'count_coupons', 
            data=group4,
            color=colour[1])
# plt.xticks(rotation=90);


# Most of the coupons are given on Established brands. It make sense that the established brand will have most redemption ratio than the local brands. But this is only an assumption. We will check on this assumption later.

# In[ ]:


#customer demographics.
customer_demo.head()


# This dataset contain information about the customers of the ABC company. All the information are not present in the dataset as it is mentioned in the data description and as we can see from the first five observations (NANs present).
# 1. age_range : age bracket under which the customer falls.
# 2. marital_status: Married/ single.
# 3. rented: Accomodation rented or not. How did they collect this data and why?. It does not seem to me that this column will be of any use here as it is irrelevant information. But we cannot conclude on it yet. We need to check it first.
# 4. family_size: size of the family.
# 5. no_of_childrens: childrens if any.
# 6. income_bracket: Label encoded as descriped in the description with higher number indicating higher income.
# 
# Many of the columns here seems to me as irrelevant and will be a noise to the model. But this needs to be verified first by creating a model and plotting the feature importances. Or we can use any other method such as the correlation of these features with the response variable.

# In[ ]:


#shape
customer_demo.shape


# In[ ]:


#unique_items.
get_unique(customer_demo)


# As we can see there are 760 customers for whihc the demographics are present.

# In[ ]:


#frequency of age range
plt.figure(figsize=(8,5))
sns.countplot(customer_demo.age_range, color=colour[2])


# Most of the customers falls in the age range of 46-55. Given the types of items sold by the ABC company such as Groceries, Pharmaceuticals, it only make sense that ther customers will mostly be middle aged group as can bee seen from the graph above.

# In[ ]:


#marital status.
customer_demo.marital_status.value_counts(dropna=False)


# Information for about 329 customer regarding their marital status is not present. Most married people are using the services of the ABC company.

# In[ ]:


#rented
plt.figure(figsize=(7, 5))
sns.countplot(customer_demo.rented, color=colour[3])


# I do not know if this data is accurate or not as many customers will not provide correct information but the above plot shows that most of the customers live in a rented apartment.
# 
# Again this may not be a good or relevant feature to use here but it is just my assumption and I need to check it.
# 

# In[ ]:


#family size.
#rented
plt.figure(figsize=(7, 5))
sns.countplot(customer_demo.family_size, color=colour[4])


# Looks like a ordinal categorical variable. An ordinal variable is a variable which has an order of importance or ranking to it. Customer with 2 members in the family including themselves are the ones who order from the store the most.

# In[ ]:


#number of childrens
#rented
plt.figure(figsize=(7, 5))
sns.countplot(customer_demo.no_of_children, color=colour[4])


# Self explainatory. The number of childrens a customer is having. Categorica feature. The data also contains NAN values and it is possible that the customers having NAN in the chidrens columns actually do no have any childrens.

# In[ ]:


#number of childrens
#rented
plt.figure(figsize=(7, 5))
sns.countplot(customer_demo.income_bracket, color=colour[4])


# Most of the customers belong to Income bracket 5 followed by 4.

# In[ ]:


#null values present in this data.
customer_demo.isnull().sum()


# In[ ]:


#Customer Transaction data.
customer_trans.head()


# This is the history of the customers transactions on the store.
# 1. date: Date of transaction.
# 2. quantity: amount of an item bought.
# 3. selling_price :price at which an item was sold.
# 4. other_discount: Discount from sources.
# 5. coupon_discount: Discount availed from retailer coupons.

# In[ ]:


#number of unique items
get_unique(customer_trans)


# In[ ]:


#distribution of quantity
sns.distplot(customer_trans.quantity, bins=50);


# The quantity value is ranging from 1 to upto 80000. This seems very odd. Who will buy 80000 items in one order.

# In[ ]:


#description
customer_trans.quantity.describe()


# In[ ]:


#subsetting.
high_order = customer_trans.loc[customer_trans.quantity >= 50]
high_order.sort_values('quantity', ascending=False, inplace=True)
high_order.head()


# The quantity of an item with id 49009 is 896338 which does not look correct. There must be something wrong in this data.

# In[ ]:


#selling price.
customer_trans.selling_price.describe()


# In[ ]:


#selling price distribution.
sns.distplot(np.log1p(customer_trans.selling_price), bins=20)


# In[ ]:


#other discount.
customer_trans.other_discount.describe()


# This data is in negative which seems alright as the discounted price will be reduced from the selling price.

# In[ ]:


#coupon discount.
customer_trans.coupon_discount.describe()


# In[ ]:


#distribution
sns.distplot(abs(customer_trans.coupon_discount))


# In[ ]:


#shape of the dataset.
customer_trans.shape

