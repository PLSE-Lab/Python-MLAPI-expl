#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Settings
# ---

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import seaborn as sns
sns.set_style('whitegrid', {'grid.linestyle': '--'})

get_ipython().run_line_magic('matplotlib', 'inline')


# # Objectives
# ---
# In this kernel I am covering three subjects with data of the Brazilian ecommerce marketplace, Olist.
# 1. Marketing Channel Effectiveness
# 2. Sales Performance Overview
# 3. Closed Deal Performance Overview

# ## 1. Marketing Channel Effectiveness
# ---
# Olist acquired sellers through diverse marketing channels. Let's find out which channel was the most effective in lead generation. Below the term 'Marketing Qualified Lead(MQL)' means a potential reseller/manufacturer who has an interest in selling their products on Olist. 

# In[ ]:


# Load 'MQL' dataset
mql = pd.read_csv('../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv',
                  parse_dates=['first_contact_date'])

print(mql.shape)
mql.head(3)


# ### Time Series Volume of Marketing Qualified Lead (MQL)

# In[ ]:


# Add a 'year-month' column
mql['first_contact_date(y-m)'] = mql['first_contact_date'].dt.to_period('M')

print(mql.shape)
mql[['first_contact_date', 'first_contact_date(y-m)']].head(3)


# In[ ]:


# Create time series table
monthly_mql = mql.groupby(by='first_contact_date(y-m)').mql_id                                                        .count()
monthly_mql.to_frame().T


# In[ ]:


# Plot the monthly MQL volume
monthly_mql.plot.line(figsize=(12, 6))
plt.title('MQL Volume (Jun 2017 - May 2018)', fontsize=14);


# Since 2018, monthly MQL volume soared to above 1,000. Let's take a look at the volume for each acquisition channel.

# ### MQL Volume by Marketing Channel
# _\* Marketing channel is recorded in 'origin' field._

# In[ ]:


# Create 'channel-time series' table
mql_origin = pd.pivot_table(mql,
                            index='origin',
                            columns='first_contact_date(y-m)',
                            values='mql_id',                            
                            aggfunc='count',
                            fill_value=0)

# Sort index from largest to smallest in volume
origin_list = mql.groupby('origin').mql_id                                    .count()                                    .sort_values(ascending=False)                                    .index

mql_origin = mql_origin.reindex(origin_list)
mql_origin


# In[ ]:


# Plot the monthly volume by channel
plt.figure(figsize=(20,8))
sns.heatmap(mql_origin, annot=True, fmt='g');


# + Paid search is the second biggest contributor to lead generation after 'organic search'.
# + The third one is 'social' which acquired MQLs more than or similar to 'paid search' since April 2018.
# + If the marginal cost of paid search increases, it would be possible to examine effectiveness of 'social' as an alternative.

# ## 2. Sales Performance Overview
# ---
# After a MQL filled a form on landing page to sign up for seller, a Sales Development Representative(SDR) contacted the MQL and gathered more information about the lead. Then a Sales Representative(SR) consulted the MQL. So interaction between SDRs/SRs and MQLs can affect conversion from MQLs to sellers.  
# 
# At this section I will deal with two aspects of sales result, conversion rate and sales length.  
# _\* A MQL who finally signed up for seller is called a closed deal._

# In[ ]:


# Load 'closed deals' dataset
cd = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv',
                 parse_dates=['won_date'])

print(cd.shape)
cd.head(3)


# In[ ]:


# Merge 'MQL' with 'closed deals'
# Merge by 'left' in order to evaluate conversion rate
mql_cd = pd.merge(mql,
                  cd,
                  how='left',
                  on='mql_id')

print(mql_cd.shape)
mql_cd.head(3)


# In[ ]:


# Add a column to distinguish signed MOLs from MQLs who left without signing up
mql_cd['seller_id(bool)'] = mql_cd['seller_id'].notna()

print(mql_cd.shape)
mql_cd[['seller_id', 'seller_id(bool)']].head()


# In[ ]:


# Compute monthly closed deals
monthly_cd = mql_cd.groupby('first_contact_date(y-m)')['seller_id(bool)'].sum()
monthly_cd.to_frame().T


# In[ ]:


# Plot the monthly volume of closed deals
monthly_cd.plot.line(figsize=(12, 6))
plt.title('Closed Deal Volume (Jun 2017 - May 2018)', fontsize=14);


# Likewise, monthly volume of closed deals sharply increased after 2018.

# ### Conversion Rate
# Conversion rate means the percentage of MQLs who finally signed up for sellers (closed deals). 

# In[ ]:


# Calculate monthly conversion rate
monthly_conversion = mql_cd.groupby(by='first_contact_date(y-m)')['seller_id(bool)'].agg(['count', 'sum'])

monthly_conversion['conversion_rate(%)'] = ((monthly_conversion['sum'] / monthly_conversion['count']) * 100).round(1)
monthly_conversion.T


# In[ ]:


# Plot the monthly conversion rate
monthly_conversion['conversion_rate(%)'].plot.line(figsize=(12, 6))
plt.title('Conversion Rate (Jun 2017 - May 2018)', fontsize=14);


# Conversion rate also increased with volume.

# ### Sales Length
# Sales length means period from first contact to signing up for seller. 

# In[ ]:


# Calculate sales length in days
mql_cd['sales_length(day)'] = np.ceil((mql_cd['won_date'] - mql_cd['first_contact_date'])
                                      .dt.total_seconds()
                                      / (60*60*24))

print(mql_cd.shape)
mql_cd[['first_contact_date', 'won_date', 'sales_length(day)']].head()


# In[ ]:


# Separate sales length for each year
closed_deal = (mql_cd['seller_id'].notna())
lead_2017 = (mql_cd['first_contact_date'].dt.year.astype('str') == '2017')
lead_2018 = (mql_cd['first_contact_date'].dt.year.astype('str') == '2018')

sales_length_2017 = mql_cd[closed_deal & lead_2017]['sales_length(day)']
sales_length_2018 = mql_cd[closed_deal & lead_2018]['sales_length(day)']

sales_length_2017.head(3), sales_length_2018.head(3)


# In[ ]:


# Plot the sales length of each year
figure, ax = plt.subplots(figsize=(12,6))

sns.kdeplot(sales_length_2017,
            cumulative=True,
            label='2017 (Jun-Dec)',
            ax=ax)
sns.kdeplot(sales_length_2018,
            cumulative=True,
            label='2018 (Jan-May)',
            ax=ax)

ax.set_title('Sales Length in Days', fontsize=14)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlim(0,500);


# Sales length was dramatically shortened as well. While 13.1% of deals were closed within 50 days in 2017, 78.9% was closed in 2018. In a nutshell, both conversion rate and sales length were improved in 2018 against 2017. 

# ### Digging into Closed Deal
# Before move on to the next subject, I will explore closed deals in more depth to see whether there is room for improvement in sales process. Specifically I'm looking into three dimensions of closed deals(__'lead type', 'business segment', 'business type'__) with __'lead behaviour profile'__ as an axis.

# #### Characteristics of Closed Deal

# In[ ]:


# Bring 'closed deals' data
cd_profile = cd[cd['lead_behaviour_profile'].notna()].copy()

print(cd_profile.shape)
cd_profile['lead_behaviour_profile'].value_counts()


# In[ ]:


# Combine four types of mixed profiles(2.4%) into 'others'
profile_list = ['cat', 'eagle', 'wolf', 'shark']

cd_profile['lead_behaviour_profile(upd)'] = cd_profile.lead_behaviour_profile                                                       .map(lambda profile: profile
                                                           if profile in profile_list
                                                           else 'others')

print(cd_profile.shape)
cd_profile['lead_behaviour_profile(upd)'].value_counts()


# In[ ]:


# Create 'profile - lead type' table
cols = cd_profile['lead_type'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_leadType = pd.pivot_table(cd_profile,
                                  index='lead_behaviour_profile(upd)',
                                  columns='lead_type',
                                  values='seller_id',
                                  aggfunc='count',
                                  fill_value=0)

profile_leadType = profile_leadType.reindex(index)[cols]
profile_leadType


# In[ ]:


# Create 'profile - business type' table
cols = cd_profile['business_type'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_businessType = pd.pivot_table(cd_profile,
                                      index='lead_behaviour_profile(upd)',
                                      columns='business_type',
                                      values='seller_id',
                                      aggfunc='count',
                                      fill_value=0)

profile_businessType = profile_businessType.reindex(index)[cols]
profile_businessType


# In[ ]:


# Create 'profile - business segment' table
cols = cd_profile['business_segment'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_segment = pd.pivot_table(cd_profile,
                                 index='lead_behaviour_profile(upd)',
                                 columns='business_segment',
                                 values='seller_id',
                                 aggfunc='count',
                                 fill_value=0)

profile_segment = profile_segment.reindex(index)[cols]
profile_segment


# In[ ]:


# Plot the above three tables
figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,20))
figure.subplots_adjust(hspace=0.3)

sns.heatmap(profile_leadType,
            annot=True,
            fmt='g',
            ax=ax1)
sns.heatmap(profile_businessType,
            annot=True,
            fmt='g',
            ax=ax2)
sns.heatmap(profile_segment,
            annot=True,
            fmt='g',
            ax=ax3)

ax1.set_title('Behaviour Profile - Lead Type', fontsize=14)
ax2.set_title('Behaviour Profile - Business Type', fontsize=14)
ax3.set_title('Behaviour Profile - Business Segement', fontsize=14);


# 'lead_behaviour_profile' is related to DISC personality test. Each type has the meaning as follows.  
# _\* Source: [DISC Profile](https://www.discprofile.com/what-is-disc/overview/)_
# 
# Behaviour_profile | DISC_profile | Description
# ------------------- | ------------- | ------------
# Cat | Steadiness | Person places emphasis on cooperation, sincerity, dependability
# Eagle | Influence | Person places emphasis on influencing or persuading others, openness, relationships
# Wolf | Conscientiousness | Person places emphasis on quality and accuracy, expertise, competency
# Shark | Dominance | Person places emphasis on accomplishing results, the bottom line, confidence
#  
# + Closed deals are won in order of cat, eagle, wolf and shark based on descending volume in all three dimensions.
# + In consideration of business context, it might make sense that conversion rate of wolf (accuracy-focused) or shark (result-focused) was lower than cat (cooperation-focused) or eagle (relationship-focused).
# 
# If so, sales performance could be improved by matching SDRs/SRs with MQLs properly.

# #### SDR/SR Performance by Behaviour Profile

# In[ ]:


# Create 'profile-SDR' table
cols = cd_profile['sdr_id'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_sdr = pd.pivot_table(cd_profile,
                             index='lead_behaviour_profile(upd)',
                             columns='sdr_id',
                             values='seller_id',
                             aggfunc='count',
                             fill_value=0)

profile_sdr = profile_sdr.reindex(index)[cols] # Sort SDR in descending order of volume 
profile_sdr


# In[ ]:


# Create 'profile-SR' table
cols = cd_profile['sr_id'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_sr = pd.pivot_table(cd_profile,
                            index='lead_behaviour_profile(upd)',
                            columns='sr_id',
                            values='seller_id',
                            aggfunc='count',
                            fill_value=0)

profile_sr = profile_sr.reindex(index)[cols] # Sort SR in descending order of volume
profile_sr


# In[ ]:


# Plot the two tables
figure, (ax1,ax2) = plt.subplots(2, 1, figsize=(20,14))
figure.subplots_adjust(hspace=0.2)

sns.heatmap(profile_sdr,
            annot=True,
            fmt='g',
            ax=ax1)
sns.heatmap(profile_sr,
            annot=True,
            fmt='g',
            ax=ax2)

ax1.set_title('SDR Performance in Descending Volume', fontsize=14)
ax2.set_title('SR Performance in Descending Volume', fontsize=14)
ax1.set_xticks([])
ax2.set_xticks([]);


# 1. SDR
#  + 1st and 3rd SDRs are eminent in handling cat.
#  + 2nd and 10th SDRs are specialized in eagle.
#  + 1st SDR is also unparalleled in dealing with wolf.
#  + As to shark 1st SDR is better than the others, but not enough to claim to be an expert.
#  + __SDR is the first contact point of MQL so they do not know the lead's behaviour profile yet. Therefore sharing top performers' expertise in cat, eagle or wolf can enhance team performance.__
#  + __In regard to shark, external resources may be helpful in building capability.__
# 
# 2. SR
#   + 1st SR has matchless skills in managing both cat and eagle.
#   + SRs on the first four places are good at handling wolf, but not as much as their highest performing fields.
#   + 2nd and 5th are the best performers in regard to shark, but hard to say 'proficient'.
#   + __Eagle can be assigned to 2nd SR. Further, spreading knowledge among the team can improve team performance.__
#   + __Like SDR, external knowledge sources can be a way to boost performance.__

# ## 3. Closed Deal Performance Overview
# ---
# In this part I will see the total revenue from closed deals after signing in and drill down the top revenue-generating segment.

# In[ ]:


# Load datasets
cd = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv')
order_items = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')
orders = pd.read_csv('../input/brazilian-ecommerce/olist_orders_dataset.csv',
                     parse_dates=['order_purchase_timestamp'])
products = pd.read_csv('../input/brazilian-ecommerce/olist_products_dataset.csv')
product_translation = pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')


# In[ ]:


print(cd.shape)
print(order_items.shape)
print(orders.shape)
print(products.shape)
print(product_translation.shape)


# In[ ]:


# Merge all of the data
data = pd.merge(cd,order_items,
                how='inner', on='seller_id')
data = pd.merge(data, orders,
                how='inner', on='order_id')
data = pd.merge(data, products,
                how='inner', on='product_id')
data = pd.merge(data, product_translation,
                how='left', on='product_category_name') # There are some data without english names
data.shape


# In[ ]:


# Sort out orders not devliered to customers
data = data[data['order_status'] == 'delivered']

# Add a 'year-month' column
data['order_purchase_timestamp(y-m)'] = data['order_purchase_timestamp'].dt.to_period('M')

print(data.shape)
data.head(3)


# ### Monthly Revenues by Business Segment
# _\* Revenue is calculated by summing up price_

# In[ ]:


cols = data.groupby(by='business_segment')            .price            .sum()            .sort_values(ascending=False)            .index

monthly_segment_revenue = data.groupby(['order_purchase_timestamp(y-m)', 'business_segment'])                               .price                               .sum()                               .unstack(level=1, fill_value=0)

monthly_segment_revenue = monthly_segment_revenue[cols]
monthly_segment_revenue


# In[ ]:


# Plot the monthly revenues by segment
monthly_segment_revenue.plot.area(figsize=(20,8))

plt.title('Monthly Revenues by Business Segment', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5));


# + Total revenues across 29 segments came in at 664,858 in the first eight months of 2018.
# + The biggest segment was 'watches', which generated 17.4% of total revenues (115,901).

# Let's dive into the 'watches' segment to see what we can learn.

# #### Watches Revenue by Product Category

# In[ ]:


# Create watches segment dataframe
watches = data[data.business_segment == 'watches']
watches.shape


# In[ ]:


# Create monthly revenues by product category
cols = watches.groupby('product_category_name_english')               .price               .sum()               .sort_values(ascending=False)               .index

monthly_revenue_category = watches.groupby(['order_purchase_timestamp(y-m)', 'product_category_name_english'])                                   .price                                   .sum()                                   .unstack(level=1, fill_value=0)

monthly_revenue_category = monthly_revenue_category[cols]
monthly_revenue_category


# In[ ]:


# Plot the monthly revenues by category
monthly_revenue_category.plot.area(figsize=(12,6))
plt.title('Monthly Revenues by Product Category of Watches', fontsize=14);


# + 'watches_gifts' category generated 79.7% of total revenue of segment.
# + 'watches_gifts' revenue soared in March and reached its peak in May. This category seems a seasonal item.
# + Except 'watches_gifts', product categories are irrelevant to watches segment.

# #### Watches Revenue by Seller

# In[ ]:


# Create 'seller - product category' table
cols = watches.groupby('product_category_name_english')               .price               .sum()               .sort_values(ascending=False)               .index

watches_seller_revenue = watches.groupby(['seller_id', 'product_category_name_english'])                                 .price                                 .sum()                                 .unstack(level=1, fill_value=0)

watches_seller_revenue = watches_seller_revenue[cols]
watches_seller_revenue['total'] = watches_seller_revenue.sum(axis=1)

watches_seller_revenue


# In[ ]:


# Create 'category - seller' table
index = watches.groupby('product_category_name_english')                .price                .sum()                .sort_values()                .index

seller_category_revenue = watches.groupby(['seller_id', 'product_category_name_english'])                                  .price                                  .sum()                                  .unstack(level=0, fill_value=0)                                  
seller_category_revenue = seller_category_revenue.reindex(index)
seller_category_revenue


# In[ ]:


# Plot the above table
seller_category_revenue.plot.barh(stacked=True, figsize=(12,6))

plt.title('Watches Revenue by Seller', fontsize=14)
plt.legend(loc='lower right');


# + Though 'watches' segment is the largest part of revenue, it has only two sellers.
# + Furthermore, the leading seller generated 97.0% of segment revenue.

# Let's uncover information about two sellers.

# In[ ]:


# Create seller info table
seller_ids = watches.groupby('seller_id').price.sum().index
fields = ['seller_id', 'won_date', 'business_segment', 'lead_type','business_type',]

data.loc[data['seller_id'].isin(seller_ids), fields]     .drop_duplicates(subset='seller_id', keep='first')


# + The leading seller is 'online big', perhaps a large internet-based company with high market share or strong brand awareness.
# + And its business type is 'reseller'. That explains why there are irrelevant product categories in 'watches' segment. The fact that a business segment may have unrelated product categories means revenue analysis should be conducted based on 'product category' rather than 'business segment'.

# Lastly I will address 'watches gifts' revenue by product to expand understanding of the category.

# #### Watches_gifts Revenue by Product

# In[ ]:


index = watches[watches['product_category_name_english'] == 'watches_gifts']                 .groupby('product_id')                 .price                 .sum()                 .sort_values(ascending=False)                 .index

product_seller_revenue = watches[watches['product_category_name_english'] == 'watches_gifts']                                   .groupby(['seller_id', 'product_id'])                                   .price                                   .sum()                                   .unstack(level=0, fill_value=0)

product_seller_revenue = product_seller_revenue.reindex(index)
product_seller_revenue.head(3)


# In[ ]:


product_seller_revenue.plot.bar(stacked=True, figsize=(20, 6))
plt.title('Product Revenues in Watches_gifts Category', fontsize=14);


# + With 46 items in 'watches gifts' category, the top selling product accounts for 24.4% and five best-selling items form 53.9% of category revenue.
# + 'watches_gifts' may be a relatively homogeneous market so securing popular items is more important than pursuing a broad range of products. It implies that a category leader should be acquired to boost category revenue.

# In[ ]:




