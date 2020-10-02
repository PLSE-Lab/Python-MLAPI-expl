#!/usr/bin/env python
# coding: utf-8

# # Brazilian Ecommerce Analysis

# This kernel has the objective of making an exploratory analysis of the datasets made available by Olist under the name of Brazilian E-Commerce Public Dataset by Olist.

# ## 1. Introduction

# #### 1.1 Importing libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd


# In[ ]:


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


# #### 1.2 Loading Datasets

# <b>Geolocation</b><br>
# According to the dataset description: "This dataset includes random latitudes and lengths from a given zip prefix code."

# In[ ]:


geolocation = pd.read_csv('../input/geolocation_olist_public_dataset.csv')
geolocation.head()


# <b>Classified Public Dataset </b><br>
# According to the dataset description:<br> 
# "This dataset was classified by three independent analysts. Each analyst voted for a class that he thinks a comment should belong to. After that we have classified a comment by choosing the most voted class.
# 
# <b>votes_before_estimate:</b> votes received for delivery before the estimated date messages.<br>
# <b>votes_delayed:</b> votes received for delayed complaints.<br>
# <b>votes_low_quality:</b> votes received for low product quality complaints.<br>
# <b>votes_return:</b> votes received for wishing to return product to seller complaints.<br>
# <b>votes_not_as_anounced:</b> votes received for product not as announced complaints.<br>
# <b>votes_partial_delivery:</b> votes received for partial delivery (not all products delivered) complaints.<br>
# <b>votes_other_delivery:</b> votes received for other delivery related complaints.<br>
# <b>votes_other_order:</b> votes received for other order related complaints.<br>
# <b>votes_satisfied:</b> votes received for customer satisfied messages.<br>
# <b>most_voted_subclass:</b> selects the most voted subclass for that comment<br>
# <b>most_voted_class:</b> aggregate subclasses into 3 class (satisfied, delivery complaints and quality complaints)<br>

# In[ ]:


public_classified = pd.read_csv('../input/olist_classified_public_dataset.csv')
public_classified.head()


# In[ ]:


# Columns of dataset
public_classified.columns


# <b>Unclassified Orders Dataset</b>

# In[ ]:


public_dataset = pd.read_csv('../input/olist_public_dataset_v2.csv')
public_dataset.head()


# In[ ]:


# Dataset Columns
public_dataset.columns


# <b> Customer Dataset </b>

# In[ ]:


customers = pd.read_csv('../input/olist_public_dataset_v2_customers.csv')
customers.head()


# <b> Payments Dataset </b>

# In[ ]:


payments = pd.read_csv('../input/olist_public_dataset_v2_payments.csv')
payments.head()


# ## 2. Data Exploration

# ### 2.1 Public Classified Analysis

# Here are the analyzes of the classification of the customers as to the quality of service provided. Basically they are divided into three classes called: Delivery Problems, Quality Problems and Satisfied with the request.<br>
# 
# Regarding the problems with the deliveries we can highlight the complaints for partial deliveries and for delay in delivery.<br>
# 
# Thus, when analyzing the problems with quality, it turns out that the main complaint is how much the difference with the advertised product.<br>
# 
# Finally, we have to customer satisfaction is around 58%, even considering the customers who responded as satisfied even before the delivery of the product.

# In[ ]:


public_classified.shape


# In[ ]:


types_class = public_classified.groupby('most_voted_class').sum()
types = types_class[['votes_before_estimate',
             'votes_delayed',
             'votes_low_quality',
             'votes_return',
             'votes_not_as_anounced',
             'votes_partial_delivery',
             'votes_other_delivery',
             'votes_other_order',
             'votes_satisfied']]


# In[ ]:


sns.set()
types.plot(kind='bar', stacked=True, figsize=(12,6))


# In[ ]:


type_class_city = public_classified.groupby(['most_voted_class']).agg({'most_voted_class':'count'})
type_class_city.plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')


# #### Type of Problems by State

# In[ ]:


k = public_classified.groupby(['customer_state','most_voted_class'])['most_voted_class']                        .size().groupby(level=0)                        .apply(lambda x: 100 * x / x.sum())                        .unstack()
k.fillna(0)

ax = k.plot(kind='bar',stacked=True, figsize=(20,10), title='% to Type of Response')
pl = ax.legend(bbox_to_anchor=(1.2, 0.5))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.1f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


# ### 2.3 Sales Analysis

# #### Total Sales by state

# In[ ]:


invoice_state = public_dataset.groupby(['customer_state']).sum()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))


# #### Sales Amount by State

# In[ ]:


invoice_state = public_dataset.groupby(['customer_state']).count()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))


# #### Median Sales by State

# In[ ]:


invoice_state = public_dataset.groupby(['customer_state']).median()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))


# #### Median Freight by State

# In[ ]:


invoice_state = public_dataset.groupby(['customer_state']).median()

#invoice_state['order_products_value'].plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
invoice_state['order_freight_value'].plot.bar(figsize=(16,8))


# ### 2.4 Payment Analysis

# #### Percentage by type of payment

# In[ ]:


type_payments = payments.groupby(['payment_type']).agg({'payment_type':'count'})
type_payments.plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')


# #### Merge the payment with the sales dataset

# In[ ]:


merge_payments = pd.merge(payments, public_dataset, on=['order_id'])


# #### Median Installments by State

# In[ ]:



result_payments = merge_payments.groupby(['customer_state']).median()

#invoice_state['order_products_value'].plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
result_payments['installments'].plot.bar(figsize=(16,8))


# #### Median values by type of payment and by state

# In[ ]:


k = merge_payments.groupby(['customer_state','payment_type'])['value'].median().unstack().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))


# #### Payment type by state

# In[ ]:



k = merge_payments.groupby(['customer_state','payment_type'])['payment_type']                        .size().groupby(level=0)                        .apply(lambda x: 100 * x / x.sum())                        .unstack()
k.fillna(0)

ax = k.plot(kind='bar',stacked=True, figsize=(20,10))
pl = ax.legend(bbox_to_anchor=(1.2, 0.5))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.1f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


# ### 2.5 Analysis of delivery times, payment confirmation and delays

# #### Difference between the times

# In[ ]:


order_p = pd.to_datetime(merge_payments['order_purchase_timestamp'], errors='coerce')
order_a = pd.to_datetime(merge_payments['order_aproved_at'], errors='coerce')
order_estimated_delivery = pd.to_datetime(merge_payments['order_estimated_delivery_date'], errors='coerce')
order_delivery = pd.to_datetime(merge_payments['order_delivered_customer_date'], errors='coerce')

# difference time in payment (in hours)
difference_time_payment = (order_a - order_p).astype('timedelta64[m]')
merge_payments['difference_time_payment'] = difference_time_payment

# difference time in estimetad delivery (in days)
difference_time_delivery = (order_delivery - order_estimated_delivery).astype('timedelta64[h]')/24
merge_payments['difference_time_delivery'] = difference_time_delivery

# difference time in aprovad to delivery
time_to_delivery = (order_delivery - order_a ).astype('timedelta64[h]')/24
merge_payments['time_to_delivery'] = time_to_delivery


# #### Median Payment Confirmation Time (in minutes)

# In[ ]:


k = merge_payments.groupby(['payment_type'])['difference_time_payment'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))


# #### Median of expected delivery date x actual delivery date

# In[ ]:


k = merge_payments.groupby(['customer_state'])['difference_time_delivery'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))


# #### Median Time to Delivery (in days) by State

# In[ ]:


k = merge_payments.groupby(['customer_state'])['time_to_delivery'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))


# ## 3. Conclusion

# This was an initial analysis of the data, as future work, it would be important to implement other intersections of important information such as the seasonal nature of sales, either total and / or state.

# ## See you soon!

# In[ ]:




