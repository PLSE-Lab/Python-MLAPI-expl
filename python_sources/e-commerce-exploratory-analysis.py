#!/usr/bin/env python
# coding: utf-8

# <h2>Introduction</h2>
# 
# The Olist Brazilian Ecommerce Dataset has information about 100k purchases (orders) from 2016 to 2018. Prices are in Brazillian Real (BRL) and there are five tables in this dataset: payments, geolocation, customers, orders and classifications. The last one contains votes from three different analysts for around 3k reviews made by customers after they've received their products (through email satisfaction survey).
# 
# This notebook will go through all five tables with a quick analysis:
# 
# 1. Payments
# 2. Orders (purchases)
# 3. Classifications/Reviews
# 4. Geolocation
# 5. Customers

# In[ ]:


import numpy as np
import pandas as pd
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly library
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import colorlover as cl
# Others
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

# Load datasets
payments = pd.read_csv("../input/olist_public_dataset_v2_payments.csv")
orders = pd.read_csv("../input/olist_public_dataset_v2.csv")
reviews = pd.read_csv("../input/olist_classified_public_dataset.csv")
geo = pd.read_csv("../input/geolocation_olist_public_dataset.csv")
customers = pd.read_csv("../input/olist_public_dataset_v2_customers.csv")
translation = pd.read_csv("../input/product_category_name_translation.csv")


# <h2>1. Payments</h2>
# 
# Customers can pay with more than one payment method and therefore we have duplicate order ids in this frame. The sequential feature is used to indicate the payment method order.

# In[ ]:


payments['value_log'] = payments['value'].apply(lambda x: np.log(x) if x > 0 else 0)
unique_ = payments['order_id'].nunique()
print("DataFrame shape: {}; unique order ids: {}".format(payments.shape, unique_))
payments.head()


# <h3>Value distribution</h3>
# 
# Each row corresponds to a payment method used on some product order. As stated before, the customer can use more than a single payment method for each order.

# In[ ]:


def plot_dist(values, log_values, title, color="#D84E30"):
    fig, axis = plt.subplots(1, 2, figsize=(12,4))
    axis[0].set_title("{} - linear scale".format(title))
    axis[1].set_title("{} - logn scale".format(title))
    ax1 = sns.distplot(values, color=color, ax=axis[0])
    ax2 = sns.distplot(log_values, color=color, ax=axis[1])
log_value = payments.value.apply(lambda x: np.log(x) if x > 0 else 0)
plot_dist(payments.value, log_value, "Value distribution")


# In[ ]:


payments.describe()


# <h3>Payment method</h3>
# 
# There are four payment methods: credit card, debit card, boleto and voucher. Boleto is a brazillian payment method which is similar to a payment/bank slip.

# In[ ]:


method_count = payments['payment_type'].value_counts().to_frame().reset_index()
method_value = payments.groupby('payment_type')['value'].sum().to_frame().reset_index()
# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Number of payments (left) and Total payments value (right)", 
              height=400, width=800,)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)


# No surprises here: most payments are done with credit card (almost 75%) and another 20% with boleto.

# In[ ]:


ax = sns.catplot(x="payment_type", y="value",data=payments, aspect=2, height=3.8)


# Most orders are less than 200 BRL, but we have a wide range of values,  so its better to plot the natural log as before:

# In[ ]:


plt.figure(figsize=(10,4))
plt.title("Payments distributions - logn scale")
p1 = sns.kdeplot(payments[payments.payment_type == 'credit_card']['value_log'], color="navy", label='Credit card')
p2 = sns.kdeplot(payments[payments.payment_type == 'boleto']['value_log'], color="orange", label='Boleto')
p3 = sns.kdeplot(payments[payments.payment_type == 'voucher']['value_log'], color="green", label='Voucher')
p4 = sns.kdeplot(payments[payments.payment_type == 'debit_card']['value_log'], color="red", label='Debit card')


# <h3>Installments</h3>
# 
# Only credit cards can have more than one installment:

# In[ ]:


payments[payments['installments'] > 1]['payment_type'].value_counts().to_frame()


# When we plot the number of installments we can see some patterns. Most sellers in Brazil offer the option to divide the payment up to 10 installments and the mean product value increases until this number. Seven and nine installments are not usual.

# In[ ]:


ins_count = payments.groupby('installments').size()
ins_mean = payments.groupby('installments')['value'].mean()

trace0 = go.Bar(
    x=ins_count.index,
    y=ins_count.values,
    name='Number of orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=ins_mean.index,
    y=ins_mean.values,
    name='Mean value',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Number of installments',
                     legend=dict(orientation="h"))
iplot(fig)


# The next plot shows the number of orders (left) and the total value (right) for payments with a single installment. The distribution is quite difference here.

# In[ ]:


pay_one_inst = payments[payments['installments'] == 1]
method_count = pay_one_inst['payment_type'].value_counts().to_frame().reset_index()
method_value = pay_one_inst.groupby('payment_type')['value'].sum().to_frame().reset_index()
# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Orders and value for a single installment", 
              height=400, width=800,)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)


# <h2>2. Orders</h2>

# In[ ]:


unique_ = orders['order_id'].nunique()
print("DataFrame shape: {}; unique order ids: {}".format(orders.shape, unique_))
orders.head(3)


# There are rows with the same order_id when the customer buys more than one product.
# 
# Let's look at one example:

# In[ ]:


orders[orders['order_id'] == '000330af600103828257923c9aa98ae2']


# Here the costumer bought one Domestic Utilities product for 42 and a Foods and Beverage for 17.49. Timestamps, review and customer id are also the same for both rows. The next plot shows the number of orders with more than one product:

# In[ ]:


count_products = orders.groupby('order_id').size().value_counts()
trace = go.Bar(
    x= count_products.index,
    y= count_products.values,
    marker=dict(
        color=['rgba(204,204,204,1)', 'rgba(222,45,38,0.8)',
               'rgba(204,204,204,1)', 'rgba(204,204,204,1)',
               'rgba(204,204,204,1)']),
)
layout = go.Layout(title='Number of orders for number of products', height=420, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')


# <h3>Products and Freight value</h3>

# In[ ]:


# Products value
sum_value = orders.groupby('order_id')['order_products_value'].sum()
plot_dist(sum_value, np.log(sum_value), 'Products value')
# Freights value
sum_value = orders.groupby('order_id')['order_freight_value'].sum()
plot_dist(sum_value, sum_value.apply(lambda x: np.log(x) if x > 0 else 0), 'Freight value', color="#122aa5")


# <h3>Timeseries</h3>

# In[ ]:


# Product value by date
orders['datetime'] =  pd.to_datetime(orders['order_purchase_timestamp'])
value_date = orders.groupby([orders['datetime'].dt.date])['order_products_value'].sum()
freight_date = orders.groupby([orders['datetime'].dt.date])['order_freight_value'].sum()
# Plot timeseries
trace0 = go.Scatter(x=value_date.index.astype(str), y=value_date.values, opacity = 0.8, name='Product value')
trace1 = go.Scatter(x=freight_date.index.astype(str), y=freight_date.values, opacity = 0.8, name='Freight value')
layout = dict(
    title= "Product and freight value by date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=12, label='12m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)
fig = dict(data= [trace0, trace1], layout=layout)
iplot(fig)

# Sales for month
value_month = orders[['datetime', 'order_products_value']].copy()
value_month.set_index('datetime', inplace=True)
value_month = value_month.groupby(pd.Grouper(freq="M"))['order_products_value'].sum()
trace = go.Bar(x= value_month.index, y= value_month.values)
layout = go.Layout(title='Sales per month (product value)', height=420, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# Some inferences:
# * There is a huge spike in Nov 24 due to Black Friday
# * Sales are weak after Dec 20 (end-year holidays)
# * In 2016, there are some sales in october, but almost any in the following months
# * There is a spike in products value in Jul 18, but not in freigth

# <h3>Product category</h3>
# 
# There are 71 different categories with names in portuguese. The english names are avaliable in the translation csv file.

# In[ ]:


# Orders by category (less 1000 orders grouped into others)
orders_count = orders.groupby('product_category_name').size()
orders_count['others'] = orders_count[orders_count < 1000].sum()
orders_count = orders_count[orders_count >= 1000].sort_values(ascending=True)
orders_value = orders.groupby('product_category_name')['order_products_value'].sum()
orders_value = orders_value[orders_count.index]
translation = pd.Series(translation.product_category_name_english.values, index=translation.product_category_name)

trace0 = go.Bar(
    y=translation[orders_count.index],
    x=orders_count.values,
    name='Number of orders',
    orientation='h',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    y=translation[orders_value.index],
    x=orders_value.values,
    name='Total value',
    orientation='h',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(
    height=1000,
    width=800,
    title='Products category',
    margin=dict(l=150, r=10, t=100, b=100),
    legend=dict(orientation="h")
)
fig['layout']['xaxis1'].update(title='Orders by category', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='Products value by category', domain=[0.6, 1])
iplot(fig)


# <h3>Items and Sellers for order</h3>

# In[ ]:


items_count = orders.groupby('order_items_qty').size()
sellers_count = orders.groupby('order_sellers_qty').size()

trace0 = go.Bar(
    x=items_count.index,
    y=items_count.values,
    name='#Orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=sellers_count.index,
    y=sellers_count.values,
    name='#Orders',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Items and Sellers quantity')
fig['layout']['xaxis1'].update(title='Items quantity', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='Sellers quantity', domain=[0.6, 1])
iplot(fig)


# <h3>Product name and description</h3>

# In[ ]:


#product_name_lenght	product_description_lenght	product_photos_qty
fig, axis = plt.subplots(1, 2, figsize=(12,4))
axis[0].set_title("Produt name lenght")
axis[1].set_title("Product description lenght")
ax1 = sns.distplot(orders['product_name_lenght'], color="#D84E30", ax=axis[0]) #rgba(204,204,204,1)', 'rgba(222,45,38,0.8)'
ax2 = sns.distplot(orders['product_description_lenght'], color="#7E7270", ax=axis[1]) #"#D84E30"


# <h3>Number of photos</h3>

# In[ ]:


photo_qty = orders.groupby('product_photos_qty').size()
photo_value = orders.groupby('product_photos_qty')['order_products_value'].mean()
trace0 = go.Bar(
    x=photo_qty.index,
    y=photo_qty.values,
    name='Number of Orders',
    marker=dict(color='rgba(222,45,38,0.8)')
)
trace1 = go.Bar(
    x=photo_value.index,
    y=photo_value.values,
    name='Produt mean value',
    marker=dict(color='rgba(204,204,204, 0.8)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Photo quantity',
                     legend=dict(orientation="h"))
#fig['layout']['xaxis1'].update(title='photo quantity', domain=[0, 0.40])
#fig['layout']['xaxis2'].update(title='photo quantity', domain=[0.6, 1])
iplot(fig)


# <h3>Review score</h3>

# In[ ]:


review_qty = orders.groupby('review_score').size()
review_value = orders.groupby('review_score')['order_products_value'].mean()
trace0 = go.Bar(
    x=review_qty.index,
    y=review_qty.values,
    name='Number of orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=review_value.index,
    y=review_value.values,
    name='Produt mean value',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Review Score')
fig['layout']['xaxis1'].update(title='review score', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='review score', domain=[0.6, 1])
iplot(fig)


# Scores are usually good and the bad ones seems to be related to slightly more expensive products (or larger orders).

# <h3>Delivery time</h3>

# In[ ]:


# Convert columns to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_aproved_at'] = pd.to_datetime(orders['order_aproved_at'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
# Calculate differences in hours
orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_aproved_at']).dt.total_seconds() / 86400
orders['estimated_delivery_time'] = (orders['order_estimated_delivery_date'] - orders['order_aproved_at']).dt.total_seconds() / 86400
# Delivery estimated time and actual delivery time
plt.figure(figsize=(10,4))
plt.title("Delivery time in days")
ax1 = sns.kdeplot(orders['delivery_time'].dropna(), color="#D84E30", label='Delivery time')
ax2 = sns.kdeplot(orders['estimated_delivery_time'].dropna(), color="#7E7270", label='Estimated delivery time')


# There are some outliers for the delivery time. In the next plot the delivery time was limited to two months and the x-axis is the review score:

# In[ ]:


ax = sns.catplot(x="review_score", y="delivery_time", kind="box",
                 data=orders[orders.delivery_time < 60], height=4, aspect=1.5)


# <h2>3. Reviews</h2>

# In[ ]:


reviews.head(3)


# In[ ]:


class_voted = reviews.groupby('most_voted_class').size()
subclass_voted = reviews.groupby('most_voted_subclass').size()
trace0 = go.Bar(
    x=class_voted.index,
    y=subclass_voted.values,
    name='Number of reviews',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=subclass_voted.index,
    y=subclass_voted.values,
    name='Number of reviews',
    marker=dict(color='rgba(204,204,204, 0.8)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=810, title='Most voted class (left) and subclass (right)')
iplot(fig)


# In[ ]:


review_qty = orders.groupby('review_score').size()
review_value = orders.groupby('review_score')['order_products_value'].mean()

fig = tools.make_subplots(rows=4, cols=2, print_grid=False)
cols = ['votes_before_estimate', 'votes_delayed', 'votes_low_quality', 'votes_return',
        'votes_not_as_anounced', 'votes_partial_delivery', 'votes_other_delivery', 'votes_other_order',
        'votes_satisfied']
cols_color = ["#F97B40", "#DA6C38", "#BB5C30", "#9C4D28",
             "#7C3D20", "#5D2E18", "#3E1F10", "#1F0F08"]

col_index = 0
for i in range(4):
    for j in range(2):
        count_ = reviews.groupby(cols[col_index]).size()
        trace = go.Bar(
            x=count_.index,
            y=count_.values,
            name=cols[col_index],
            marker=dict(color=cols_color[col_index])
        )
        fig.append_trace(trace, i+1, j+1)
        col_index += 1

fig['layout'].update(height=900, width=800, title='Votes')
iplot(fig)


# <h2>4. Geolocation</h2>
# 
# This dataset has three times more rows

# In[ ]:


print("DataFrame shape:", geo.shape)
geo.head(3)


# <h3>State and City</h3>

# In[ ]:


count_state = geo['state'].value_counts()
count_state['others'] = count_state[count_state < 10000].sum()
trace = go.Bar(
    x= count_state[count_state >= 10000].index,
    y= count_state[count_state >= 10000].values,
    marker=dict(color='rgba(204,204,204, 0.9)')
)
layout = go.Layout(title='Number of rows for each state', height=400, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')


# In[ ]:


count_city = geo['city'].value_counts()
count_city['others'] = count_city[count_city < 3000].sum()
trace = go.Bar(
    x= count_city[count_city >= 3000].index,
    y= count_city[count_city >= 3000].values,
    marker=dict(color='rgba(222,45,38,0.8)')
)
layout = go.Layout(title='Number of rows for each city', height=400, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')


# <h2>5. Customers</h2>
# 
# The customer_id column in the orders dataset is unique for each order. To find if one customer has more than one order we need to use the customers table. Let's look at one example and see how it works.

# In[ ]:


unique_ = customers['customer_unique_id'].nunique()
print("DataFrame shape: {}; unique customers: {}".format(customers.shape, unique_))
customers.head(3)


# The following order is from january 2017:

# In[ ]:


orders[orders.customer_id == '109cf3ecc53afd27745a79a618cb5ec4']


# Looking for this customer_id (109cf...) in the customers table we can find the unique_customer_id:

# In[ ]:


customers[customers.customer_id == "109cf3ecc53afd27745a79a618cb5ec4"]


# This unique customer (b237...) has two customer_id:

# In[ ]:


customers[customers.customer_unique_id == 'b237307cd63e0bd318ec30a97ad25fce']


# The second customer id_points to a different order (from may 2018)

# In[ ]:


orders[orders.customer_id == 'f64b4d4b9e4185ce59b00e617e565bca']


# <h3>work in progress...</h3>
