#!/usr/bin/env python
# coding: utf-8

# # Analysing E Commerce Dataset

# The dataset is made public by Olist , the largest department store in Brazil . There are multiple datasets consisting of customer transaction details . The dataset provides a great opportunity to let us explore in different angles like 
# 
# * What is the average transaction value ?
# * Most bought customer categories 
# * Trend of sales over the day,week,month
# * Number of products boight by each customer
# * Delivery days ...
# 

#  Latest version changes - Version 9 - Uploaded the previous version of the Olist dataset ( private one) since I have done all my analysis on this and the current version uploaded by Olist team required a lot of changes.

# In[ ]:


## Loading the required libraries:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt
import calendar
from scipy.stats import skew,kurtosis

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from subprocess import check_output
print(check_output(["ls","../input/olist-v5-data"]).decode("utf8"))


# In[ ]:


## Reading the datas:
order = pd.read_csv("../input/olist-v5-data/olist_public_dataset_v2.csv")  ## Unclassified orders dataset
customer=pd.read_csv("../input/olist-v5-data/olist_public_dataset_v2_customers.csv")  ### Unique customer id 
payment =pd.read_csv("../input/olist-v5-data/payments_olist_public_dataset.csv")  ### Payment dataset
product = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")  ## Product translation to english
geo=pd.read_csv("../input/olist-v5-data/geolocation_olist_public_dataset.csv")  ## Location data
sellers=pd.read_csv("../input/olist-v5-data/sellers_olist_public_dataset_.csv") ## Seller information


# In[ ]:


order.shape


# In[ ]:


payment.shape


# In[ ]:


customer.shape


# In[ ]:


product.shape


# In[ ]:


geo.shape


# In[ ]:


## Joining the order and payment :
#order_pay=pd.merge(order,payment,how="left",on=['order_id','order_id'])
## Joining the order_payment with product category translation :
#order_product=pd.merge(order_pay,product,how="left",on=['product_category_name','product_category_name'])


# In[ ]:


#Now that we have joined the relevant tables,lets take a look at the data:


# ### Order Summary:

# In[ ]:


print("Total number of orders in the database:",order['order_id'].nunique())
print("Total Number of customers:",order['customer_id'].nunique())


# Lets check the order status

# In[ ]:


status=order.groupby('order_status')['order_id'].nunique().sort_values(ascending=False)


# In[ ]:


status


# A majority of the orders in the database are delivered products , while some are in transit .

# ### Order Value :

# Lets take a look at the order value column . We see that there are no currency value specified and we assume that it is in local currency unit -BRL . 

# In[ ]:


## Executive Summary:
print("Maximum order amount is BRL:",order['order_products_value'].max())
print("Minumum order amount is BRL:",order['order_products_value'].min())
print("Average order value is BRL:",order['order_products_value'].mean())
print("Median order value is BRL:",order['order_products_value'].median())


# We summarise the order with the help of order id and have a look at the order value and freight value .

# In[ ]:


value = order.groupby('order_id')['order_products_value','order_freight_value'].sum().sort_values(by='order_products_value',ascending=False).reset_index()


# The top 5 orders and their value is shown below:

# In[ ]:


value.head()


# Lets plot a histogram of the product value and freight value to understand the skewness of the data.

# In[ ]:


plt.figure(figsize=(12,10))

plt.subplot(221)
g = sns.distplot(np.log(order['order_products_value'] + 1))
g.set_title("Product Value of Orders - Distribution", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Frequency", fontsize=12)

plt.subplot(222)
g1 = sns.distplot(np.log(order['order_freight_value'] + 1))
g1.set_title("Freight Value of Orders - Distribution", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Frequency", fontsize=12)


# Both the distributions seems to follow a normal curve.Lets check the skewness and kurtosis.

# In[ ]:


print("Skewness of the transaction value:",skew(np.log(order['order_products_value']+1)))
print("Excess Kurtosis of the transaction value:",kurtosis(np.log(order['order_products_value']+1)))


# Skewness of the data suggests that the data is fairly symmetrical and positive whereas kurtosis value is  < 3.

# ### Number of products people usually order:

# In[ ]:


order_usual=order.groupby('order_id')['order_items_qty'].aggregate('sum').reset_index()
order_usual=order_usual['order_items_qty'].value_counts()
order_usual.head()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x=order_usual.index,y=order_usual.values,color="green")
ax.set_xlabel("Number of products added in order")
ax.set_ylabel("Number of orders")
ax.set_title("Number of products people usually order")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# Most of them have ordered only 1 product.The number of people ordering more than 2 items is very less.

# ### Most Bought product categories:

# In[ ]:


order_product=pd.merge(order,product,on='product_category_name',how='left')
order_product.shape


# In[ ]:


most_product=order_product.groupby('product_category_name_english').aggregate({'order_id':'count'}).rename(columns={'order_id':'order_count'}).sort_values(by='order_count',ascending=False).reset_index()
most_product.head()


# In[ ]:


### Visualising top 10 most bought product categories:
sns.barplot(x='product_category_name_english',y='order_count',data=most_product[:10],color="blue")
plt.xlabel("Product Category")
plt.ylabel("Total Number of orders")
plt.title("Most bought product categories")
plt.xticks(rotation='vertical')
plt.show()


# Bed Bath table,health beauty,sports are some categories that are bought most often by the customers.The difference in order count between bed bath table and health beauty is high.

# ### Order Trend:

# In[ ]:


order['order_purchase_timestamp']=pd.to_datetime(order['order_purchase_timestamp'])
order['order_delivered_customer_date']=pd.to_datetime(order['order_delivered_customer_date'])


# In[ ]:


## Create new columns for date,day,time,month:
order['weekday']=order['order_purchase_timestamp'].dt.weekday_name
order['year']=order['order_purchase_timestamp'].dt.year
order['monthday']=order['order_purchase_timestamp'].dt.day
order['weekday'] = order['order_purchase_timestamp'].dt.weekday
order['month']=order['order_purchase_timestamp'].dt.month
order['hour']=order['order_purchase_timestamp'].dt.hour


# In[ ]:


# Trend by Year:
trend_year=pd.DataFrame(order.groupby('year')['order_products_value'].sum().sort_values(ascending=False)).reset_index()
ax=sns.barplot(x='year',y='order_products_value',data=trend_year,palette=sns.set_palette(palette='viridis_r'))
#ax.ticklabel_format()
ax.set_xlabel('Year')
ax.set_ylabel('Total Transaction Value')
ax.set_title('Transaction Value by Year')


# The data is available from Sept 2016 to Sept 2018 . Therefore we see a very low sales for the year 2016 .Entire year data is available for 2017 and hence that year is on the higher side whereas sales till Sept for 2018 is plotted.For lack of entire data,we are unable to conclude any significant findings here.

# In[ ]:


## Boxplot for transactions by year:
plt.figure(figsize=(8,8))
ax=sns.boxplot(x='year',y='order_products_value',data=order,palette=sns.set_palette(palette='viridis_r'))
ax.set_xlabel('Year')
ax.set_ylabel('Total Value')
ax.set_title('Box Plot of transactions over the year')


# We find that most of the transaction fall below BRL 2000 . There are more outliers for the year 2017 ( since there is complete one year data for this).The maximum transaction value has been 14000 . 

# ### Average value of transactions per month:

# In[ ]:


## The below code is inspired from Sbans kernel -https://www.kaggle.com/shivamb/deep-exploration-of-gun-violence-in-us 
trend_month=pd.DataFrame(order.groupby('month').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'mean_transaction'})).reset_index()
x1 = trend_month.month.tolist()
y1 = trend_month.mean_transaction.tolist()
mapp = {}
for m,v in zip(x1, y1):
    mapp[m] = v
xn = [calendar.month_abbr[int(x)] for x in sorted(x1)]
vn = [mapp[x] for x in sorted(x1)]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=xn,y=vn, color='#ed5569')
ax.set_title("Average value of transaction per month")
ax.set_xlabel('Month')
ax.set_ylabel('Value')


# The average value of transaction is high during the month of October and September.

# ### Average value of transaction by day of the week:

# In[ ]:


trend_weekday=pd.DataFrame(order.groupby('weekday').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'Mean_Transaction'})).reset_index()


# In[ ]:


x2 = trend_weekday.index.tolist()
y2 = trend_weekday.Mean_Transaction.tolist()

weekmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
x2 = [weekmap[x] for x in x2]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = y2[j]
order_week = list(weekmap.values())
ordervals = [wkmp[val] for val in order_week]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=order_week,y=ordervals, color='#ed5569')
ax.set_title("Average value of transaction by day of the week")
ax.set_xlabel('Day')
ax.set_ylabel('Value')


# There seems to be not much trend observed during the day of the transaction . Lets check the frequency of the orders.

# In[ ]:


freq_weekday=pd.DataFrame(order.groupby('weekday').agg({'order_id':'count'}).rename(columns={'order_id':'order_count'})).reset_index()


# In[ ]:


x3 = freq_weekday.index.tolist()
y3 = freq_weekday.order_count.tolist()

weekmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
x3 = [weekmap[x] for x in x3]
wkmp = {}
for j,x in enumerate(x3):
    wkmp[x] = y3[j]
order_week = list(weekmap.values())
ordervals = [wkmp[val] for val in order_week]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=order_week,y=ordervals, palette=sns.color_palette(palette="Set2"))
ax.set_title("Total Number of orders by day of the week")
ax.set_xlabel('Day')
ax.set_ylabel('Value')


# The frequency of the orders has been higher on Mon,Tue whereas the freq of orders is low during Saturday and sundays.This means that during weekend people are not interested in online shopping going only by the frequency of the orders but combining this with the average value of transactions during the day there is a relatively high average value of transaction happening during saturdays compared to other days.

# In[ ]:


week=pd.merge(trend_weekday,freq_weekday,on='weekday',how='inner')


# In[ ]:


plt.figure(figsize=(8,8))
sns.jointplot(x='Mean_Transaction', y='order_count',data=week, size=10,color='red')
plt.ylabel('Order Count', fontsize=12)
plt.xlabel('Average value of transaction', fontsize=12)
plt.show()


# ### Order Trend by hour :

# In[ ]:


trend_hour=order.groupby('hour').agg({'order_id':'count'}).rename(columns={'order_id':'freq_order'}).reset_index()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x=trend_hour['hour'],y=trend_hour['freq_order'],color="red")
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Order Count')
ax.set_title("Frequency of transaction over the hour")


# From the plot we see that the frequency of the order steadly rises as the day progresses and reaches the peak after noon and continues till 22 hrs . There is a dip in the transaction during evening time between 18-19 hrs and it sees a rise after that . 

# ### Frequency of orders during the hour over the day:

# In[ ]:


day_hour=order.groupby(['weekday','hour']).agg({'order_id':'count'}).rename(columns={'order_id':'freq'}).reset_index()


# In[ ]:


day_hour.weekday=day_hour.weekday.map(weekmap)
day_hour.head()


# In[ ]:


### Sorting it so that the plot order is correct.
day_hour['weekday']=pd.Categorical(day_hour['weekday'],categories=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],ordered=True)


# In[ ]:


day_hour=day_hour.pivot('weekday','hour','freq')


# In[ ]:


plt.figure(figsize=(15,8))
ax=sns.heatmap(day_hour,annot=True,fmt="d",cmap="OrRd")
ax.set_xlabel("Hour")
ax.set_ylabel("Day")
ax.set_title("Heatmap of tranactions over the hour by day",size=10)


# * As the day progresses,the number of orders placed increases . 
# * There is clearly a difference in the order frequency between weekdays and weekends . 
# * While during weekdays , the order frequency increases steadly after 9 AM , the order frequency picks up only after 15:00 hrs during sundays . 

# ### City.State and Transaction Value :

# In[ ]:


trans_state=pd.DataFrame(order.groupby('customer_state').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'avg_trans'}).sort_values(by='avg_trans',ascending=False)).reset_index()


# In[ ]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='customer_state',y='avg_trans',data=trans_state,palette=sns.color_palette(palette="viridis_r"))
ax.set_xlabel('Customer State')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Avg transaction value')
ax.set_title("Average Transaction Value for each state")


# States Acre,Rondonia,Alagoas have a higher transaction value whereas Sao Paulo ,Minas Gerias have the lowest average transaction.

# In[ ]:


### By City :
trans_city=pd.DataFrame(order.groupby('customer_city').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'avg_trans'}).sort_values(by='avg_trans',ascending=False)).reset_index()


# In[ ]:


trans_city[:10]


# In[ ]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='customer_city',y='avg_trans',data=trans_city[:10],palette=sns.color_palette(palette="Set2"))
ax.set_xlabel('Customer City')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Avg transaction value')
ax.set_title("Top 10 - Average Transaction Value for each City")


# ### Order Delivery :

# Lets check the average number of days between order and delivery.

# In[ ]:


order['day_to_delivery']=(order['order_delivered_customer_date']-order['order_purchase_timestamp']).dt.days


# In[ ]:


print("Average days to delivery {}".format(np.round(order['day_to_delivery'].mean(),0)))


# Lets check the overall scenario.

# In[ ]:


delivery=order.groupby('day_to_delivery')['order_id'].aggregate({'order_id':'count'}).rename(columns={'order_id':'freq'}).reset_index().dropna()
delivery['freq']=delivery['freq'].astype(int)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x='day_to_delivery',y='freq',data=delivery,color="blue")
plt.title("Days to delivery")
plt.xlabel("Days")
plt.xticks(rotation="vertical")
plt.ylabel("Number of orders")
plt.show()


# A majority of the orders are getting delivered within a week whereas there were few orders that is taking over 1.5 months too .

# ### Payments :

# Lets check the mode of payments used for transaction

# In[ ]:


pay_type=payment.groupby('payment_type').aggregate({'order_id':'count'}).rename(columns={'order_id':'count'}).sort_values(by='count',ascending=False).reset_index()


# In[ ]:


pay_type['perc']=np.round((pay_type['count']/pay_type['count'].sum())*100,2)

plt.figure(figsize=(8,8))
ax=sns.barplot(x='payment_type',y='count',data=pay_type,color='cyan')
plt.title("Mode of Payment")
plt.xlabel('Payment Type')
plt.ylabel('Number of instances')


# A large number of online buyers use credit card their prefered mode of payment followed by boleto.According to [wiki](https://en.wikipedia.org/wiki/Boleto) ,boleto is a  a payment method in Brazil regulated by FEBRABAN, short for Brazilian Federation of Banks.A boleto can be paid at ATMs, branch facilities and internet banking of any Bank, Post Office, Lottery Agent and some supermarkets until its due date. After the due date it can only be paid at the issuer bank facilities.
# 
# Lets check the average value of transaction used for each type of payment.

# In[ ]:


print("Average value of transaction on credit card : BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='credit_card']['value'])))
print("Average value of transaction on boleto : BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='boleto']['value'])))
print("Average value of transaction on voucher: BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='voucher']['value'])))
print("Average value of transaction on debit card: BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='debit_card']['value'])))


# For each of the transaction types , the value at the quantiles is printed out for better interpretation . Distribution and box plot is also tried out for visualisation purpose.

# In[ ]:


print("Credit Card quantiles")
print(payment[payment.payment_type=='credit_card']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Boleto quantiles")
print(payment[payment.payment_type=='boleto']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Voucher quantiles")
print(payment[payment.payment_type=='voucher']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Debit Card quantiles")
print(payment[payment.payment_type=='debit_card']['value'].quantile([.01,.25,.5,.75,.99]))


# For transactions greater than BRL 100 , people use debit card,boleto or credit card . For transaction of higher value , people have used credit mode of payment followed by bolero and then debit card . The preference of using vouchers for transaction is on the lower side .Lets check the distribution of the transactions.

# In[ ]:


plt.figure(figsize=(10,8))
ax=sns.boxplot(x=payment.payment_type,y=payment.value,palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Boxplot for different payment type")
ax.set_xlabel("Transaction type")
ax.set_ylabel("Transaction Value")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[ ]:


payment=payment[payment['value']!=0]
plt.figure(figsize=(10,8))
plt.subplot(221)
ax=sns.distplot(np.log(payment[payment.payment_type=='credit_card']['value'])+1,color="red")
ax.set_xlabel("Log Transaction value (BRL)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution plot for credit card transactions")
plt.subplot(222)
ax1=sns.distplot(np.log(payment[payment.payment_type=='boleto']['value'])+1,color="red")
ax1.set_xlabel("Log Transaction value (BRL)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution plot for boleto transactions")
plt.subplot(223)
ax2=sns.distplot(np.log(payment[payment.payment_type=='debit_card']['value'])+1,color="red")
ax2.set_xlabel("Log Transaction value (BRL)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution plot for debit card transactions")
plt.subplot(224)
ax3=sns.distplot(np.log(payment[payment.payment_type=='voucher']['value'])+1,color="red")
ax3.set_xlabel("Log Transaction value (BRL)")
ax3.set_ylabel("Frequency")
ax3.set_title("Distribution plot for voucher transactions")


plt.subplots_adjust(wspace = 0.5, hspace = 0.5,
                    top = 1.3)

plt.show()


# From the distribution & box plot we understand the following
# 
# The distribution for credit card type of transaction is nearly normal and from the boxplot it is seen that there are extreme outliers in this case.This means that for higher value of transactions ,people prefer to buy on credit and pay later.
# Transaction through boleto is multimodal and seems to be for lesser value of BRL and here too the transactions are dominated by outliers.There seems to be a significant difference between credit card and boleto type of transactions .
# Debit card and vouchers are not used much and there are two modes dominating the distribution plot .

# ### Analysis on Sellers:

# In[ ]:


### Joining with the transaction data:
order_pay=pd.merge(order,sellers,how='left',on=['order_id','product_id'])
order_pay.shape


# #### Sellers who have sold most number of products:

# In[ ]:


plt.figure(figsize=(18,6))
ax=sns.barplot(order_pay['seller_id'].value_counts()[:15].index,order_pay['seller_id'].value_counts()[:15].values,palette='Set2')
ax.set_title('Top 15 sellers in Olist')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()


# Lets see the top 3 products sold by each of these sellers.
# 

# In[ ]:


top_15=order_pay.groupby('seller_id').apply(lambda x:x['product_category_name'].unique()).to_frame().reset_index()
top_15.columns=['seller_id','products']
top_15['product_count']=[len(c) for c in top_15['products']]
top_15.sort_values(by='product_count',ascending=False,inplace=True)


# In[ ]:


top_15.head(15)


# There are 21 products sold by one seller.Overall the top 15 sellers by product count have 13 to 21 products in their portfolio.

# ### Summary :
# The following are some of the key points noted from the Olist E commerce analysis.
# 
# * Maximum order amount is BRL 13440 and Minumum order amount is BRL 2.
# * Most of the time,the number of products ordered has always been < 3 .Bed Bath table,health beauty,sports are some categories that are bought most often by the customers.
# * The frequency of the orders has been higher on Mon,Tue whereas the frequency of orders is low during Saturday and sundays.
# * In a day,the number of transactions happening rises after 11 and continues till 22:00 Hrs.
# * Average time taken for delivery is a week and maximum has gone up to 1.5 months.
# * Credit card,boleto have been used for transactions of high value whereas people prefer using vouchers for low transaction values.
# * The anonymised seller data tells that the top 15 sellers with maximum portfolio of products have 13 to 21 products in their category .
