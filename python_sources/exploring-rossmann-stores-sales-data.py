#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# # EDA

# ## Read Data

# In[68]:


store = pd.read_csv("../input/store.csv")
train = pd.read_csv("../input/train.csv",index_col = "Date",parse_dates = ['Date'],low_memory=False)
test = pd.read_csv("../input/test.csv",index_col = "Date",parse_dates = ['Date'],low_memory=False)
sample_submission = pd.read_csv("../input/sample_submission.csv")


# ## Understanding the problem

# In[70]:


train.sample(5)


# In[4]:


store.sample(5)


# The stores index is one less than store id. To avoid data incosistency problem, we will make index as same as store id.

# In[5]:


store.index = store.Store


# In[6]:


print ("Shape of data set is ",train.shape)


# In[7]:


train.head(5).sort_values('Date')


# In[8]:


train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear
train['DayOfWeek'] = train.index.dayofweek
train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].fillna(0)
train['SalePerCustomer'].describe()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
year_records = train['Year'].value_counts()

sns.barplot(x=year_records.index,y=year_records.values)
plt.show()


# We will use the records for 2013 and 2014 to construct a model. Using the model, we will make a prediction for the year 2015.

# In[11]:


#train = train.loc[ (train.Year == 2013) | (train.Year == 2014)]
#test = train.loc[ (train.Year == 2015)]


# In[12]:


store[( store.Store == 322 )]


# In[13]:


store.sample(5)


# **Understanding Assortment**
# 
# A deep assortment of products means that a retailer carries a number of variations of a single product (the opposite of a narrow assortment); a wide variety of products means that a retailer carries a large number of different products (the opposite of a narrow variety).
# 
# 
# **Understanding Store Types**
# 
# There are 4 basic model of stores - a,b,c and d.

# ## Observation of a store data

# In[14]:


store_id = store.sample(1).Store.values
print ("Store id is ",store_id[0])

store_data = train[train['Store'] == store_id[0]]
print ("Number of entries of store ",store_id[0]," is ",store_data.shape[0])


# We remove the days when the store is closed. Those days will have 0 sales. We ignore those data points.

# In[15]:


plt.figure(figsize=(20,6))
plt.plot(store_data[store_data.Open == 1].Sales)
plt.title("Sales for store id {} excluding store closed days".format(store_id[0]))
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()


# During christmas, we can see a spike in sales. Let's make a time series decomposition of the above data.

# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = seasonal_decompose(store_data[store_data.Open == 1].Sales, model='multiplicative',freq=365)
plt.figure(figsize=(20,6))
fig = decomposition.plot()
plt.show()


# When repeated the above code for different stores, there were some common observations:
# 
# * On majority of the plots for trend component, the stores saw a dip in sales between late 2014 3rd quarter and 2014 4th quarter ( Between 2014-08 to 2014-09 ).
# 
# * The residual component graph is same for all stores.
# 
# * The seasonal component shows a spike in sales during Christmas time.

# ### Exploring State holidays
# There are four types in state holidays
# * 0 - In case it is not a state holiday
# * a - Public Holiday
# * b - Easter Holiday
# * c - Christmas Holiday

# In[17]:


sale_per_customer = train[train.Open == 1].groupby('StateHoliday')['SalePerCustomer'].sum() /                     train[train.Open == 1].groupby('StateHoliday')['SalePerCustomer'].count()
print (sale_per_customer)
plt.figure(figsize=(5,3))
sns.barplot(x=sale_per_customer.index,y=sale_per_customer.values)
plt.ylabel('Sale Per Customer')
plt.show()


# ## Analysis of sales by day of week
# Day 0 - Monday .  Day 1 - Tuesday .  Day 2 - Wednesday .  Day 3 - Thursday .
# Day 4 - Wednesday . Day 5 - Saturday .  Day 6 - Sunday .
# 
# 

# In[18]:


data_by_day_of_week = train[train.Open == 1].groupby('DayOfWeek').mean()


# In[19]:


mean_customer_count = data_by_day_of_week.Customers
mean_sales_value = data_by_day_of_week.Sales
mean_sale_per_customer = data_by_day_of_week.SalePerCustomer
promo = data_by_day_of_week.Promo

mean_sales_value = mean_sales_value.sort_values(ascending=True)
mean_customer_count = mean_customer_count.sort_values(ascending=True)
mean_sale_per_customer = mean_sale_per_customer.sort_values(ascending=True)
promo = promo.sort_values(ascending=True)

fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(20,11))
sns.barplot(x=mean_customer_count.index, y=mean_customer_count, ax=axs[0][0])
sns.barplot(x=mean_sales_value.index, y=mean_sales_value, ax=axs[0][1])
sns.barplot(x=mean_sale_per_customer.index, y=mean_sale_per_customer, ax=axs[1][0])
sns.barplot(x=promo.index, y=promo, ax=axs[1][1])

axs[0][0].set_title('Customer Count')
axs[0][1].set_title('Sales Value')
axs[1][0].set_title('Average Sales Per Customer')
axs[1][1].set_title('Promo')

axs[0][0].set_xlabel('')
axs[0][1].set_xlabel('')
axs[1][0].set_xlabel('')
axs[1][1].set_xlabel('')

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
axs[0][0].set_xticklabels(days,rotation = 10)
axs[0][1].set_xticklabels(days,rotation = 10)
axs[1][0].set_xticklabels(days,rotation = 10)
axs[1][1].set_xticklabels(days,rotation = 10)

plt.show()


# **Observation:**
# 
# Monday has got the least number of customer among weekdays but has higher average sale per customer than other days.
# 
# Other week days ( Tuesday - Friday ) have higher customer count than mondays but lower sales and average sales per customer. A point to note 56% of Monday have got promo's while for other week days, this percentage is lower. Increasing promo on other week days will generate more revenue for Rossmann.
# 
# Many store holidays falls on sunday. Of the 144730 sundays the data is available, the store is open only for 3593 of them. 2.4% of the sundays are working days. The sunday when the store is open would be festive days. To make more customers into revnue for Rossmann, promo on Sundays are recommended.

# ## Top 3 and Bottom 3 based on Sales, Customer Count and Sale Per Customer
# 
# ### Based on Sales

# To find total sales by a store, a store which has been open for more number of days wil have more sales when compared to a store which has been opened for less number of days. A store which has opened recently will have less total sales than a store which has opened a year back. To avoid the data incosistency problems, we will use total sales per day( total sales of store *i* / by number of days store *i* was open ).

# In[20]:


sales_by_store = train[['Sales','Store','Open']]
total_sales_by_store = sales_by_store.groupby('Store').sum()
total_sales_by_store['SalePerDay'] = total_sales_by_store['Sales']/total_sales_by_store['Open']
total_sales_by_store.sort_values(['SalePerDay'],inplace=True)

plt.figure(figsize=(20,5))
sns.boxplot(x=total_sales_by_store['SalePerDay'])
plt.title('Box Plot of stores sale per day')
plt.xlabel('Per Day Sales')
plt.show()


# 50% of values between 5322 ( 25th quartile ) and 7964 (75th quartile) with a mean sale per day pf 6589.9 and a standard deviation of 2383.91.

# In[21]:


#Making of top 3 and bottom 3
total_sales_by_store.sort_values(['SalePerDay'],inplace=True,ascending=True)
bottom_3 = total_sales_by_store[0:3]
total_sales_by_store.sort_values(['SalePerDay'],inplace=True,ascending=False)
top_3 = total_sales_by_store[0:3]
frames = [top_3, bottom_3]
top_3_bottom_3 = pd.concat(frames)

top_3_bottom_3['Store'] = top_3_bottom_3.index
top_3_bottom_3 = top_3_bottom_3.sort_values(['SalePerDay']).reset_index(drop=True)

#Plotting bar plot of top 3 and bottom 3
plt.figure(figsize=(8,6))
ax = sns.barplot(top_3_bottom_3.index, top_3_bottom_3.SalePerDay)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Store", ylabel='SalePerDay')
# adding proper x labels
ax.set_xticklabels(top_3_bottom_3.Store)
for item in ax.get_xticklabels(): 
    item.set_rotation(0)
for i, v in enumerate(top_3_bottom_3["SalePerDay"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(round(v[1],2)), color='m', va ='bottom', rotation=45)
plt.tight_layout()
plt.show()


# Stores with least sales are 307, 543, 198( in increasing order ).
# 
# Stores with highest sales 1114, 262, 817( in increasing order ).

# **Stores and competitor distance for the top 3 and bottom 3**

# In[22]:


store_and_competitor_distance = store.loc[top_3_bottom_3.Store]
store_and_competitor_distance = store_and_competitor_distance[['Store','CompetitionDistance']]

store_and_competitor_distance['Store'] = store_and_competitor_distance.index
store_and_competitor_distance = store_and_competitor_distance.sort_values(['CompetitionDistance']).reset_index(drop=True)

#Plotting bar plot of top 3 and bottom 3
plt.figure(figsize=(8,6))
ax = sns.barplot(store_and_competitor_distance.index, store_and_competitor_distance.CompetitionDistance)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="Store", ylabel='CompetitionDistance')
# adding proper x labels
ax.set_xticklabels(store_and_competitor_distance.Store)
for item in ax.get_xticklabels(): 
    item.set_rotation(0)
for i, v in enumerate(store_and_competitor_distance["CompetitionDistance"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(round(v[1],2)), color='m', va ='bottom', rotation=45)
plt.tight_layout()
plt.show()

Store 198 has the competitor at the farthest distance and it also has the third least sale per day.

Store 543 has got the competitor at the closest distance ( 250 ) and it has the second least sale  per day.

Store 307 has third least competitor distance and it has the least sale per day.

Store 817 which has highest sale per day also has competitor at second closes distance.

Let us make a correlation plot between competitor distance and daily sales to understand this a bit further.
# In[23]:


total_sales_by_store['Store'] = total_sales_by_store.index
total_sales_by_store = total_sales_by_store[["SalePerDay","Store"]]
store_and_competitor_distance = store[["Store","CompetitionDistance"]]
joined_df = pd.merge(store_and_competitor_distance, total_sales_by_store, on='Store', how='inner')
sns.jointplot(x="SalePerDay",y="CompetitionDistance",data=joined_df , kind="reg")
plt.show()


# In[24]:


corr = joined_df['SalePerDay'].corr(joined_df['CompetitionDistance'])
print ("Correlation between sale per day and competition distance is ",corr)


# It seems there is no significant correlation between sale per day and competitor distance. In fact, the correlation coefficient is negative (-0.04) and it is a insignificant value.

# In the following part, we will look at relationship between store type and sale per day, assortment and sale per day, promo and sale per day. We will also drop columns related to competitor since a competitor has no effects on sale per day of a store.
