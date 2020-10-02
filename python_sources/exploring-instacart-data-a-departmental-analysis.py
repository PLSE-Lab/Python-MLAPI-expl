#!/usr/bin/env python
# coding: utf-8

# ##Exploring Instacart data: A departmental analysis
# 
# This notebook will analyze the following three basic variables that vary across each department:
# 
#  - quantity of products purchased
#  - hour of day that products are purchased
#  - days since reorder
# 
# Disclaimer: Before performing this analysis I knew very little pandas, matplotlib, seaborn, and general python for that matter. I've always been interested in data and statistics so I just jumped straight in to see what happened and taught myself along the way. Some elements of the code may seem redundant, unnecessary, or inefficient to seasoned *panda-ers* and kagglers so just bare with me (bear with me?). **I'd also love feedback so I can improve!**
# 
# Lets do this!!

# In[ ]:


# imports and fill our 5 tables that instacart provided as dataframes into a dictionary called data
# note: only taking order_products__prior product purchases for this analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

files = ['aisles.csv','departments.csv', 'orders.csv','products.csv', 'order_products__prior.csv']
data = {}

for f in files:
    d = pd.read_csv('../input/{0}'.format(f))
    data[f.replace('.csv','')] = d
    
# rename each df for easier coding and readability
products= data['products']
order_products_prior = data['order_products__prior']
departments = data['departments']
orders = data['orders']
ailes = data['aisles']

OPPsample = order_products_prior.sample(n=3000000)

merged = products.merge(OPPsample,on='product_id',how='inner')
merged = departments.merge(merged,on='department_id',how='inner')
merged = orders.merge(merged,on='order_id',how='inner')

merged.head()


# I run a potato computer so that's why I only took a 3,000,000 sample of individual product purchases.
# 
# note: these purchases can come from any order so we wont have full orders in our set, this analysis is purely for individual purchase analysis
# 
# We can continue by separating our big merged table into separate departmental tables...

# In[ ]:


department_list = list(departments.department)

department_data = {}

for n in department_list:
    d = merged.loc[merged['department']=='{0}'.format(n)]
    department_data['{0}'.format(n)] = d
    
list(department_data)


# We now create a dictionary for the first part of our analysis that looks at products by volume of purchases
# 
#  - Take each department table in our **department_data** dictionary, group
#    it by **product_name** to get quantity purchased of each unique product
#  - clean it up

# In[ ]:


department_product_data = {}

for n in department_list:
    d = department_data['{0}'.format(n)].groupby(['product_name']).count().reset_index()
    department_product_data['{0}'.format(n)] = d
    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].iloc[:,0:2]
    department_product_data['{0}'.format(n)].columns = ['product_name','quantity']
    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].sort_values('quantity',ascending=False)
    department_product_data['{0}'.format(n)].reset_index(inplace=True)
    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].iloc[:,1:4]


# In[ ]:


# sanity check random table in department_product_data
department_product_data['babies'].head()


# For the next part of our analysis we will look at how **order_hour_of_day** varies by purchases in each department

# In[ ]:


# define the columns we are interested in from our big merged table and make a new merged table with only those columns
time_columns = ['order_id','order_hour_of_day','department','product_name']

time_merged = pd.DataFrame(merged,columns=time_columns)
time_merged.head()


# Next we create a dictionary for the second part of our analysis called **department_time_data** in a similar manner that we did before
# 
#  - group by **order_hour_of_day** to get quantity of products of ordered for
#    each hour of the day
#  - clean it up

# In[ ]:


columns=['Order hour of day','Quantity'] #columns to rename to

department_time_data = {}

for n in department_list:
    d = time_merged.loc[time_merged['department']=='{0}'.format(n)] # Insert data from time_merged table into our new dictionary per each department
    department_time_data['{0}'.format(n)] = d
    department_time_data['{0}'.format(n)] = department_time_data['{0}'.format(n)].groupby('order_hour_of_day').count()
    department_time_data['{0}'.format(n)].reset_index(inplace=True)
    department_time_data['{0}'.format(n)] = department_time_data['{0}'.format(n)][department_time_data['{0}'.format(n)].columns[0:2]]
    department_time_data['{0}'.format(n)].columns = columns


# In[ ]:


# sanity check random table in department_time_data
department_time_data['alcohol']


# before we make the the **days_since_reorder** dictionary, we need to address the NaN values in that column

# In[ ]:


orders.head()


# Let's see how many there are...

# In[ ]:


pd.isnull(orders['days_since_prior_order']).sum()


# Check the unique values in that column to make sure that all numbers 0-30 are accounted for before removing the NaN rows

# In[ ]:


# check unique values in that column
orders['days_since_prior_order'].unique()


# In[ ]:


orders = orders.dropna()


# Next, let's only take columns that we need for this analysis and put that data into a new table called **reorder_merged**

# In[ ]:


reorder_columns = ['order_id','product_id','days_since_prior_order','department']
reorder_merged = pd.DataFrame(merged,columns=reorder_columns)


# Okay.  Now lets do the same thing for the reorder analysis that we did for the previous variables this time grouping by **days_since_prior_order** which will give us the quantity of purchases that had an **order_id** with the same **days_since_prior_order** value
# 
# This one is quite a bit trickier to explain:
# 
# That is to say that our result will NOT represent the days since prior order for that specific product, rather it will represent the days since prior order for a specific USER's last order that ordered an item in their current order from that specified department.
# 
# make sense? Me neither.

# In[ ]:


columns2 = ['Days since prior order','Quantity'] # columns to rename to

department_reorder_data = {}

for n in department_list:
    d = reorder_merged.loc[reorder_merged['department']=='{0}'.format(n)]
    department_reorder_data['{0}'.format(n)] = d
    department_reorder_data['{0}'.format(n)] = department_reorder_data['{0}'.format(n)].groupby('days_since_prior_order').count()
    department_reorder_data['{0}'.format(n)].reset_index(inplace=True)
    department_reorder_data['{0}'.format(n)] = department_reorder_data['{0}'.format(n)][department_reorder_data['{0}'.format(n)].columns[0:2]]
    department_reorder_data['{0}'.format(n)].columns = columns2


# In[ ]:


# sanity check table in department_reorder_data
department_reorder_data['produce']


# Now that we have all of our data in the right format for all three modes of analysis, lets look at the first one: 
# 
#  - quantity of products purchased
# 
# We'll start by defining a function that plots the top ten products given a specific department

# In[ ]:


def toptenplot(name):
    p = sns.cubehelix_palette(10, start=0.6,dark=0.5, rot=1,light=0.8,reverse=True)
    plot = sns.barplot(palette = p,y='product_name',x='quantity',data=department_product_data['{0}'.format(name)].head(n=10))
    sns.plt.title('{0}'.format(name))
    plot.set(xlabel='Quantity',ylabel='Product Name')


# In[ ]:


toptenplot('produce')


# In[ ]:


toptenplot('dairy eggs')


# In[ ]:


toptenplot('alcohol')


# Now for the second part of our analysis: 
# 
#  - hour of day that products are purchased
# 
# Let's just stack all of the department tables in the **department_time_data** dictionary onto one line graph and see what we get

# In[ ]:


for n in department_list:
    sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_data['{0}'.format(n)],markers='',linestyles='-')


# Whoops... Lets normalize that to better compare the data to each other

# In[ ]:


department_time_norm = {}

for n in department_list:
    #calculate normalized quantity
    q = department_time_data['{0}'.format(n)]['Quantity']
    q_norm = (q-q.mean())/(q.max()-q.min())
    
    #copy "department_time_data" to "department_time_norm"
    d = department_time_data['{0}'.format(n)]
    department_time_norm['{0}'.format(n)] = d
    
    #replace the quantity with our new normalized quantity "q_norm"
    department_time_norm['{0}'.format(n)]['Quantity']=q_norm


# Okay lets set up our plot again. I've already colored the interesting ones...

# In[ ]:


paper_rc = {'lines.linewidth': 1}                  
sns.set_context("paper", rc = paper_rc)
plt.figure(figsize=(12, 6))
plt.ylabel('Normalized Quantity')

for n in department_list: 
    sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['{0}'.format(n)],markers='',linestyles='-')

sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['alcohol'],markers='',linestyles='-',color = 'g')
sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['babies'],markers='',linestyles='-',color = 'r')


# So there isn't a big significant difference accross the departments.  In general it seems that most of the purchasing happens between 9pm and 4pm.  There are two slight purchasing peaks at both ends of that timeframe, one around 9-10am and another around 3-4pm.  It looks like for most departments, the first peak(9-10am) is greater than the second peak(3-4pm) except for the alcohol department.  Lets look at the two departments that are colored: 
# 
#  - **babies** in red
#  - **alcohol** in green
# 
# The alcohol department seems to be most popular during the late night and early morning (12am - 6am) and early afternoon (1pm - 3pm) and least popular between 6am - 10am and 7pm - 11pm
# 
#  The babies department is highest in relative quantity purchased in the early morning (6am-9am) hours and late night hours (8pm-10pm) and dips the lowest in the middle of the day around 12pm and is also the relative lowest popular department between midnight and 5am.
# 
# The babies department also has relatively high variation throughout the day compared to the other departments
# 
# **Note**: To clarify what this graph truly represents, the **order_hour_of_day** isn't specifically for that individual product.  Since the **order_hour_of_day** is linked to the **order_id**, the x axis of this graph represents **order_hour_of_day** for the **order_id** that contained these individual product purchases
# 
# **Another note!!!** I repeated this notebook a few times to ensure that the insights from this dataset were consistent across multiple 3,000,000 samples

# Now onto the third part of our analysis:
# 
#  - days since reorder
# 
# Lets start by learning from the past and normalizing the **quantity** column so we can compare departments to each other on a single graph

# In[ ]:


department_reorder_norm = {}

for n in department_list:
    #calculate normalized quantity
    q = department_reorder_data['{0}'.format(n)]['Quantity']
    q_norm = (q-q.mean())/(q.max()-q.min())
    
    #copy "department_data" to "department_data_norm"
    d = department_reorder_data['{0}'.format(n)]
    department_reorder_norm['{0}'.format(n)] = d
    
    #replace the quantity with our new normalized quantity "q_norm"
    department_reorder_norm['{0}'.format(n)]['Quantity']=q_norm


# In[ ]:


# check random department
department_reorder_norm['produce']


# Good. Onto plotting...
# 
# Let's set our plot up again by overlaying each department.  Again, I've colored the interesting ones.
# 
# spoiler alert its alcohol and babies again...

# In[ ]:


paper_rc = {'lines.linewidth': 1}                  
sns.set_context("paper", rc = paper_rc)
plt.figure(figsize=(12, 6))
plt.ylabel('Normalized Quantity')

for n in department_list:
    sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['{0}'.format(n)],markers='',linestyles='-')
    
sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['alcohol'],markers='',linestyles='-',color = 'r')
sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['babies'],markers='',linestyles='-',color = 'g')


# As in the previous analysis, these insights are consistent across different 3,000,000 samples:
# 
#  - spike at 30 is due to the fact that the **days_since_prior_order**
#    column maximum value is 30. so the quantity values at 30 days since
#    prior order include purchases made greater than 30 days since prior
#    order
#  - we see spikes at 7, 14, and 21 indicating a tendency to reorder
#    products weekly, biweekly, and triweekly.
# 
# Oh look who it is again, alcohol and babies.
# 
#  - Of all the departments, the alcohol department has the highest
#    normalized quantity of purchases that are reordered within the 0-3
#    day range
#  - the same applies to the babies department within the 4-6 day range
#  - the alcohol department purchase quantity is generally the lowest past
#    the one week mark of days since reorder
#  - babies need diapers and other baby stuff. a lot of diapers and other baby stuff. This could explain the relative maximum reorder rate of the baby department between the 4-6 day period.
# 
# I want to be able to say the following:
# 
#  - This frequent reorder rate in the 0-3 day range suggests that products from the alcohol department are consumed on 
#  a more regular basis than any of the other products from other departments within this time range
# 
# **Or**
# 
#  - This graph shows that out of all departments, alcohol products are the most frequently reordered within 3 days of their prior order relative to other departments
# 
# However, I'm a little weary because of "days_since_prior_order" applies to orders and not products.  If anyone has a better way to explain this or a better way to do this I'd love to know!
# 
# **Update**
# 
# - Great comment from AlphaMeow concerning how to define **days_since_prior_order** that states:
# 
#  "Basically I grouped on users then products, worked out the time duration (cummulated days_since_prior_order?) that this person keeps purchases the item, say, banana, and how many banana purchases he made during this time. Then I averaged on all person purchases banana, to get a general banana purchase frequency."
# 
# Grouping by users first would be the right way to organize the data to gain insights from **days_since_prior_order** since what this column represents is more user-focused and not product-focused.
# 

# And that is it! Again, this is my first ever data analysis project and first time really getting into python and pandas.  I would love constructive feedback!
# 
# Thanks for reading through! :)
# 
#  -Josh

# In[ ]:




