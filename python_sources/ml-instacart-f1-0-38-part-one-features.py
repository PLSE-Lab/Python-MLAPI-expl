#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel has been created by the [Information Systems Lab](http://islab.uom.gr) at the University of Macedonia, Greece for the needs of the elective course Special Topics of Information Systems I at the [Business Administration](http://www.uom.gr/index.php?tmima=2&categorymenu=2) department of the University of Macedonia, Greece.
#  <br>
# 
# # Objective
# ## 1.1 Objective
# The objective of this Kernel is to create a DataFrame that contains as index the user_id and product_id and next to it all the available features that we have created in our EDA Notebooks
# 
# To create this, we calculate the various predictor features (X) that will describe the characteristics of a product and the behaviour of a user regarding one or multiple products. We will do so by analysing the prior orders of the dataset (as we did in our EDA Notebooks). 
# 
# <img src="https://i.imgur.com/xuX3sGu.jpg" width="600">

# # 1. Import the required packages and datasets from Instacart
# Here the gc package will clean-up the reserved RAM memory from Python. This will eliminate our risk to exceed the 17.2 of available RAM, that Kaggle offers.

# In[ ]:


import pandas as pd               # for data manipulation
import numpy as np                # for aggregation functions
import gc                         # for clean-up memory


# Now we load all the orders (prior, train, test) with their metadata (order hour/day etc.).
# Furthermore, we load the order_products_prior which includes all the products that have been purchased in each prior order.

# In[ ]:


orders = pd.read_csv('../input/orders.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')

# Get the product tables also --> To create metrics for aisles / departments 
products = pd.read_csv('../input/products.csv')
products.product_name = products.product_name.str.replace(' ', '_').str.lower()
products.product_name = products.product_name.str.replace(',', '-').str.lower()


# # 2. Create a DataFrame with the orders and the products that have been purchased on prior orders
# We perform a left join on the orders DataFrame which contains all the order (prior, train, test). The final results will have all the orders and for these that are prior we will have such rows as the total products that have been purchased.

# In[ ]:


prd = orders.merge(order_products_prior, on='order_id', how='left')
prd.head()

# Optionally we can get the name of the products
#prd['product_name'] = products.product_name.str.replace(' ', '_').str.lower()
#prd['product_name'] = products.product_name.str.replace(',', '-').str.lower()
#TRIM DATASETS
#prd = prd.iloc[0:100000]
#prd = prd.iloc[0:15434766]
gc.collect()


# # 3. Create our final DataFrame - Main User X Products (uxp)
# In this DataFrame we will store our final features that will be calculated upon the prd DataFrame. The first feature that we create (so we can constuct the DataFrame) is the total times a user bought a product.

# In[ ]:


uxp = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
uxp.columns = ['uxp_total_bought']
uxp = uxp.reset_index()


# After the creation of this uxp, we can continue by including different features that describe the user, the product or the combination of them. <br>
# The uxp table can include of these features or a selection from them. <br>
# The next blocks can be executed as different modules.

# # 4. Create the (user X product) features

# In[ ]:


#Does users frequently reorder a product? (one-shot ratio)
item_one = uxp[uxp.uxp_total_bought==1].groupby('product_id')[['uxp_total_bought']].count()
item_one.columns = ['uxp_customers_one_shot']
item_size = uxp.groupby('product_id')[['user_id']].count()
item_size.columns = ['uxp_unique_customers']

userxproduct_var= item_one.merge(item_size, how='left', left_index=True, right_on='product_id')
userxproduct_var['one_shot_ratio_product'] = userxproduct_var.uxp_customers_one_shot / userxproduct_var.uxp_unique_customers
userxproduct_var = userxproduct_var.reset_index()


###########
uxp = uxp.merge(userxproduct_var[['product_id', 'one_shot_ratio_product']],how='left')
gc.collect()


# In[ ]:


#last 5 orders
prd['order_number_back'] = prd.groupby('user_id')['order_number'].transform(max) - prd.order_number +1 
prd5 = prd[prd.order_number_back <= 5]
last_five = prd5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five['times_last5_ratio'] = last_five.times_last5 / 5

#############
uxp = uxp.merge(last_five , on=['user_id', 'product_id'], how='left')
del [last_five, prd5]
gc.collect()


# In[ ]:


#How frequently a customer bought a product after its first purchase ?
times = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
total_orders = prd.groupby('user_id')[['order_number']].max()
total_orders.columns = ['total_orders']
first_order_number = prd.groupby(['user_id', 'product_id'])[['order_number']].min()
first_order_number.columns = ['first_order_number']
first_order_number_reset = first_order_number.reset_index()
span = pd.merge(total_orders, first_order_number_reset, on='user_id', how='right')
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
order_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
order_ratio['Order_Ratio_user_id_X_product_id'] = order_ratio.Times_Bought_N / order_ratio.Order_Range_D
del [times, total_orders, first_order_number, span]

###########
uxp = uxp.merge(order_ratio , on=['user_id', 'product_id'], how='left')
del order_ratio
gc.collect()


# # 5. Create the product features
# Features that are derived from the behaviour of all users towards a product

# In[ ]:


#Which products have the highest probability of being reordered?
product_var = prd.groupby('product_id')[['reordered']].mean()
product_var.columns = ['reorder_ratio']

#What is the average position of a product in an order?
product_var['mean_add_to_cart_order'] = prd.groupby('product_id')[['add_to_cart_order']].mean()
product_var.head()

##########
uxp = uxp.merge(product_var, on='product_id', how='left')
del product_var
gc.collect()


# In[ ]:


prd['user_max_onb'] = prd.groupby('user_id').order_number.transform(np.max)

from collections import defaultdict
item_cnt    = defaultdict(int)
item_chance = defaultdict(int)


# In[ ]:


pid_back = uid_back = onb_back = None

for user_id, product_id, order_number, max_onb in prd[['user_id', 'product_id', 'order_number', 'user_max_onb']].values:
        
    if user_id==uid_back and product_id==pid_back and (order_number-onb_back==1):
        item_cnt[product_id] +=1
    if order_number!=max_onb:
        item_chance[product_id] +=1
    
    uid_back = user_id
    pid_back = product_id
    onb_back = order_number
    
item_cnt = pd.DataFrame.from_dict(item_cnt, orient='index').reset_index()
item_cnt.columns = ['product_id', 'item_first_cnt']
item_chance = pd.DataFrame.from_dict(item_chance, orient='index').reset_index()
item_chance.columns = ['product_id', 'item_first_chance']
df = pd.merge(item_cnt, item_chance, on='product_id', how='outer').fillna(0)
df['item_first_ratio'] = df.item_first_cnt/df.item_first_chance


# In[ ]:


item_N2_cnt    = defaultdict(int)
item_N2_chance = defaultdict(int)
item_N3_cnt    = defaultdict(int)
item_N3_chance = defaultdict(int)
item_N4_cnt    = defaultdict(int)
item_N4_chance = defaultdict(int)
item_N5_cnt    = defaultdict(int)
item_N5_chance = defaultdict(int)

pid_back = uid_back = onb_back = None

for product_id, user_id, order_number, max_order_number in prd[['product_id', 'user_id', 'order_number','user_max_onb']].values:
        
    if product_id==pid_back and user_id==uid_back and (order_number-onb_back)<=2 and (max_order_number-order_number) >=2:
        item_N2_cnt[product_id] +=1
    if product_id==pid_back and user_id==uid_back and (max_order_number-order_number) >=2:
        item_N2_chance[product_id] +=1

    if product_id==pid_back and user_id==uid_back and (order_number-onb_back)<=3 and (max_order_number-order_number) >=3:
        item_N3_cnt[product_id] +=1
    if product_id==pid_back and user_id==uid_back and (max_order_number-order_number) >=3:
        item_N3_chance[product_id] +=1

    if product_id==pid_back and user_id==uid_back and (order_number-onb_back)<=4 and (max_order_number-order_number) >=4:
        item_N4_cnt[product_id] +=1
    if product_id==pid_back and user_id==uid_back and (max_order_number-order_number) >=4:
        item_N4_chance[product_id] +=1

    if product_id==pid_back and user_id==uid_back and (order_number-onb_back)<=5 and (max_order_number-order_number) >=5:
        item_N5_cnt[product_id] +=1
    if product_id==pid_back and user_id==uid_back and (max_order_number-order_number) >=5:
        item_N5_chance[product_id] +=1

    pid_back = product_id
    uid_back = user_id
    onb_back = order_number


# In[ ]:


item_N2_cnt = pd.DataFrame.from_dict(item_N2_cnt, orient='index').reset_index()
item_N2_cnt.columns = ['product_id', 'item_N2_cnt']
item_N2_chance = pd.DataFrame.from_dict(item_N2_chance, orient='index').reset_index()
item_N2_chance.columns = ['product_id', 'item_N2_chance']

item_N3_cnt = pd.DataFrame.from_dict(item_N3_cnt, orient='index').reset_index()
item_N3_cnt.columns = ['product_id', 'item_N3_cnt']
item_N3_chance = pd.DataFrame.from_dict(item_N3_chance, orient='index').reset_index()
item_N3_chance.columns = ['product_id', 'item_N3_chance']

item_N4_cnt = pd.DataFrame.from_dict(item_N4_cnt, orient='index').reset_index()
item_N4_cnt.columns = ['product_id', 'item_N4_cnt']
item_N4_chance = pd.DataFrame.from_dict(item_N4_chance, orient='index').reset_index()
item_N4_chance.columns = ['product_id', 'item_N4_chance']

item_N5_cnt = pd.DataFrame.from_dict(item_N5_cnt, orient='index').reset_index()
item_N5_cnt.columns = ['product_id', 'item_N5_cnt']
item_N5_chance = pd.DataFrame.from_dict(item_N5_chance, orient='index').reset_index()
item_N5_chance.columns = ['product_id', 'item_N5_chance']


df2 = pd.merge(item_N2_cnt, item_N2_chance, on='product_id', how='outer')
df3 = pd.merge(item_N3_cnt, item_N3_chance, on='product_id', how='outer')
df4 = pd.merge(item_N4_cnt, item_N4_chance, on='product_id', how='outer')
df5 = pd.merge(item_N5_cnt, item_N5_chance, on='product_id', how='outer')

df_2_3_4_5 = pd.merge(pd.merge(df2, df3, on='product_id', how='outer'),
              pd.merge(df4, df5, on='product_id', how='outer'), 
              on='product_id', how='outer').fillna(0)

df = df.merge(df_2_3_4_5, on='product_id', how='left')

df['item_N2_ratio'] = df['item_N2_cnt']/df['item_N2_chance']
df['item_N3_ratio'] = df['item_N3_cnt']/df['item_N3_chance']
df['item_N4_ratio'] = df['item_N4_cnt']/df['item_N4_chance']
df['item_N5_ratio'] = df['item_N5_cnt']/df['item_N5_chance']


# In[ ]:


df.fillna(0, inplace=True)
df.reset_index(drop=True, inplace=True)

df=df[['product_id', 'item_first_ratio', 'item_N2_ratio','item_N3_ratio', 'item_N4_ratio', 'item_N5_ratio' ]]

del [item_cnt, item_chance, item_N2_cnt, item_N2_chance ,item_N3_cnt ,item_N3_chance,  item_N4_cnt  ,item_N4_chance,  item_N5_cnt, item_N5_chance, df_2_3_4_5]
gc.collect()


############
uxp = uxp.merge(df, on='product_id', how='left')
del df
gc.collect()


# # 6. Create Aisle-Department features for each product
# - Derived from the products table

# In[ ]:


#Which aisle has the most products?
aisle_top = products.groupby('aisle_id')[['product_id']].count()
aisle_top.columns = ['total_products_aisle']
aisle_top.head()

prod_temp = products.merge(aisle_top, on='aisle_id', how='left')

dept_top = products.groupby('department_id')[['product_id']].count()
dept_top.columns = ['total_products_dept']
dept_top.head()

prod_temp = prod_temp.merge(dept_top, on='department_id', how='left')

prod_temp['total_products_aisle_ratio'] = prod_temp.total_products_aisle/total_products
prod_temp['total_products_dept_ratio'] = prod_temp.total_products_dept/total_products


#########
uxp = uxp.merge(prod_temp.drop(['aisle_id', 'department_id', 'product_name'],axis=1), on='product_id', how='left')
del prod_temp, products
gc.collect()


# # 7. Create User features

# In[ ]:


#ass 3
#Get the average, maximum & minimum order size for each customer.
order_size = prd.groupby(['user_id', 'order_id'])[['product_id']].count()
order_size.columns = ['size'] 
results = order_size.groupby('user_id')[['size']].mean()
results.columns = ['order_size_avg']   
results = results.reset_index()

##########
uxp = uxp.merge(results, on=['user_id'], how='left')
del [order_size, results]
gc.collect()


# # 8. Order Features (Not used on uxp table)
# These features cannot be included in the final table. We want features that describe the users and the products. Not each order!

# In[ ]:


'''#Find the size (number of products) of each order. (1)
order_var = order_products.groupby('order_id')[['product_id']].count()
order_var.columns= ['order_size']
order_var = order_var.reset_index()

#Find the number of orders for each basket size. (2)
size_results = order_var.groupby('order_size')[['order_size']].count()
size_results.columns = ['total_orders']

# merge (2) to (1)
order_var = order_var.merge(size_results, on='order_size', how='left')
order_var.head()

# How frequent an order has reordered products?
order_var = order_var.set_index('order_id')
               
order_var['reordered_ratio_order']=order_products.groupby('order_id')[['reordered']].mean()
order_var.head()

order_var = order_var.reset_index()

prd = prd.merge(order_var[['order_id', 'reordered_ratio_order']], on='order_id', how='left')

del [order_var , size_results]
gc.collect()'''


# # 9. Finalizing

# In[ ]:


uxp.info()


# In[ ]:


uxp.head(30)


# In[ ]:


#Convert to category
#prdunique = prd.iloc[:10000].nunique()

#max_ob = prdunique.values.max()
#for index,obs in enumerate(prdunique):
   # if obs < max_ob*0.10:
    #    prd.iloc[:,index]= prd.iloc[:,index].astype('category')


# # 10. Exporting to pickle and .csv files

# In[ ]:


uxp.to_pickle('uxp.pkl')
uxp.to_csv('uxp.csv')

